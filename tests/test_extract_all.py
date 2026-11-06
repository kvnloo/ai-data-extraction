import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import combine_conversations

from filter_privacy import discover_inputs, filter_jsonl


class ExtractAllTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.scripts = self.root / 'scripts'
        self.scripts.mkdir()
        source = Path(__file__).resolve().parents[1]
        for path in source.glob('*.py'):
            shutil.copy2(path, self.scripts / path.name)
        shutil.copy2(source / 'extract_all.sh', self.scripts / 'extract_all.sh')

        # A fresh child environment isolates every extractor and Git config.
        # Only synthetic fixtures under this temporary directory are discoverable.
        fixture_home = self.root / 'home'
        fixture_home.mkdir()
        self.hermes = fixture_home / '.hermes'
        self.environment = {
            'PATH': os.environ.get('PATH', os.defpath),
            'HOME': str(fixture_home),
            'USERPROFILE': str(fixture_home),
            'APPDATA': str(fixture_home / 'appdata'),
            'LOCALAPPDATA': str(fixture_home / 'localappdata'),
            'XDG_CONFIG_HOME': str(fixture_home / 'config'),
            'XDG_DATA_HOME': str(fixture_home / 'data'),
            'HERMES_HOME': str(self.hermes),
            'PI_CODING_AGENT_DIR': str(fixture_home / 'pi'),
            'OMP_CODING_AGENT_DIR': str(fixture_home / 'omp'),
            'OMP_HOME': str(fixture_home / 'omp'),
            'OMP_DATA_DIR': str(fixture_home / 'omp'),
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_CONFIG_SYSTEM': os.devnull,
            'GIT_CONFIG_GLOBAL': os.devnull,
            'GIT_AUTHOR_NAME': 'Fixture',
            'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
            'GIT_COMMITTER_NAME': 'Fixture',
            'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
            'GIT_AUTHOR_DATE': '2026-01-01T00:01:00+00:00',
            'GIT_COMMITTER_DATE': '2026-01-01T00:01:00+00:00',
        }

    def git(self, *arguments):
        result = subprocess.run(
            ['git', '-c', 'init.defaultBranch=main',
             '-c', f'core.hooksPath={self.root / "hooks"}',
             '-c', 'commit.gpgSign=false', *map(str, arguments)],
            cwd=self.root, env=self.environment, check=True,
            capture_output=True, text=True, encoding='utf-8',
        )
        return result.stdout.strip()

    def run_script(self, *repos):
        return subprocess.run(
            ['bash', 'extract_all.sh', *map(str, repos)],
            cwd=self.scripts, env=self.environment,
            capture_output=True, text=True, encoding='utf-8', timeout=60,
        )

    def rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip()]

    def make_database(self, repo):
        self.hermes.mkdir()
        database = self.hermes / 'state.db'
        connection = sqlite3.connect(database)
        self.addCleanup(connection.close)
        connection.executescript('''
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, model TEXT, title TEXT,
                started_at REAL, ended_at REAL, api_call_count INTEGER,
                input_tokens INTEGER, cache_read_tokens INTEGER,
                cache_write_tokens INTEGER, output_tokens INTEGER,
                tool_call_count INTEGER, parent_session_id TEXT,
                cwd TEXT, git_repo_root TEXT
            );
            CREATE TABLE messages (
                session_id TEXT, role TEXT, content TEXT, tool_calls TEXT,
                tool_name TEXT, timestamp REAL
            );
        ''')
        connection.execute(
            'INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('fixture-session', 'cli', 'fixture-model', 'Fixture extraction',
             1767225600, 1767225720, 2, 30, 7, 2, 5, 1, None,
             str(repo), str(repo)),
        )
        connection.executemany(
            'INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)',
            [('fixture-session', 'user', 'Update the fixture.', None, None, 1767225600),
             ('fixture-session', 'assistant', 'Updated.', None, None, 1767225720)],
        )
        connection.commit()
        connection.close()
        return database

    def test_runs_real_extractors_and_links_without_recombining_old_outputs(self):
        repo = self.root / 'repository with spaces'
        self.git('init', '-q', repo)
        (repo / 'fixture.txt').write_text('fixture\n', encoding='utf-8')
        self.git('-C', repo, 'add', '--', 'fixture.txt')
        self.git('-C', repo, 'commit', '-qm', 'Fix #17')
        sha = self.git('-C', repo, 'rev-parse', 'HEAD')
        database = self.make_database(repo)
        original_database = database.read_bytes()

        output = self.scripts / 'extracted_data'
        output.mkdir()
        stale = output / 'ALL_CONVERSATIONS_19990101_000000.jsonl'
        stale.write_text(json.dumps({
            'session_id': 'stale-combined', 'messages': [], 'source': 'hermes',
        }) + '\n', encoding='utf-8')

        previous_snapshot = output / 'hermes_conversations_19990101_000001.jsonl'
        previous_snapshot.write_text(json.dumps({
            'session_id': 'fixture-session', 'source': 'hermes',
            'source_file': str(database.resolve()), 'metadata': {'profile': 'default'},
            'messages': [{'role': 'assistant', 'content': 'obsolete snapshot'}],
        }) + '\n', encoding='utf-8')

        for _ in range(2):
            result = self.run_script(repo)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            combined = sorted(output.glob('ALL_CONVERSATIONS_*.jsonl'))[-1]
            conversations = self.rows(combined)
            self.assertEqual(len(conversations), 1)
            self.assertNotIn('obsolete snapshot', json.dumps(conversations))
            self.assertEqual({row['session_id'] for row in conversations}, {'fixture-session'})
            self.assertTrue(all(row['source'] == 'hermes' and 'messages' in row
                                for row in conversations))
            self.assertTrue(all('sha' not in row for row in conversations))
            commits_path = sorted((output / 'git').glob('git_commits_*.jsonl'))[-1]
            links_path = sorted((output / 'git').glob('git_session_links_*.jsonl'))[-1]
            commits = self.rows(commits_path)
            links = self.rows(links_path)
            self.assertEqual([commit['sha'] for commit in commits], [sha])
            self.assertEqual(commits[0]['files_changed'], ['fixture.txt'])
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]['commit_sha'], sha)
            self.assertEqual(links[0]['session_id'], 'fixture-session')

        self.assertEqual(database.read_bytes(), original_database)
        self.assertEqual(self.git('-C', repo, 'rev-parse', 'HEAD'), sha)
        discovered = {path for path, _ in discover_inputs([output])}
        self.assertIn(commits_path, discovered)
        self.assertIn(links_path, discovered)
        filtered = self.root / 'filtered' / commits_path.name
        stats = filter_jsonl(commits_path, filtered, lambda value: value.replace('Fixture', '<REDACTED>'))
        self.assertEqual(stats.records, 1)
        self.assertGreater(stats.changed_strings, 0)
        self.assertNotIn('Fixture', filtered.read_text(encoding='utf-8'))


    def test_cursor_agent_stores_survive_aggregation_and_snapshot_refresh(self):
        cursor_root = Path(self.environment['HOME']) / '.cursor' / 'chats'

        def write_store(agent_id, reply):
            database = cursor_root / 'shared-chat' / agent_id / 'store.db'
            database.parent.mkdir(parents=True, exist_ok=True)
            messages = [
                {'role': 'user', 'content': 'Update the fixture.'},
                {'role': 'assistant', 'content': reply},
            ]
            blobs = []
            for message in messages:
                data = json.dumps(message).encode('utf-8')
                blobs.append((hashlib.sha256(data).hexdigest(), data))
            root_data = b''.join(
                b'\x0a\x20' + bytes.fromhex(blob_id) for blob_id, _ in blobs
            )
            root_id = hashlib.sha256(root_data).hexdigest()
            metadata = {
                'agentId': agent_id,
                'latestRootBlobId': root_id,
                'createdAt': 1767225600000,
                'name': 'Fixture conversation',
                'mode': 'agent',
            }
            connection = sqlite3.connect(database)
            try:
                with connection:
                    connection.executescript('''
                        CREATE TABLE IF NOT EXISTS blobs
                            (id TEXT PRIMARY KEY, data BLOB);
                        CREATE TABLE IF NOT EXISTS meta (value TEXT);
                        DELETE FROM blobs;
                        DELETE FROM meta;
                    ''')
                    connection.executemany(
                        'INSERT INTO blobs VALUES (?, ?)',
                        blobs + [(root_id, root_data)],
                    )
                    connection.execute(
                        'INSERT INTO meta VALUES (?)',
                        (json.dumps(metadata).encode('utf-8').hex(),),
                    )
            finally:
                connection.close()
            return database

        agent_a = write_store('agent-a', 'first reply from A')
        agent_b = write_store('agent-b', 'reply from B')
        output = self.scripts / 'extracted_data'

        for reply_a in ('first reply from A', 'updated reply from A'):
            with self.subTest(reply_a=reply_a):
                if reply_a == 'updated reply from A':
                    write_store('agent-a', reply_a)
                before = {path: path.read_bytes() for path in (agent_a, agent_b)}
                result = self.run_script()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                raw_path = sorted(output.glob('cursor_cli_*.jsonl'))[-1]
                raw = self.rows(raw_path)
                combined_path = sorted(output.glob('ALL_CONVERSATIONS_*.jsonl'))[-1]
                combined = self.rows(combined_path)
                self.assertEqual(len(raw), 2)
                self.assertEqual(len(combined), 2)
                self.assertCountEqual(combined, raw)
                self.assertEqual(
                    {(row['chat_id'], row['agent_id']) for row in combined},
                    {('shared-chat', 'agent-a'), ('shared-chat', 'agent-b')},
                )
                self.assertEqual(
                    {row['agent_id']: row['messages'][-1]['content']
                     for row in combined},
                    {'agent-a': reply_a, 'agent-b': 'reply from B'},
                )
                self.assertEqual(
                    {path: path.read_bytes() for path in before}, before,
                )
                if reply_a == 'first reply from A':
                    # Keep an old export even if both runs share a filename second.
                    raw_path.rename(output / 'cursor_cli_19990101_000000.jsonl')

    def test_empty_fixture_home_has_no_conversations_or_git_scan(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.scripts / 'extracted_data'
        combined = list(output.glob('ALL_CONVERSATIONS_*.jsonl'))
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0].read_text(encoding='utf-8'), '')
        self.assertFalse((output / 'git').exists())
        self.assertIn('Git extraction skipped', result.stdout)

    def test_invalid_git_repository_fails_the_aggregate_command(self):
        result = self.run_script(self.root / 'missing-repository')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Git extraction failed', result.stderr)
        self.assertNotIn('COMPLETE!', result.stdout)


class CombineSnapshotsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def snapshot(self, name, rows):
        path = self.root / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return path

    def conversation(self, **changes):
        return {
            'source': 'hermes', 'session_id': 'shared',
            'source_file': str(self.root / 'state.db'),
            'metadata': {'profile': 'default'},
            'messages': [{'role': 'assistant', 'content': 'old'}],
            **changes,
        }

    def test_latest_snapshot_replaces_session_without_repeating_old_usage(self):
        old = self.conversation(metadata={'profile': 'default', 'input_tokens': 100})
        retained = self.conversation(session_id='retained')
        new = self.conversation(
            metadata={'profile': 'default', 'input_tokens': 123},
            messages=[{'role': 'assistant', 'content': 'new', 'input_tokens': 23}],
        )
        yesterday = self.snapshot('hermes_conversations_20260101_100000.jsonl', [old, retained])
        today = self.snapshot('hermes_conversations_20260101_100000_000001.jsonl', [new])
        # A copied old file must not outrank the newer exporter timestamp.
        os.utime(yesterday, (2000000000, 2000000000))
        before = {path: path.read_bytes() for path in (yesterday, today)}
        rows = combine_conversations.latest_conversations([today, yesterday, today])
        self.assertEqual(len(rows), 2)
        by_id = {row['session_id']: row for row in rows}
        self.assertEqual(by_id['shared'], new)
        self.assertEqual(by_id['retained'], retained)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_source_profile_database_and_branch_identity_stay_distinct(self):
        variants = [
            self.conversation(),
            self.conversation(metadata={'profile': 'review'}),
            self.conversation(source_file=str(self.root / 'other' / 'state.db')),
            self.conversation(branch_id='leaf-a'),
            self.conversation(branch_id='leaf-b'),
            self.conversation(source='omp'),
        ]
        old = self.snapshot('snapshots_20260101_000000.jsonl', variants)
        newest = self.conversation(messages=[{'role': 'assistant', 'content': 'updated'}])
        new = self.snapshot('snapshots_20260102_000000.jsonl', [
            newest, variants[3], variants[4],
        ])
        rows = combine_conversations.latest_conversations([new, old])
        self.assertEqual(len(rows), len(variants))
        self.assertIn(newest, rows)
        for variant in variants[1:]:
            self.assertIn(variant, rows)
        self.assertNotIn(variants[0], rows)

    def test_newest_session_snapshot_replaces_old_leaf_ids_and_keeps_current_forks(self):
        first_leaf = self.conversation(source='pi', branch_id='old-leaf')
        old = self.snapshot('pi_conversations_20260101_000000.jsonl', [first_leaf])
        active = self.conversation(
            source='pi', branch_id='new-leaf',
            messages=[{'role': 'assistant', 'content': 'continued branch'}],
        )
        fork = self.conversation(source='pi', branch_id='other-fork')
        new = self.snapshot('pi_conversations_20260102_000000.jsonl', [active, fork])
        rows = combine_conversations.latest_conversations([new, old])
        self.assertEqual(rows, [active, fork])
        self.assertNotIn(first_leaf, rows)

    def test_unknown_identity_is_retained_and_side_tables_and_combined_are_excluded(self):
        unknown = {'source': 'cursor-aiservice', 'messages': []}
        snapshot = self.snapshot('snapshot_20260101_000000.jsonl', [
            unknown, unknown, {'source': 'git', 'sha': 'fixture'},
        ])
        previous = self.snapshot('ALL_CONVERSATIONS_20260102_000000.jsonl', [
            self.conversation(session_id='previous-combined'),
        ])
        self.assertEqual(
            combine_conversations.latest_conversations([snapshot, previous]),
            [unknown, unknown],
        )


    def test_cursor_composite_identity_preserves_other_agents_and_unknown_rows(self):
        def cursor(chat_id, agent_id, content):
            return {
                'source': 'cursor-agent-cli',
                'chat_id': chat_id,
                'agent_id': agent_id,
                'messages': [{'role': 'assistant', 'content': content}],
            }

        # These pairs collide if components are joined with a bare colon.
        old_a = cursor('shared:a', 'b', 'old A')
        retained_b = cursor('shared', 'a:b', 'retained B')
        new_a = cursor('shared:a', 'b', 'new A')
        unknown = [
            cursor('unidentified', None, 'missing agent field'),
            cursor('unidentified', None, 'null agent'),
            cursor('unidentified', '', 'empty agent'),
        ]
        del unknown[0]['agent_id']

        old_path = self.snapshot(
            'cursor_cli_20260101_000000.jsonl',
            [old_a, retained_b, *unknown],
        )
        new_path = self.snapshot(
            'cursor_cli_20260102_000000.jsonl', [new_a],
        )
        before = {path: path.read_bytes() for path in (old_path, new_path)}
        rows = combine_conversations.latest_conversations(
            [new_path, old_path, new_path],
        )

        self.assertEqual(len(rows), 5)
        self.assertCountEqual(rows, [new_a, retained_b, *unknown])
        self.assertNotIn(old_a, rows)
        self.assertEqual(
            {path: path.read_bytes() for path in before}, before,
        )

    def test_invalid_snapshot_does_not_replace_combined_output(self):
        invalid = self.root / 'snapshot_20260101_000000.jsonl'
        invalid.write_text('{"messages": []}\ninvalid\n', encoding='utf-8')
        combined = self.root / 'combined.jsonl'
        combined.write_text('existing output\n', encoding='utf-8')
        with patch('sys.stderr', new_callable=io.StringIO) as errors:
            result = combine_conversations.main([
                '--output', str(combined), str(invalid),
            ])
        self.assertEqual(result, 1)
        self.assertIn('line 2', errors.getvalue())
        self.assertEqual(combined.read_text(encoding='utf-8'), 'existing output\n')


if __name__ == '__main__':
    unittest.main()
