import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

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

        for _ in range(2):
            result = self.run_script(repo)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            combined = sorted(output.glob('ALL_CONVERSATIONS_*.jsonl'))[-1]
            conversations = self.rows(combined)
            self.assertTrue(conversations)
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


if __name__ == '__main__':
    unittest.main()
