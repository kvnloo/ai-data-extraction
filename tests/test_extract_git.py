from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import extract_git
import extract_pi
from filter_privacy import discover_inputs, filter_jsonl


class ExtractGitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / 'home'
        self.home.mkdir()
        self.template = self.root / 'empty-template'
        self.template.mkdir()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('GIT_')}
        env.update({
            'HOME': str(self.home), 'XDG_CONFIG_HOME': str(self.home / 'config'),
            'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
            'GIT_AUTHOR_NAME': 'Fixture', 'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
            'GIT_COMMITTER_NAME': 'Fixture', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
        })
        self.environment = patch.dict(os.environ, env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        home_guard = patch.object(Path, 'home', side_effect=AssertionError('No home discovery'))
        home_guard.start()
        self.addCleanup(home_guard.stop)
        self.repo = self.init_repo(self.root / 'repo with spaces')

    def git(self, repository, *arguments, **environment):
        result = subprocess.run(
            ['git', '-c', f'core.hooksPath={self.template}',
             '-c', 'commit.gpgSign=false', '-C', str(repository), *arguments],
            env={**os.environ, **environment}, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.decode('utf-8').strip()

    def init_repo(self, path):
        path.mkdir(parents=True)
        self.git(path, 'init', '--quiet', '--initial-branch=main',
                 f'--template={self.template}')
        return path

    def commit(self, repository=None, subject='Implement #12',
               time='2026-01-01T10:01:00+00:00', author_time=None):
        repository = repository or self.repo
        self.git(repository, 'add', '-A', '--')
        self.git(repository, 'commit', '--quiet', '--allow-empty', '-m', subject,
                 GIT_AUTHOR_DATE=author_time or time, GIT_COMMITTER_DATE=time)
        return self.git(repository, 'rev-parse', 'HEAD')

    def session(self, session_id='session-1', **extra):
        return {
            'session_id': session_id, 'source': 'codex', 'cwd': str(self.repo / 'src'),
            'timestamp': '2026-01-01T10:00:00Z',
            'messages': [{'role': 'assistant', 'content': 'Complete.',
                          'timestamp': '2026-01-01T10:02:00Z'}],
            **extra,
        }

    def test_metadata_root_files_issue_refs_and_read_only_git(self):
        names = ['ordinary.txt', 'space name.txt', 'tab\tline\nname.txt', 'unicode-λ.txt']
        for name in names:
            (self.repo / name).write_text('fixture\n', encoding='utf-8')
        sha = self.commit(subject='Implement #12\n\nRefs example/repo#34 and '
                                  'https://github.com/example/repo/pull/56',
                          author_time='2026-01-01T09:00:00+00:00')
        before = {name: (self.repo / '.git' / name).read_bytes()
                  for name in ('HEAD', 'index', 'config', 'refs/heads/main')}
        (self.repo / 'src').mkdir()
        records = extract_git.extract_repositories([self.repo, self.repo / 'src', self.repo])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['sha'], sha)
        self.assertEqual(record['author'], 'Fixture')
        self.assertEqual(extract_git.parse_timestamp(record['time']),
                         datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc).timestamp())
        self.assertEqual(extract_git.parse_timestamp(record['author_time']),
                         datetime(2026, 1, 1, 9, tzinfo=timezone.utc).timestamp())
        self.assertEqual(record['subject'], 'Implement #12')
        self.assertEqual(record['files_changed'], sorted(names))
        self.assertEqual(record['issue_refs'],
                         ['#12', 'example/repo#34', 'https://github.com/example/repo/pull/56'])
        self.assertEqual(before, {name: (self.repo / '.git' / name).read_bytes()
                                  for name in before})

    def test_all_refs_include_rename_and_delete_paths(self):
        original = self.repo / 'old.txt'
        original.write_text('unchanged content\n', encoding='utf-8')
        first = self.commit()
        self.git(self.repo, 'checkout', '--quiet', '-b', 'topic')
        original.rename(self.repo / 'new.txt')
        renamed = self.commit(subject='Rename', time='2026-01-01T10:02:00+00:00')
        self.git(self.repo, 'checkout', '--quiet', 'main')
        original.unlink()
        deleted = self.commit(subject='Delete', time='2026-01-01T10:03:00+00:00')
        records = {row['sha']: row for row in extract_git.extract_repositories([self.repo])}
        self.assertEqual(set(records), {first, renamed, deleted})
        self.assertEqual(records[renamed]['files_changed'], ['new.txt', 'old.txt'])
        self.assertEqual(records[deleted]['files_changed'], ['old.txt'])

    def test_empty_repository_has_no_commits(self):
        self.assertEqual(extract_git.extract_repositories([self.repo]), [])
        with redirect_stdout(io.StringIO()) as output:
            result = extract_git.main([str(self.repo), '--output-dir', str(self.root / 'out')])
        self.assertEqual(result, 0)
        self.assertIn('Total commits: 0', output.getvalue())
        self.assertFalse((self.root / 'out').exists())

    def test_links_require_both_closed_window_and_matching_path(self):
        sha = self.commit()
        records = extract_git.extract_repositories([self.repo])
        sessions = [
            self.session(),
            self.session('prefix', cwd=str(self.repo) + '-other'),
            self.session('relative', cwd='src'),
            self.session('home', cwd='~/project'),
            self.session('missing-path', cwd=None),
            self.session('wrong-root', git_repo_root=str(self.root / 'different')),
            self.session('early', ended_at='2026-01-01T10:00:30Z'),
            self.session('late', timestamp='2026-01-01T10:01:01Z'),
            self.session('reversed', ended_at='2026-01-01T09:59:59Z'),
            self.session('bad-start', timestamp='invalid'),
            self.session('bad-end', ended_at='invalid'),
            self.session('unbounded', messages=[]),
        ]
        links = extract_git.link_sessions(records, sessions)
        self.assertEqual([row['session_id'] for row in links], ['session-1'])
        self.assertEqual(links[0]['commit_sha'], sha)
        self.assertEqual(links[0]['path_field'], 'cwd')
        self.assertEqual(links[0]['session_start'], '2026-01-01T10:00:00Z')
        self.assertEqual(links[0]['session_end'], '2026-01-01T10:02:00Z')

    def test_profiles_branches_and_source_files_disambiguate_duplicate_session_ids(self):
        self.commit()
        commits = extract_git.extract_repositories([self.repo])
        seconds = datetime(2026, 1, 1, 10, tzinfo=timezone.utc).timestamp()
        first = self.session('shared', source='hermes', cwd=None, timestamp=None,
                             source_file=str(self.root / 'state.db'), metadata={
                                 'profile': 'default', 'git_repo_root': str(self.repo),
                                 'started_at': seconds, 'ended_at': (seconds + 120) * 1000,
                             })
        second = {**first, 'metadata': {**first['metadata'], 'profile': 'work'}}
        branch = {**first, 'branch_id': 'leaf'}
        other_file = {**first, 'source_file': str(self.root / 'other.db')}
        later_snapshot = {**first, 'metadata': {**first['metadata'], 'ended_at': seconds + 180}}
        links = extract_git.link_sessions(commits, [first, second, first, later_snapshot, branch, other_file])
        self.assertEqual(len(links), 4)
        self.assertEqual({row['profile'] for row in links}, {'default', 'work'})
        self.assertEqual({row['branch_id'] for row in links}, {None, 'leaf'})
        self.assertTrue(all(row['path_field'] == 'git_repo_root' for row in links))

    def test_epoch_offsets_and_message_bounds_match_inclusively(self):
        self.commit()
        commits = extract_git.extract_repositories([self.repo])
        at_commit = self.session('point', timestamp='2026-01-01T05:01:00-05:00',
                                 ended_at='2026-01-01T10:01:00Z')
        pi = self.session('pi', source='pi', cwd=None, project_path=str(self.repo),
                          timestamp=None, created_at='2026-01-01T10:00:00Z')
        observed = self.session('observed', timestamp=None, messages=[
            {'timestamp': '2026-01-01T10:02:00Z'}, {'timestamp': '2026-01-01T10:00:00Z'},
        ])
        links = extract_git.link_sessions(commits, [at_commit, pi, observed])
        self.assertEqual({row['session_id'] for row in links}, {'point', 'pi', 'observed'})
        self.assertEqual(extract_git.parse_timestamp(0), 0)
        self.assertEqual(extract_git.parse_timestamp('0'), 0)
        for bad in (True, '2026-01-01T10:01:00', 'invalid', float('nan'), float('inf'), 10**100):
            with self.subTest(timestamp=bad):
                self.assertIsNone(extract_git.parse_timestamp(bad))

    def test_nested_repositories_do_not_match_the_outer_checkout(self):
        self.commit()
        nested = self.init_repo(self.repo / 'vendor' / 'nested')
        self.commit(nested)
        session = self.session(cwd=str(nested / 'src'))
        outer = extract_git.extract_repositories([self.repo])
        self.assertEqual(extract_git.link_sessions(outer, [session]), [])
        both = extract_git.extract_repositories([self.repo, nested])
        links = extract_git.link_sessions(both, [session])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['git_repo_root'], str(nested))

    def test_worktrees_keep_path_identity_even_with_shared_commits(self):
        self.commit()
        worktree = self.root / 'worktree'
        self.git(self.repo, 'worktree', 'add', '--quiet', '-b', 'work', str(worktree))
        commits = extract_git.extract_repositories([self.repo, worktree])
        self.assertEqual(len(commits), 2)
        links = extract_git.link_sessions(commits, [self.session(cwd=str(worktree))])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['git_repo_root'], str(worktree))

    def test_nested_bare_repository_is_not_attributed_to_outer_checkout(self):
        self.commit()
        bare = self.repo / 'cache.git'
        bare.mkdir()
        self.git(bare, 'init', '--quiet', '--bare', '--initial-branch=main',
                 f'--template={self.template}')
        self.assertEqual(extract_git.repository_root(bare), bare)
        commits = extract_git.extract_repositories([self.repo])
        self.assertEqual(extract_git.link_sessions(commits, [self.session(cwd=str(bare))]), [])

    def test_jsonl_inputs_skip_side_tables_and_report_invalid_json(self):
        sha = self.commit()
        folder = self.root / 'exports'
        folder.mkdir()
        source = folder / 'sessions.jsonl'
        source.write_text(json.dumps(self.session()) + '\n[]\n' +
                          json.dumps({'sha': sha}) + '\n', encoding='utf-8')
        self.assertEqual(extract_git.load_sessions([folder, source]), [self.session()])
        source.write_text('{}\ninvalid\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            extract_git.load_sessions([source])

    def test_cli_writes_separate_jsonl_covered_by_privacy_filter(self):
        self.commit()
        source = self.root / 'sessions.jsonl'
        source.write_text(json.dumps(self.session()) + '\n', encoding='utf-8')
        output = self.root / 'git-output'
        with redirect_stdout(io.StringIO()) as stdout:
            status = extract_git.main([str(self.repo), '--sessions', str(source),
                                       '--output-dir', str(output)])
        self.assertEqual(status, 0)
        self.assertIn('Total commits: 1', stdout.getvalue())
        self.assertIn('Session links: 1', stdout.getvalue())
        files = sorted(output.glob('*.jsonl'))
        self.assertEqual(len(files), 2)
        self.assertEqual(len(discover_inputs([output])), 2)
        commit_file = next(output.glob('git_commits_*.jsonl'))
        self.assertEqual(json.loads(commit_file.read_text())['author'], 'Fixture')
        filtered = self.root / 'filtered.jsonl'
        stats = filter_jsonl(commit_file, filtered, lambda value: value.replace('Fixture', '<AUTHOR>'))
        self.assertEqual(stats.records, 1)
        self.assertEqual(json.loads(filtered.read_text())['author'], '<AUTHOR>')
        with redirect_stdout(io.StringIO()):
            status = extract_git.main([str(self.repo), '--sessions', str(source),
                                       '--output-dir', str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(len(list(output.glob('*.jsonl'))), 4)

    def test_symlink_parent_paths_cannot_create_false_repository_links(self):
        self.commit()
        outside = self.root / 'outside'
        outside.mkdir()
        (self.repo / 'link').symlink_to(outside, target_is_directory=True)
        commits = extract_git.extract_repositories([self.repo])
        sessions = [self.session('symlink', cwd=str(self.repo / 'link' / 'src')),
                    self.session('parent', cwd=str(self.repo / 'link' / '..' / 'src'))]
        self.assertEqual(extract_git.link_sessions(commits, sessions), [])

    def test_non_utf8_filenames_use_filterable_escaped_text(self):
        filename = os.fsdecode(b'nonutf8-\xff.txt')
        (self.repo / filename).write_bytes(b'fixture\n')
        self.commit()
        output = self.root / 'byte-path-output'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(extract_git.main([str(self.repo), '--output-dir', str(output)]), 0)
        source = next(output.glob('git_commits_*.jsonl'))
        record = json.loads(source.read_text(encoding='utf-8'))
        self.assertEqual(record['files_changed'], ['nonutf8-\\xff.txt'])
        destination = self.root / 'filtered-byte-path.jsonl'
        self.assertEqual(filter_jsonl(source, destination, lambda value: value).records, 1)
        self.assertEqual(json.loads(destination.read_text(encoding='utf-8')), record)

    def test_missing_session_input_fails_without_writing_output(self):
        self.commit()
        output = self.root / 'out'
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            status = extract_git.main([str(self.repo), '--sessions', str(self.root / 'missing'),
                                       '--output-dir', str(output)])
        self.assertEqual(status, 1)
        self.assertIn('Session input does not exist', errors.getvalue())
        self.assertFalse(output.exists())

    def test_no_repository_arguments_do_not_discover_home(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(extract_git.main([]), 0)
        self.assertIn('Git extraction skipped', output.getvalue())

    def test_ambient_git_directory_cannot_redirect_requested_repository(self):
        expected = self.commit(subject='Requested repository')
        other = self.init_repo(self.root / 'other-repo')
        self.commit(other, subject='Unrelated repository')
        with patch.dict(os.environ, {'GIT_DIR': str(other / '.git')}):
            records = extract_git.extract_repositories([self.repo])
        self.assertEqual([row['sha'] for row in records], [expected])


    def ask_session(self, content, session_id='ask', **extra):
        fields = {
            'cwd': None,
            'messages': [{'role': 'user', 'content': content,
                          'timestamp': '2026-01-01T10:02:00Z'}],
        }
        fields.update(extra)
        return self.session(session_id, **fields)

    def pull_request(self, sha, **extra):
        return {
            'url': 'https://github.com/example/repo/pull/410', 'number': 410,
            'title': 'Fix extraction evidence', 'state': 'OPEN',
            'headRefName': 'feat/issue-410', 'baseRefName': 'main',
            'headRefOid': sha, 'mergeCommit': None, **extra,
        }

    def test_local_branch_metadata_and_qualified_issue_ask_without_path(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'git@github.com:example/repo.git')
        sha = self.commit(subject='Fix #410')
        self.git(self.repo, 'branch', 'feat/issue-410')
        records = extract_git.extract_repositories([self.repo])
        record = records[0]
        self.assertEqual(record['git_repository'], 'example/repo')
        self.assertEqual(record['pull_requests'], [])
        branch = next(row for row in record['branches'] if row['name'] == 'feat/issue-410')
        self.assertEqual(branch, {
            'name': 'feat/issue-410', 'ref': 'refs/heads/feat/issue-410',
            'tip_sha': sha, 'evidence_kind': 'current_ref_reachability',
        })
        ask = self.ask_session('Fix example/repo#410.', source='hermes',
                               source_file='/exports/work/state.db',
                               branch_id='conversation-leaf',
                               metadata={'profile': 'work'})
        links = extract_git.link_sessions(records, [ask, ask])
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertEqual(link['commit_sha'], sha)
        self.assertEqual(link['match_basis'], 'repository_issue_reference_and_time_window')
        self.assertEqual(link['evidence'], [
            {'kind': 'commit_issue_reference', 'reference': 'example/repo#410'}])
        self.assertIsNone(link['path_field'])
        self.assertIsNone(link['matched_path'])
        self.assertTrue(link['candidate'])
        self.assertEqual((link['source'], link['profile'], link['branch_id'], link['source_file']),
                         ('hermes', 'work', 'conversation-leaf', '/exports/work/state.db'))

    def test_cross_repository_tracker_references_keep_exact_namespaces(self):
        self.git(self.repo, 'remote', 'add', 'origin',
                 'https://github.com/example/code.git')
        short_sha = self.commit(subject='Fix example/tracker#410')
        url_sha = self.commit(
            subject='Fix https://github.com/example/tracker/issues/410')
        bare_sha = self.commit(subject='Fix #410')
        self.commit(subject='Refs https://github.com/../tracker/issues/410 and '
                            'https://github.com/example/../issues/410')
        sessions = [
            self.ask_session('Fix example/tracker#410.', 'tracker-short'),
            self.ask_session('Fix https://github.com/example/tracker/issues/410',
                             'tracker-url'),
            self.ask_session('Fix example/tracker#410.', 'declared-code',
                             metadata={'git_repository': 'example/code'}),
            self.ask_session('Fix example/code#410.', 'origin-qualified'),
            self.ask_session('Fix #410.', 'origin-bare',
                             metadata={'git_repository': 'example/code'}),
            self.ask_session('Fix #410.', 'unqualified'),
            self.ask_session('Fix other/tracker#410.', 'wrong-tracker'),
            self.ask_session('Fix example/tracker#411.', 'wrong-number'),
            self.ask_session('Fix https://github.com/../tracker/issues/410',
                             'invalid-owner'),
            self.ask_session('Fix https://github.com/example/../issues/410',
                             'invalid-repository'),
            self.ask_session('Fix example/tracker#410.', 'outside-window',
                             ended_at='2026-01-01T10:00:30Z'),
            self.ask_session('Fix example/tracker#410.', 'foreign-declaration',
                             metadata={'git_repository': 'example/tracker'}),
        ]
        for session in sessions:
            session['source'] = 'hermes'
        exported = self.root / 'tracker-sessions.jsonl'
        exported.write_text(
            ''.join(json.dumps(row) + '\n' for row in sessions), encoding='utf-8')
        output = self.root / 'tracker-git-output'
        with redirect_stdout(io.StringIO()):
            status = extract_git.main([
                str(self.repo), '--sessions', str(exported), '--output-dir', str(output),
            ])
        self.assertEqual(status, 0)
        links = [json.loads(line) for line in
                 next(output.glob('git_session_links_*.jsonl')).read_text(
                     encoding='utf-8').splitlines()]
        expected = {
            (session_id, sha)
            for session_id in ('tracker-short', 'tracker-url', 'declared-code')
            for sha in (short_sha, url_sha)
        } | {('origin-qualified', bare_sha), ('origin-bare', bare_sha)}
        self.assertEqual(len(links), len(expected))
        self.assertEqual({(row['session_id'], row['commit_sha']) for row in links},
                         expected)
        for link in links:
            self.assertEqual(link['git_repository'], 'example/code')
            self.assertEqual(link['git_repo_root'], str(self.repo))
            self.assertEqual(link['source'], 'hermes')
            self.assertIsNone(link['matched_path'])
            self.assertEqual(link['match_basis'],
                             'repository_issue_reference_and_time_window')
            reference = 'example/code#410' if link['commit_sha'] == bare_sha else (
                'example/tracker#410')
            self.assertEqual(link['evidence'], [
                {'kind': 'commit_issue_reference', 'reference': reference}])

    def test_generated_summary_markers_cannot_supply_ask_evidence(self):
        self.git(self.repo, 'remote', 'add', 'origin',
                 'https://github.com/example/repo.git')
        sha = self.commit(subject='Fix #410')
        snapshot = self.root / 'summary-prs.json'
        snapshot.write_text(json.dumps([self.pull_request(
            sha, number=411, url='https://github.com/example/repo/pull/411')]),
            encoding='utf-8')
        commits = extract_git.extract_repositories(
            [self.repo], extract_git.load_pull_requests([snapshot]))
        cases = [
            ('Fix example/repo#410.', 'repository_issue_reference_and_time_window'),
            ('Review https://github.com/example/repo/pull/411',
             'repository_pull_request_reference_and_time_window'),
            ('Use example/repo@main.', 'repository_branch_reference_and_time_window'),
        ]
        markers = [
            {'synthetic': True},
            {'compaction_summary': True},
            {'event': 'compaction'},
            {'event': 'branch_summary'},
        ]
        for content, basis in cases:
            ask = self.ask_session(content)
            ask['messages'][0].update({
                'synthetic': False, 'compaction_summary': False, 'event': 'message',
            })
            with self.subTest(content=content, control='genuine-user'):
                expected = extract_git.link_sessions(commits, [ask])
                self.assertEqual(len(expected), 1)
                self.assertEqual(expected[0]['match_basis'], basis)
                self.assertEqual(expected[0]['commit_sha'], sha)
            for marker in markers:
                with self.subTest(content=content, marker=marker):
                    generated = {
                        **ask,
                        'messages': [{**ask['messages'][0], **marker}],
                    }
                    self.assertEqual(
                        extract_git.link_sessions(commits, [generated]), [])
                    mixed = {
                        **generated,
                        'messages': generated['messages'] + ask['messages'],
                    }
                    self.assertEqual(
                        extract_git.link_sessions(commits, [mixed]), expected)

    def test_pi_omp_generated_summaries_do_not_link_through_git_cli(self):
        self.git(self.repo, 'remote', 'add', 'origin',
                 'https://github.com/example/repo.git')
        sha = self.commit(subject='Implement extraction')
        conversations, expected_ids = [], set()
        for source in ('pi', 'omp'):
            for event in ('compaction', 'branch_summary'):
                for has_ask in (False, True):
                    session_id = f'{source}-{event}-{has_ask}'
                    with self.subTest(source=source, event=event, has_ask=has_ask):
                        rows = [
                            {'type': 'session', 'id': session_id,
                             'timestamp': '2026-01-01T10:00:00Z'},
                            {'type': event, 'id': 'summary', 'parentId': None,
                             'timestamp': '2026-01-01T10:00:30Z',
                             'summary': 'Use example/repo@main.'},
                        ]
                        if has_ask:
                            rows.append({
                                'type': 'message', 'id': 'ask', 'parentId': 'summary',
                                'timestamp': '2026-01-01T10:00:45Z',
                                'message': {'role': 'user',
                                            'content': 'Use example/repo@main.'},
                            })
                            expected_ids.add(session_id)
                        rows.append({
                            'type': 'message', 'id': 'answer',
                            'parentId': 'ask' if has_ask else 'summary',
                            'timestamp': '2026-01-01T10:02:00Z',
                            'message': {'role': 'assistant', 'content': 'Continuing.'},
                        })
                        fixture = self.root / f'{session_id}.jsonl'
                        fixture.write_text(
                            ''.join(json.dumps(row) + '\n' for row in rows),
                            encoding='utf-8')
                        conversation = extract_pi.extract_session(fixture, source)
                        self.assertIsNotNone(conversation)
                        self.assertIsNone(conversation['project_path'])
                        summary = conversation['messages'][0]
                        self.assertEqual(summary['content'], 'Use example/repo@main.')
                        self.assertEqual(summary['role'], 'user')
                        self.assertEqual(summary['event'], event)
                        self.assertTrue(summary['synthetic'])
                        self.assertEqual(
                            summary['compaction_summary'], event == 'compaction')
                        conversations.append(conversation)

        sessions = self.root / 'summary-conversations.jsonl'
        sessions.write_text(
            ''.join(json.dumps(row) + '\n' for row in conversations),
            encoding='utf-8')
        output = self.root / 'summary-git-output'
        with redirect_stdout(io.StringIO()):
            status = extract_git.main([
                str(self.repo), '--sessions', str(sessions), '--output-dir', str(output),
            ])
        self.assertEqual(status, 0)
        links = [json.loads(line) for line in
                 next(output.glob('git_session_links_*.jsonl')).read_text(
                     encoding='utf-8').splitlines()]
        self.assertEqual(len(links), len(expected_ids))
        self.assertEqual({row['session_id'] for row in links}, expected_ids)
        self.assertEqual({row['source'] for row in links}, {'pi', 'omp'})
        self.assertTrue(all(row['commit_sha'] == sha for row in links))
        self.assertTrue(all(
            row['match_basis'] == 'repository_branch_reference_and_time_window'
            for row in links))

    def test_branch_asks_use_exact_qualified_current_refs_and_closed_windows(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        self.commit(subject='Implement extraction')
        self.git(self.repo, 'branch', 'feat/issue-410')
        records = extract_git.extract_repositories([self.repo])
        sessions = [
            self.ask_session('Work on example/repo@feat/issue-410.', 'qualified'),
            self.ask_session('Use https://github.com/example/repo/tree/feat/issue-410', 'url'),
            self.ask_session('Use `example/repo@feat/issue-410`.', 'inline-code'),
            self.ask_session('Use "https://github.com/example/repo/tree/feat/issue-410".',
                             'quoted-url'),
            self.ask_session('Continue.', 'metadata',
                             metadata={'git_repository': 'example/repo',
                                       'git_branch': 'feat/issue-410'}),
            self.ask_session('Work on feat/issue-410.', 'unqualified'),
            self.ask_session('Use example/repo@feat/issue-41.', 'prefix'),
            self.ask_session('Use other/repo@feat/issue-410.', 'foreign'),
            self.ask_session('Use example/repo@feat/issue-410.', 'late',
                             timestamp='2026-01-01T10:03:00Z'),
            self.ask_session('Use example/repo@feat/issue-410.', 'unbounded',
                             timestamp=None, messages=[]),
            self.ask_session('', 'assistant', messages=[
                {'role': 'assistant', 'content': 'Use example/repo@feat/issue-410.',
                 'timestamp': '2026-01-01T10:02:00Z'}]),
            self.ask_session('', 'conversation-branch', branch_id='feat/issue-410'),
            self.ask_session('Use example/repo@feat/issue-410.', 'conflicting-branches',
                             git_repository='example/repo', git_branch='feat/issue-410',
                             metadata={'git_branch': 'different'}),
        ]
        links = extract_git.link_sessions(records, sessions)
        self.assertEqual({row['session_id'] for row in links},
                         {'qualified', 'url', 'metadata', 'inline-code', 'quoted-url'})
        for link in links:
            self.assertEqual(link['match_basis'], 'repository_branch_reference_and_time_window')
            self.assertEqual(link['evidence'][0]['kind'], 'current_ref_reachability')
            self.assertEqual(link['evidence'][0]['branch'], 'feat/issue-410')

    def test_issue_references_do_not_cross_repositories_with_shared_shas(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        self.git(self.repo, 'remote', 'add', 'upstream', 'https://github.com/other/repo.git')
        sha = self.commit(subject='Fix #410')
        other = self.init_repo(self.root / 'other')
        self.git(other, 'remote', 'add', 'origin', 'https://github.com/other/repo.git')
        self.assertEqual(self.commit(other, subject='Fix #410'), sha)
        records = extract_git.extract_repositories([self.repo, other])
        sessions = [
            self.ask_session('Fix example/repo#410.', 'qualified'),
            self.ask_session('Fix https://github.com/example/repo/issues/410', 'url'),
            self.ask_session('Fix #410.', 'bare'),
            self.ask_session('Fix #410.', 'metadata',
                             metadata={'git_repository': 'example/repo'}),
        ]
        links = extract_git.link_sessions(records, sessions)
        self.assertEqual({row['session_id'] for row in links}, {'qualified', 'url', 'metadata'})
        self.assertTrue(all(row['git_repo_root'] == str(self.repo) for row in links))
        foreign = self.commit(subject='Refs other/repo#411')
        records = extract_git.extract_repositories([self.repo])
        links = extract_git.link_sessions(records, [self.ask_session('Fix example/repo#411.')])
        self.assertEqual(links, [])
        self.assertIn(foreign, {row['sha'] for row in records})

    def test_pathless_references_reject_ambiguous_worktrees(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        self.commit(subject='Fix #410')
        worktree = self.root / 'worktree'
        self.git(self.repo, 'worktree', 'add', '--quiet', '-b', 'work', str(worktree))
        records = extract_git.extract_repositories([self.repo, worktree])
        ask = self.ask_session('Fix example/repo#410.')
        self.assertEqual(extract_git.link_sessions(records, [ask]), [])
        located = self.ask_session('Fix example/repo#410.', cwd=str(worktree))
        links = extract_git.link_sessions(records, [located])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['git_repo_root'], str(worktree))

    def test_invalid_or_conflicting_paths_cannot_fall_back_to_text_evidence(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        self.commit(subject='Fix #410')
        records = extract_git.extract_repositories([self.repo])
        outside = self.root / 'outside'
        outside.mkdir()
        (self.repo / 'link').symlink_to(outside, target_is_directory=True)
        sessions = [
            self.ask_session('Fix example/repo#410.', 'relative', cwd='src'),
            self.ask_session('Fix example/repo#410.', 'wrong', cwd=str(outside)),
            self.ask_session('Fix example/repo#410.', 'symlink',
                             cwd=str(self.repo / 'link' / 'src')),
            self.ask_session('Fix example/repo#410.', 'conflicting',
                             git_repo_root=str(self.repo), cwd=str(outside)),
            self.ask_session('Fix example/repo#410.', 'metadata-path',
                             cwd=str(self.repo), metadata={'cwd': str(outside)}),
            self.ask_session('Fix example/repo#410.', 'invalid-repo',
                             metadata={'git_repository': 'not-qualified'}),
            self.ask_session('Fix example/repo#410.', 'foreign-metadata',
                             metadata={'git_repository': 'other/repo'}),
            self.ask_session('Fix #410.', 'conflicting-repositories',
                             git_repository='example/repo',
                             metadata={'git_repository': 'other/repo'}),
        ]
        self.assertEqual(extract_git.link_sessions(records, sessions), [])

    def test_valid_path_evidence_survives_unrelated_repository_references(self):
        self.commit(subject='Fix #410')
        ask = self.ask_session('Compare other/repo#410 and example/repo@topic.',
                               cwd=str(self.repo))
        for with_origin in (False, True):
            with self.subTest(with_origin=with_origin):
                if with_origin:
                    self.git(self.repo, 'remote', 'add', 'origin',
                             'https://github.com/example/repo.git')
                records = extract_git.extract_repositories([self.repo])
                links = extract_git.link_sessions(records, [ask])
                self.assertEqual(len(links), 1)
                self.assertEqual(links[0]['match_basis'], 'repository_path_and_time_window')
                self.assertEqual(links[0]['matched_path'], str(self.repo))

    def test_pull_request_snapshot_exports_only_exact_head_and_merge_evidence(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        earlier = self.commit(subject='Earlier unrelated change')
        head = self.commit(subject='Implement extraction')
        merged = self.commit(subject='Merge extraction')
        snapshot = self.root / 'pull-requests.json'
        snapshot.write_text(json.dumps([self.pull_request(
            head, state='MERGED', mergeCommit={'oid': merged})]), encoding='utf-8')
        pull_requests = extract_git.load_pull_requests([snapshot])
        records = {row['sha']: row for row in extract_git.extract_repositories(
            [self.repo], pull_requests)}
        self.assertEqual(records[earlier]['pull_requests'], [])
        self.assertEqual(records[head]['pull_requests'][0]['commit_relationship'], 'head_sha')
        self.assertEqual(records[merged]['pull_requests'][0]['commit_relationship'], 'merge_sha')
        links = extract_git.link_sessions(list(records.values()), [
            self.ask_session('Review https://github.com/example/repo/pull/410'),
            self.ask_session('Review example/repo#410', 'outside',
                             ended_at='2026-01-01T10:00:30Z'),
        ])
        self.assertEqual({row['commit_sha'] for row in links}, {head, merged})
        self.assertTrue(all(row['session_id'] == 'ask' for row in links))
        self.assertTrue(all(row['match_basis'] ==
                            'repository_pull_request_reference_and_time_window' for row in links))
        self.assertTrue(all(row['evidence'][0]['evidence_source'] == str(snapshot) for row in links))

    def test_pull_request_snapshot_cli_emits_filterable_separate_table(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        sha = self.commit(subject='Implement extraction')
        snapshot = self.root / 'pull-requests.json'
        snapshot.write_text(json.dumps([self.pull_request(sha)]), encoding='utf-8')
        sessions = self.root / 'sessions.jsonl'
        sessions.write_text(json.dumps(self.ask_session('Fix example/repo#410.')) + '\n',
                            encoding='utf-8')
        output = self.root / 'pr-output'
        with redirect_stdout(io.StringIO()):
            status = extract_git.main([str(self.repo), '--sessions', str(sessions),
                                       '--pull-requests', str(snapshot),
                                       '--output-dir', str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(len(list(output.glob('*.jsonl'))), 3)
        self.assertEqual(len(discover_inputs([output])), 3)
        pr_file = next(output.glob('git_pull_requests_*.jsonl'))
        record = json.loads(pr_file.read_text(encoding='utf-8'))
        self.assertEqual(record['git_repository'], 'example/repo')
        self.assertEqual(record['head_sha'], sha)
        self.assertEqual(record['head_branch'], 'feat/issue-410')
        self.assertEqual(record['base_branch'], 'main')
        self.assertEqual(record['evidence_kind'], 'exported_pull_request_snapshot')
        filtered = self.root / 'filtered-pr.jsonl'
        self.assertEqual(filter_jsonl(pr_file, filtered, lambda value: value).records, 1)
        self.assertEqual(json.loads(filtered.read_text(encoding='utf-8')), record)

    def test_invalid_or_foreign_pull_request_snapshots_fail_without_output(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        sha = self.commit()
        snapshot = self.root / 'invalid-prs.json'
        invalid = [
            {'not': 'an array'},
            [self.pull_request(sha, number=True)],
            [self.pull_request(sha, number=411)],
            [self.pull_request('not-a-sha')],
            [self.pull_request(sha, headRefName='../branch')],
            [self.pull_request(sha, state='MERGED', mergeCommit={'oid': 'invalid'})],
            [self.pull_request(sha, mergeCommit={'oid': sha})],
            [self.pull_request(sha, state='CLOSED', mergeCommit={'oid': sha})],
            [self.pull_request(sha, url='https://github.com/other/repo/pull/410')],
            [self.pull_request(sha), self.pull_request(sha, state='CLOSED')],
        ]
        for index, records in enumerate(invalid):
            with self.subTest(index=index):
                snapshot.write_text(json.dumps(records), encoding='utf-8')
                output = self.root / f'invalid-output-{index}'
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    status = extract_git.main([str(self.repo), '--pull-requests', str(snapshot),
                                               '--output-dir', str(output)])
                self.assertEqual(status, 1)
                self.assertFalse(output.exists())

    def test_empty_repository_can_export_pr_snapshot_without_inventing_links(self):
        self.git(self.repo, 'remote', 'add', 'origin', 'https://github.com/example/repo.git')
        other = self.init_repo(self.root / 'snapshot-source')
        sha = self.commit(other, subject='Snapshot commit outside the empty checkout')
        snapshot = self.root / 'pull-requests.json'
        snapshot.write_text(json.dumps([self.pull_request(sha)]), encoding='utf-8')
        sessions = self.root / 'sessions.jsonl'
        sessions.write_text(json.dumps(self.ask_session('Fix example/repo#410.')) + '\n',
                            encoding='utf-8')
        output = self.root / 'empty-pr-output'
        with redirect_stdout(io.StringIO()):
            status = extract_git.main([str(self.repo), '--sessions', str(sessions),
                                       '--pull-requests', str(snapshot),
                                       '--output-dir', str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(len(list(output.glob('*.jsonl'))), 3)
        pr_file = next(output.glob('git_pull_requests_*.jsonl'))
        self.assertEqual(json.loads(pr_file.read_text(encoding='utf-8'))['head_sha'], sha)
        self.assertEqual(next(output.glob('git_commits_*.jsonl')).read_text(), '')
        self.assertEqual(next(output.glob('git_session_links_*.jsonl')).read_text(), '')

    def test_utc_timestamp_spellings_compare_as_instants(self):
        expected = datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc).timestamp()
        for spelling in ('2026-01-01T10:01:00Z', '2026-01-01T10:01:00+00:00',
                         '2026-01-01T05:01:00-05:00'):
            with self.subTest(spelling=spelling):
                self.assertEqual(extract_git.parse_timestamp(spelling), expected)


if __name__ == '__main__':
    unittest.main()

