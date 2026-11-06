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
        self.assertEqual(record['time'], '2026-01-01T10:01:00+00:00')
        self.assertEqual(record['author_time'], '2026-01-01T09:00:00+00:00')
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


if __name__ == '__main__':
    unittest.main()
