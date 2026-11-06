#!/usr/bin/env python3
"""Extract local Git commits and conservative links to exported conversations.

Only explicitly supplied repositories are read. Commit ``time`` is the committer
time; ``author_time`` is retained separately. Links are candidates, not proof of
authorship. A closed session window and either a matching repository path or
repository-qualified issue/branch evidence are required. Branch refs describe
current reachability, not historical branch membership. Optional pull-request
snapshots are supplied explicitly; the extractor never contacts GitHub.
If a session has no ended_at, its last timestamped message bounds the window.
Git records are written separately from conversation JSONL, under extracted_data/git.
Non-UTF-8 filename bytes are represented as literal \\xNN escapes, so the JSONL
also remains readable by the privacy filter's UTF-8 writer.
"""

import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote


ISSUE_REF = re.compile(
    r'https?://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/[1-9]\d*\b'
    r'|(?<![\w/.-])(?:[\w.-]+/[\w.-]+)?#[1-9]\d*\b'
)



def _repository_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', value):
        return None
    if any(part in ('.', '..') for part in value.split('/')):
        return None
    return value.lower()


def _origin_repository(root):
    """Use only the explicit checkout's origin; never infer from an upstream."""
    config = _git(root, 'config', '--local', '--null', '--list')
    urls = [entry.partition(b'\n')[2].decode('utf-8', errors='replace')
            for entry in config.split(b'\0')
            if entry.partition(b'\n')[0].lower() == b'remote.origin.url']
    if len(urls) != 1:
        return None
    match = re.fullmatch(
        r'(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)'
        r'([\w.-]+/[\w.-]+?)(?:\.git)?/?', urls[0], re.IGNORECASE)
    return _repository_name(match[1]) if match else None


def _valid_branch(value):
    if not isinstance(value, str) or not value or value == '@':
        return False
    if (re.search(r'[\x00-\x20\x7f~^:?*\[\\]', value)
            or value.startswith(('/', '-')) or value.endswith(('/', '.'))
            or '..' in value or '//' in value or '@{' in value):
        return False
    return all(not part.startswith('.') and not part.endswith('.lock')
               for part in value.split('/'))


def _branches_containing(root, sha):
    raw = _git(root, 'for-each-ref', f'--contains={sha}',
               '--format=%(refname)%00%(objectname)', 'refs/heads')
    branches = []
    for line in raw.splitlines():
        ref, separator, tip = line.partition(b'\0')
        if not separator:
            raise ValueError(f'Unexpected Git branch format in {root}')
        ref = ref.decode('utf-8', errors='backslashreplace')
        branches.append({
            'name': ref.removeprefix('refs/heads/'), 'ref': ref,
            'tip_sha': tip.decode('ascii'),
            'evidence_kind': 'current_ref_reachability',
        })
    return branches


def _issue_key(reference, default_repository=None):
    match = re.fullmatch(
        r'https?://github\.com/([\w.-]+/[\w.-]+)/(?:issues|pull)/([1-9]\d*)',
        reference)
    if match:
        return _repository_name(match[1]), int(match[2])
    match = re.fullmatch(r'([\w.-]+/[\w.-]+)?#([1-9]\d*)', reference)
    if match:
        repository = _repository_name(match[1]) if match[1] else default_repository
        return (repository, int(match[2])) if repository else None
    return None


def load_pull_requests(paths):
    """Read explicit JSON arrays from a GitHub PR export, without network access."""
    records = {}
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        with path.open('r', encoding='utf-8') as source:
            data = json.load(source)
        if not isinstance(data, list):
            raise ValueError(f'Pull-request input must be a JSON array: {path}')
        for position, item in enumerate(data, 1):
            error = f'Invalid pull-request record {position} in {path}'
            if not isinstance(item, dict):
                raise ValueError(error)
            url = item.get('url')
            match = re.fullmatch(
                r'https://github\.com/([\w.-]+/[\w.-]+)/pull/([1-9]\d*)',
                url) if isinstance(url, str) else None
            number = item.get('number')
            repository = _repository_name(match[1]) if match else None
            state = item.get('state')
            head_sha = item.get('headRefOid')
            merge = item.get('mergeCommit')
            merge_sha = merge.get('oid') if isinstance(merge, dict) else None
            if (repository is None or type(number) is not int
                    or number != int(match[2])
                    or not isinstance(item.get('title'), str)
                    or state not in ('OPEN', 'CLOSED', 'MERGED')
                    or not _valid_branch(item.get('headRefName'))
                    or not _valid_branch(item.get('baseRefName'))
                    or not isinstance(head_sha, str)
                    or not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', head_sha)
                    or (merge is not None and state != 'MERGED')
                    or (merge is not None and (
                        not isinstance(merge, dict) or not isinstance(merge_sha, str)
                        or not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', merge_sha)))):
                raise ValueError(error)
            record = {
                'source': 'git', 'git_repository': repository, 'number': number,
                'url': f'https://github.com/{repository}/pull/{number}',
                'title': item['title'], 'state': state,
                'head_branch': item['headRefName'], 'base_branch': item['baseRefName'],
                'head_sha': head_sha.lower(),
                'merge_sha': merge_sha.lower() if merge_sha else None,
                'evidence_kind': 'exported_pull_request_snapshot',
                'evidence_source': str(path),
            }
            key = repository, number
            previous = records.get(key)
            if previous and any(previous[field] != record[field]
                                for field in record if field != 'evidence_source'):
                raise ValueError(f'Conflicting pull-request snapshots for {repository}#{number}')
            records.setdefault(key, record)
    return [records[key] for key in sorted(records)]


def _git(repository, *arguments):
    # Ambient GIT_DIR / GIT_CONFIG_* must not redirect an explicit repository.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update({
        'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0',
        'GIT_NO_LAZY_FETCH': '1', 'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_CONFIG_GLOBAL': os.devnull,
    })
    result = subprocess.run(
        ['git', '--no-pager', '-c', 'core.fsmonitor=false',
         '-c', 'core.quotePath=false', '-c', 'protocol.allow=never',
         '-c', 'i18n.logOutputEncoding=UTF-8', '-C', str(repository), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False,
    )
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'Git read failed in {repository}: {detail}')
    return result.stdout


def repository_root(path):
    """Resolve a supplied worktree/subdirectory or bare repository, without writes."""
    path = Path(path).resolve(strict=True)
    bare = _git(path, 'rev-parse', '--is-bare-repository') == b'true\n'
    option = '--absolute-git-dir' if bare else '--show-toplevel'
    raw = _git(path, 'rev-parse', option).removesuffix(b'\n')
    return Path(os.fsdecode(raw)).resolve(strict=True)


def extract_repository(root, pull_requests=()):
    """Export commits reachable from all refs and HEAD, including root/merge changes."""
    raw = _git(root, 'log', '--all', '-z', '--no-color', '--no-decorate',
               '--no-notes', '--no-show-signature',
               '--format=%H%x00%an%x00%aI%x00%cI%x00%s%x00%B')
    if not raw:
        return []
    fields = raw.split(b'\0')
    if fields[-1] == b'':
        fields.pop()  # Remove the record terminator, not an empty final field.
    if len(fields) % 6:
        raise ValueError(f'Unexpected Git log format in {root}')

    repository = _origin_repository(root)
    commits = []
    for offset in range(0, len(fields), 6):
        sha, author, author_time, time, subject, body = (
            value.decode('utf-8', errors='backslashreplace')
            for value in fields[offset:offset + 6]
        )
        # --no-renames preserves both paths of a rename; -m unions changes
        # against each merge parent. -z preserves tabs/newlines in filenames.
        changed = _git(root, 'diff-tree', '--root', '--no-commit-id',
                       '--name-only', '--no-renames', '--no-ext-diff',
                       '--no-textconv', '-r', '-m', '-z', sha, '--')
        files = sorted({name.decode('utf-8', errors='backslashreplace')
                        for name in changed.split(b'\0') if name})
        commits.append({
            'source': 'git', 'git_repo_root': str(root), 'sha': sha,
            'author': author, 'time': time, 'author_time': author_time,
            'subject': subject, 'files_changed': files,
            'issue_refs': sorted(set(ISSUE_REF.findall(body))),
            'git_repository': repository,
            'branches': _branches_containing(root, sha),
            'pull_requests': [dict(record, commit_relationship=(
                'head_sha' if sha == record['head_sha'] else 'merge_sha'))
                for record in pull_requests
                if record['git_repository'] == repository
                and sha in (record['head_sha'], record['merge_sha'])],
        })
    return sorted(commits, key=lambda row: (row['time'], row['sha']))


def extract_repositories(paths, pull_requests=()):
    roots = list(dict.fromkeys(repository_root(path) for path in paths))
    repositories = {_origin_repository(root) for root in roots}
    if any(record['git_repository'] not in repositories for record in pull_requests):
        raise ValueError('Pull-request snapshots must match a supplied repository origin')
    return [commit for root in roots
            for commit in extract_repository(root, pull_requests)]


def load_sessions(paths):
    """Read explicit JSONL inputs; directories may also contain Git side tables."""
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(candidate for candidate in sorted(path.rglob('*.jsonl'))
                         if candidate.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f'Session input does not exist: {path}')
    sessions = []
    for path in dict.fromkeys(file.resolve() for file in files):
        with path.open('r', encoding='utf-8') as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON in {path} at line {number}') from exc
                if isinstance(record, dict) and isinstance(record.get('messages'), list):
                    sessions.append(record)
    return sessions


def parse_timestamp(value):
    """Accept epoch seconds/milliseconds or ISO timestamps with an explicit zone."""
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            if re.fullmatch(r'-?\d+(?:\.\d+)?', value):
                value = float(value)
            else:
                date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return date.timestamp() if date.tzinfo is not None else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            seconds = value / 1000 if abs(value) >= 100_000_000_000 else value
            return datetime.fromtimestamp(seconds, timezone.utc).timestamp()
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _value(session, *keys):
    metadata = session.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    for key in keys:
        for record in (session, metadata):
            if record.get(key) is not None:
                return record[key]
    return None


def session_window(session):
    """Missing bounds use observed messages; invalid explicit bounds fail closed."""
    observed = [parse_timestamp(message.get('timestamp'))
                for message in session.get('messages', []) if isinstance(message, dict)]
    observed = [time for time in observed if time is not None]
    start_value = _value(session, 'started_at', 'created_at', 'timestamp')
    end_value = _value(session, 'ended_at')
    start = parse_timestamp(start_value) if start_value is not None else min(observed, default=None)
    end = parse_timestamp(end_value) if end_value is not None else max(observed, default=None)
    if start is None or end is None or end < start:
        return None
    return start, end


def _absolute_path(value):
    # Never expand '~' or interpret relative paths against the extractor's cwd.
    if not isinstance(value, str) or not value or '\0' in value:
        return None
    path = Path(value)
    # Collapsing link/.. lexically could hide a symlink to a different checkout.
    if '..' in path.parts:
        return None
    return path if path.is_absolute() else None


def _path_match(session, root, roots):
    declared_root = _value(session, 'git_repo_root')
    if declared_root is not None:
        candidate = _absolute_path(declared_root)
        return ('git_repo_root', str(candidate)) if candidate == root else None

    field = 'cwd' if _value(session, 'cwd') is not None else 'project_path'
    candidate = _absolute_path(_value(session, field))
    if candidate is None or not candidate.is_relative_to(root):
        return None
    # A nested supplied repository takes precedence over its containing checkout.
    if any(other != root and other.is_relative_to(root) and candidate.is_relative_to(other)
           for other in roots):
        return None
    # Do not attribute an unselected nested checkout/worktree to an outer repo.
    current = candidate
    while current != root:
        bare = ((current / 'HEAD').is_file() and (current / 'objects').is_dir()
                and ((current / 'refs').is_dir() or (current / 'reftable').is_dir()
                     or (current / 'packed-refs').is_file()))
        if (current / '.git').exists() or bare or current.is_symlink():
            return None
        current = current.parent
    return field, str(candidate)


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace('+00:00', 'Z')


def _json_safe(value):
    """Render undecodable path surrogates as escaped text, not invalid Unicode."""
    if isinstance(value, str):
        return value.encode('utf-8', errors='backslashreplace').decode('utf-8')
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value



def _path_values(session):
    metadata = session.get('metadata')
    records = (session, metadata) if isinstance(metadata, dict) else (session,)
    return [(field, record[field]) for record in records
            for field in ('git_repo_root', 'cwd', 'project_path')
            if record.get(field) is not None]


def _declarations_agree(session, field):
    metadata = session.get('metadata')
    if (not isinstance(metadata, dict) or session.get(field) is None
            or metadata.get(field) is None):
        return True
    left, right = session[field], metadata[field]
    if field == 'git_repository':
        left, right = _repository_name(left), _repository_name(right)
    return left == right


def _consistent_path_match(session, root, roots):
    # Every explicit path must agree. A rejected path cannot become text evidence.
    values = _path_values(session)
    if not values or any(_path_match({field: value}, root, roots) is None
                         for field, value in values):
        return None
    return _path_match(session, root, roots)


def _session_references(session):
    """Only explicit user asks and qualified metadata are evidence, never echoes."""
    repository = _repository_name(_value(session, 'git_repository'))
    issues, branches = set(), set()
    branch = _value(session, 'git_branch')
    if repository and _valid_branch(branch):
        branches.add((repository, branch))
    for message in session.get('messages', []):
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, list):
            content = '\n'.join(part['text'] for part in content
                                if isinstance(part, dict)
                                and isinstance(part.get('text'), str))
        if not isinstance(content, str):
            continue
        issues.update(key for reference in ISSUE_REF.findall(content)
                      if (key := _issue_key(reference, repository)) is not None)
        for pattern in (
                r"https://github\.com/([\w.-]+/[\w.-]+)/tree/([^\s?#<>()\[\]'\"`]+)",
                r"(?<![\w./@-])([\w.-]+/[\w.-]+)@([^\s?#<>()\[\]'\"`]+)"):
            for repo, name in re.findall(pattern, content):
                repo = _repository_name(repo)
                name = unquote(name.rstrip('.,;!'))
                if repo and _valid_branch(name):
                    branches.add((repo, name))
    return issues, branches


def _reference_match(commit, issues, branches):
    repository = commit.get('git_repository')
    if not repository:
        return None
    commit_issues = {key for reference in commit.get('issue_refs', [])
                     if (key := _issue_key(reference, repository)) is not None
                     and key[0] == repository}
    matched = sorted(commit_issues & issues)
    if matched:
        return 'repository_issue_reference_and_time_window', [
            {'kind': 'commit_issue_reference', 'reference': f'{repo}#{number}'}
            for repo, number in matched]
    pull_requests = [record for record in commit.get('pull_requests', [])
                     if (repository, record['number']) in issues]
    if pull_requests:
        return 'repository_pull_request_reference_and_time_window', [
            {'kind': 'exported_pull_request_snapshot', 'url': record['url'],
             'commit_relationship': record['commit_relationship'],
             'evidence_source': record['evidence_source']}
            for record in pull_requests]
    matched_branches = [record for record in commit.get('branches', [])
                        if (repository, record['name']) in branches]
    if matched_branches:
        return 'repository_branch_reference_and_time_window', [
            {'kind': 'current_ref_reachability', 'ref': record['ref'],
             'branch': record['name'], 'tip_sha': record['tip_sha']}
            for record in matched_branches]
    return None


def link_sessions(commits, sessions):
    """Join bounded intervals with paths or qualified evidence; keep candidates."""
    by_root = {}
    for commit in commits:
        root = _absolute_path(commit.get('git_repo_root'))
        time = parse_timestamp(commit.get('time'))
        if root is not None and time is not None:
            by_root.setdefault(root, []).append((time, commit))
    indexed = {}
    for root, rows in by_root.items():
        rows.sort(key=lambda row: (row[0], row[1]['sha']))
        indexed[root] = ([row[0] for row in rows], rows)

    repository_roots = {}
    for root, rows in by_root.items():
        for _, commit in rows:
            if commit.get('git_repository'):
                repository_roots.setdefault(commit['git_repository'], set()).add(root)

    links, seen = [], set()
    for session in sessions:
        session_id = session.get('session_id')
        if (not isinstance(session_id, (str, int)) or isinstance(session_id, bool)
                or session_id == ''):
            continue
        window = session_window(session)
        if window is None:
            continue
        source = session.get('source')
        profile = _value(session, 'profile')
        branch_id = session.get('branch_id')
        source_file = _value(session, 'source_file', 'session_file')
        identity = [value if isinstance(value, str) else None
                    for value in (source, profile, branch_id, source_file)]
        source, profile, branch_id, source_file = identity
        if not _declarations_agree(session, 'git_repository'):
            continue
        issues, branches = _session_references(session)
        if not _declarations_agree(session, 'git_branch'):
            branches = set()
        declared_repository = _value(session, 'git_repository')
        declared_repository_name = _repository_name(declared_repository)
        if declared_repository is not None and declared_repository_name is None:
            continue
        has_path = bool(_path_values(session))
        for root, (times, rows) in indexed.items():
            match = _consistent_path_match(session, root, indexed)
            if has_path and match is None:
                continue
            left, right = bisect_left(times, window[0]), bisect_right(times, window[1])
            for _, commit in rows[left:right]:
                repository = commit.get('git_repository')
                if (declared_repository_name is not None
                        and repository != declared_repository_name):
                    continue
                evidence = []
                basis = 'repository_path_and_time_window'
                if match is None:
                    if len(repository_roots.get(repository, ())) != 1:
                        continue
                    reference_match = _reference_match(commit, issues, branches)
                    if reference_match is None:
                        continue
                    basis, evidence = reference_match
                key = (str(root), commit['sha'], session_id, *identity)
                if key in seen:
                    continue
                seen.add(key)
                links.append({
                    'commit_sha': commit['sha'], 'git_repo_root': str(root),
                    'session_id': session_id, 'source': source, 'profile': profile,
                    'branch_id': branch_id, 'source_file': source_file,
                    'commit_time': commit['time'],
                    'session_start': _iso(window[0]), 'session_end': _iso(window[1]),
                    'path_field': match[0] if match else None,
                    'matched_path': match[1] if match else None,
                    'git_repository': repository, 'candidate': True,
                    'match_basis': basis, 'evidence': evidence,
                })
    return sorted(links, key=lambda row: (
        row['git_repo_root'], row['commit_time'], row['commit_sha'],
        str(row['session_id']), row['source'] or '', row['profile'] or '',
        row['branch_id'] or '', row['source_file'] or '',
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repositories', nargs='*', type=Path,
                        help='Local repository paths; no automatic discovery')
    parser.add_argument('--sessions', action='append', type=Path, default=[],
                        help='Exported conversation JSONL file/directory (repeatable)')
    parser.add_argument('--pull-requests', action='append', type=Path, default=[],
                        help='Exported GitHub PR JSON array (repeatable; no network)')
    parser.add_argument('--output-dir', type=Path, default=Path('extracted_data/git'))
    args = parser.parse_args(argv)
    print('=' * 80)
    print('GIT DATA EXTRACTION')
    print('=' * 80)
    print()
    if not args.repositories:
        print('No repositories supplied; Git extraction skipped.')
        return 0
    try:
        sessions = load_sessions(args.sessions)
        pull_requests = load_pull_requests(args.pull_requests)
        commits = extract_repositories(args.repositories, pull_requests)
        links = link_sessions(commits, sessions)
        print(f'Total commits: {len(commits):,}')
        print(f'Session links: {len(links):,}')
        if not commits and not pull_requests:
            print('No commits found!')
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        outputs = [
            (args.output_dir / f'git_commits_{timestamp}.jsonl', commits),
            (args.output_dir / f'git_session_links_{timestamp}.jsonl', links),
        ]
        if args.pull_requests:
            outputs.append((args.output_dir / f'git_pull_requests_{timestamp}.jsonl',
                            pull_requests))
        for path, _ in outputs:
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
        for path, records in outputs:
            with path.open('x', encoding='utf-8') as output:
                for record in records:
                    output.write(json.dumps(_json_safe(record), ensure_ascii=False) + '\n')
            print(f'✅ Saved to: {path}')
        print('   Format: JSONL (separate commits and session links)')
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

