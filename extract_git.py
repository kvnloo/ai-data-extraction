#!/usr/bin/env python3
"""Extract local Git commits and conservative links to exported conversations.

Only explicitly supplied repositories are read. Commit ``time`` is the committer
time; ``author_time`` is retained separately. Links are candidates, not proof of
authorship: both the repository path and a closed session window must match.
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


ISSUE_REF = re.compile(
    r'https?://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/[1-9]\d*\b'
    r'|(?<![\w/.-])(?:[\w.-]+/[\w.-]+)?#[1-9]\d*\b'
)


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


def extract_repository(root):
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
        })
    return sorted(commits, key=lambda row: (row['time'], row['sha']))


def extract_repositories(paths):
    roots = list(dict.fromkeys(repository_root(path) for path in paths))
    return [commit for root in roots for commit in extract_repository(root)]


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


def link_sessions(commits, sessions):
    """Join path-matched session intervals to commit times; keep all candidates."""
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
        for root, (times, rows) in indexed.items():
            match = _path_match(session, root, indexed)
            if match is None:
                continue
            left, right = bisect_left(times, window[0]), bisect_right(times, window[1])
            for _, commit in rows[left:right]:
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
                    'path_field': match[0], 'matched_path': match[1],
                    'match_basis': 'repository_path_and_time_window',
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
        commits = extract_repositories(args.repositories)
        links = link_sessions(commits, sessions)
        print(f'Total commits: {len(commits):,}')
        print(f'Session links: {len(links):,}')
        if not commits:
            print('No commits found!')
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        outputs = [
            (args.output_dir / f'git_commits_{timestamp}.jsonl', commits),
            (args.output_dir / f'git_session_links_{timestamp}.jsonl', links),
        ]
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
