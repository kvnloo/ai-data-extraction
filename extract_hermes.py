#!/usr/bin/env python3
"""Extract Hermes conversations and usage from read-only state.db snapshots."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys


SESSION_FIELDS = (
    'source', 'model', 'title', 'started_at', 'ended_at', 'api_call_count',
    'input_tokens', 'cache_read_tokens', 'cache_write_tokens', 'output_tokens',
    'tool_call_count', 'parent_session_id',
)


def find_databases(root):
    """Find the default database and immediate profiles without scanning home."""
    root = Path(root)
    candidates = [(root / 'state.db', 'default')]
    candidates.extend((path, path.parent.name)
                      for path in sorted((root / 'profiles').glob('*/state.db')))
    found, seen = [], set()
    for path, profile in candidates:
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append((resolved, profile))
    return found


def _timestamp(value):
    """Hermes numeric timestamps are Unix seconds; preserve string timestamps."""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z')
        except (OverflowError, OSError, ValueError):
            return None
    return value if isinstance(value, str) else None


def _content(value):
    text = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
    if text is None:
        text = ''
    elif not isinstance(text, str):
        text = str(text)
    # Only recognize content-block arrays. Plain text, including JSON objects,
    # stays verbatim; retain the blocks as well so images are not discarded.
    if text.lstrip().startswith('['):
        try:
            blocks = json.loads(text)
        except json.JSONDecodeError:
            blocks = None
        if (isinstance(blocks, list) and blocks
                and all(isinstance(block, dict) and 'type' in block for block in blocks)):
            return {
                'content': '\n'.join(block['text'] for block in blocks
                                     if block.get('type') in ('text', 'input_text')
                                     and isinstance(block.get('text'), str)),
                'content_blocks': blocks,
            }
    return {'content': text}


def _tool_calls(value):
    """Keep usable calls and retain malformed input separately instead of losing it."""
    if value is None or value == '':
        return {}
    raw = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {'tool_calls_raw': raw}
    if parsed is None:
        return {}
    if not isinstance(parsed, list):
        return {'tool_calls_raw': raw}
    calls, malformed = [], False
    for call in parsed:
        if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
            malformed = True
            continue
        function = call['function']
        if not isinstance(function.get('name'), str):
            malformed = True
            continue
        function = dict(function)
        if 'arguments' in function and not isinstance(function['arguments'], str):
            function['arguments'] = json.dumps(function['arguments'], ensure_ascii=False)
        calls.append({**call, 'function': function})
    result = {'tool_calls': calls} if calls else {}
    if malformed:
        result['tool_calls_raw'] = raw
    return result


def _message(row):
    role = row.get('role')
    if role not in ('system', 'developer', 'user', 'assistant', 'tool'):
        return None
    message = {
        'role': 'system' if role == 'developer' else role,
        **_content(row.get('content')),
        'timestamp': _timestamp(row.get('timestamp')),
    }
    message.update(_tool_calls(row.get('tool_calls')))
    # Per-call counters belong to the recorded message, not the session total.
    # Preserve provider-specific token columns too, including explicit NULL/0;
    # an absent column stays absent rather than becoming an invented count.
    message.update({key: value for key, value in row.items()
                    if key.endswith('_tokens')})
    if role == 'tool':
        message['name'] = row.get('tool_name')
        # The baseline schema has no result ID. Never guess ownership from the
        # preceding assistant turn; extended databases can supply the real ID.
        if row.get('tool_call_id') is not None:
            message['tool_call_id'] = row['tool_call_id']
    return message


def extract_database(database, profile='default'):
    """Export one coherent snapshot, including committed WAL rows, without writes.

    Session counters and recorded per-message token columns are copied without
    inference. Empty sessions are retained with messages=[] because they can
    still carry usage information.
    Unreadable databases warn on stderr and return no records, allowing other
    profiles to export in the same best-effort run.
    """
    database = Path(database)
    if not database.is_file():
        return []
    source = snapshot = None
    try:
        database = database.resolve()
        source = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        snapshot = sqlite3.connect(':memory:')
        source.backup(snapshot)
        source.close()
        source = None
        snapshot.row_factory = sqlite3.Row

        messages = {}
        for record in snapshot.execute('SELECT * FROM messages ORDER BY timestamp, rowid'):
            row = dict(record)
            message = _message(row)
            if message is not None:
                messages.setdefault(row['session_id'], []).append(message)

        conversations = []
        for record in snapshot.execute('SELECT * FROM sessions ORDER BY started_at, id'):
            row = dict(record)
            metadata = {key: row.get(key) for key in SESSION_FIELDS}
            metadata['profile'] = profile
            conversation = {
                'messages': messages.get(row['id'], []),
                'source': 'hermes',
                'session_id': row['id'],
                'name': row.get('title'),
                'created_at': _timestamp(row.get('started_at')),
                'ended_at': _timestamp(row.get('ended_at')),
                'source_file': str(database),
                'installation': str(database.parent),
                'metadata': metadata,
            }
            if row.get('parent_session_id') is not None:
                conversation['parent_session'] = row['parent_session_id']
            # These columns are absent from the baseline Hermes schema. Export
            # them only when recorded; a database directory is not a session cwd.
            for key in ('cwd', 'git_repo_root'):
                if key in row:
                    conversation[key] = metadata[key] = row[key]
            conversations.append(conversation)
        return conversations
    except (sqlite3.Error, OSError) as exc:
        print(f'   ⚠️  state.db unreadable ({profile}): {exc}', file=sys.stderr)
        return []
    finally:
        if source is not None:
            source.close()
        if snapshot is not None:
            snapshot.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-dir', type=Path,
                        help='Hermes root (default: HERMES_HOME or ~/.hermes)')
    parser.add_argument('--output-dir', type=Path, default=Path('extracted_data'))
    args = parser.parse_args(argv)
    root = args.hermes_dir
    if root is None:
        override = os.environ.get('HERMES_HOME')
        root = Path(override) if override else Path.home() / '.hermes'

    print('=' * 80)
    print('HERMES DATA EXTRACTION')
    print('=' * 80)
    print()
    print('🔍 Searching for Hermes databases...')
    try:
        databases = find_databases(root)
        conversations = []
        for database, profile in databases:
            print(f'📂 Processing: {database} (profile: {profile})')
            records = extract_database(database, profile)
            conversations.extend(records)
            print(f'   ✅ {len(records)} conversations')

        print()
        print('=' * 80)
        print('EXTRACTION COMPLETE')
        print('=' * 80)
        print(f'Total conversations: {len(conversations):,}')
        if not conversations:
            print('No conversations found!')
            return 0
        print(f"Total messages: {sum(len(c['messages']) for c in conversations):,}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output_file = args.output_dir / f'hermes_conversations_{timestamp}.jsonl'
        with output_file.open('x', encoding='utf-8') as destination:
            for conversation in conversations:
                destination.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        print(f'✅ Saved to: {output_file}')
        print(f'   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB')
        print('   Format: JSONL (one conversation per line)')
        return 0
    except OSError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

