#!/usr/bin/env python3
"""Combine extractor snapshots, keeping the latest record per conversation identity."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys


SNAPSHOT_TIME = re.compile(r'_(\d{8}_\d{6})(?:_(\d{6}))?\.jsonl$')
SESSION_IDS = ('session_id', 'composer_id', 'conversation_id', 'chat_id', 'tab_id')


def _value(record, *keys):
    metadata = record.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    for key in keys:
        for container in (record, metadata):
            if container.get(key) is not None:
                return container[key]
    return None


def conversation_identity(record):
    """Identify a session across snapshots, independent of its changing leaf IDs."""
    source = record.get('source')
    if not isinstance(source, str) or not source:
        return None
    for field in SESSION_IDS:
        identifier = record.get(field)
        if (isinstance(identifier, (str, int)) and not isinstance(identifier, bool)
                and identifier != ''):
            break
    else:
        return None
    source_file = _value(record, 'source_file', 'session_file')
    # File provenance is stronger than mutable project/cwd metadata. Where it
    # is absent, retain the existing installation/workspace namespace instead.
    namespace = None if source_file else [
        _value(record, 'installation'), _value(record, 'workspace_id'),
    ]
    return json.dumps([
        source, field, identifier, _value(record, 'profile'), source_file,
        namespace,
    ], sort_keys=True, ensure_ascii=True)


def _snapshot_order(path):
    # Prefer the exporter timestamp: copying an old snapshot can update its
    # mtime. Unknown filenames fall back to their local filesystem timestamp.
    match = SNAPSHOT_TIME.search(path.name)
    if match:
        stamp = match[1] + '_' + (match[2] or '000000')
        try:
            return datetime.strptime(stamp, '%Y%m%d_%H%M%S_%f'), str(path)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime), str(path)


def latest_conversations(paths):
    """Select snapshots deterministically without changing any input files.

    Records without a stable source/session identity are retained individually;
    similarity of their text is not enough evidence to discard a conversation.
    """
    files = {Path(path).resolve() for path in paths
             if not Path(path).name.startswith('ALL_CONVERSATIONS_')}
    selected = {}
    for path in sorted(files, key=_snapshot_order):
        snapshot = {}
        with path.open('r', encoding='utf-8') as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON in {path} at line {number}') from exc
                if not isinstance(record, dict) or not isinstance(record.get('messages'), list):
                    continue
                identity = conversation_identity(record)
                if identity is None:
                    selected[('row', str(path), number)] = [record]
                    continue
                branch = json.dumps(_value(record, 'branch_id'), sort_keys=True)
                snapshot.setdefault(identity, {})[branch] = record
        # A Pi/OMP leaf ID changes when a branch grows. Replace the session's
        # older snapshot as a group, retaining every fork in its latest export.
        for identity, branches in snapshot.items():
            selected[('identity', identity)] = list(branches.values())
    return [record for records in selected.values() for record in records]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='*', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {path.resolve() for path in args.snapshots}:
            raise ValueError('Combined output must differ from every input snapshot')
        # Validate every input before replacing a prior combined output.
        records = latest_conversations(args.snapshots)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8') as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
        return 0
    except (OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
