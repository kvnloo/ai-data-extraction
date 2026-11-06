from datetime import datetime
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import extract_hermes
from filter_privacy import filter_jsonl


class ExtractHermesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home_guard = patch.object(Path, 'home', side_effect=AssertionError('real home must not be read'))
        home_guard.start()
        self.addCleanup(home_guard.stop)

    def database(self, relative='state.db', *, wal=False, optional=False,
                 session_id='session-1', **session_values):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        self.addCleanup(connection.close)
        if wal:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA wal_autocheckpoint=0')
        connection.execute('''CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, model TEXT, title TEXT,
            started_at REAL, ended_at REAL, api_call_count INTEGER,
            input_tokens INTEGER, cache_read_tokens INTEGER,
            cache_write_tokens INTEGER, output_tokens INTEGER,
            tool_call_count INTEGER, parent_session_id TEXT)''')
        connection.execute('''CREATE TABLE messages (
            session_id TEXT, role TEXT, content TEXT, tool_calls TEXT,
            tool_name TEXT, timestamp REAL)''')
        if optional:
            connection.execute('ALTER TABLE sessions ADD COLUMN cwd TEXT')
            connection.execute('ALTER TABLE sessions ADD COLUMN git_repo_root TEXT')
            connection.execute('ALTER TABLE messages ADD COLUMN tool_call_id TEXT')
        session = {
            'id': session_id, 'source': 'cli', 'model': 'fixture-model',
            'title': 'Inspect files', 'started_at': 1000, 'ended_at': 1030,
            'api_call_count': 2, 'input_tokens': 100, 'cache_read_tokens': 20,
            'cache_write_tokens': 10, 'output_tokens': 30, 'tool_call_count': 1,
            'parent_session_id': None, **session_values,
        }
        columns = ', '.join(session)
        parameters = ', '.join('?' for _ in session)
        connection.execute(f'INSERT INTO sessions ({columns}) VALUES ({parameters})', tuple(session.values()))
        connection.commit()
        return path, connection

    def add_messages(self, connection, rows):
        connection.executemany('''INSERT INTO messages
            (session_id, role, content, tool_calls, tool_name, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)''', rows)
        connection.commit()

    def test_preserves_schema_metadata_tools_and_chronological_order(self):
        path, writer = self.database(parent_session_id='parent-1')
        calls = [{'id': 'call-1', 'type': 'function',
                  'function': {'name': 'read', 'arguments': {'path': 'a.py'}}}]
        self.add_messages(writer, [
            ('session-1', 'assistant', 'Done', None, None, 1030),
            ('session-1', 'user', 'Inspect files', None, None, 1001),
            ('session-1', 'assistant', None, json.dumps(calls), None, 1002),
            ('session-1', 'tool', 'a.py', None, 'read', 1003),
        ])
        original = path.read_bytes()
        conversation, = extract_hermes.extract_database(path, 'review')
        self.assertEqual(conversation['source'], 'hermes')
        self.assertEqual(conversation['session_id'], 'session-1')
        self.assertEqual(conversation['name'], 'Inspect files')
        self.assertEqual(conversation['created_at'], '1970-01-01T00:16:40Z')
        self.assertEqual(conversation['ended_at'], '1970-01-01T00:17:10Z')
        self.assertEqual(conversation['source_file'], str(path.resolve()))
        self.assertEqual(conversation['installation'], str(path.parent.resolve()))
        self.assertEqual(conversation['parent_session'], 'parent-1')
        self.assertEqual(conversation['metadata'], {
            'source': 'cli', 'model': 'fixture-model', 'title': 'Inspect files',
            'started_at': 1000, 'ended_at': 1030, 'api_call_count': 2,
            'input_tokens': 100, 'cache_read_tokens': 20, 'cache_write_tokens': 10,
            'output_tokens': 30, 'tool_call_count': 1,
            'parent_session_id': 'parent-1', 'profile': 'review',
        })
        messages = conversation['messages']
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(messages[1]['content'], '')
        self.assertEqual(messages[1]['tool_calls'][0]['function'],
                         {'name': 'read', 'arguments': '{"path": "a.py"}'})
        self.assertEqual(messages[2]['name'], 'read')
        self.assertEqual(messages[2]['content'], 'a.py')
        self.assertNotIn('tool_call_id', messages[2])
        self.assertNotIn('cwd', conversation)
        self.assertNotIn('git_repo_root', conversation)
        self.assertEqual(path.read_bytes(), original)

    def test_preserves_recorded_per_call_token_columns_without_session_inference(self):
        path, writer = self.database()
        fields = ('input_tokens', 'output_tokens', 'cache_read_tokens',
                  'cache_write_tokens', 'reasoning_tokens')
        for field in fields:
            writer.execute(f'ALTER TABLE messages ADD COLUMN {field} INTEGER')
        self.add_messages(writer, [
            ('session-1', 'user', 'Inspect files', None, None, 1001),
            ('session-1', 'assistant', 'First call', None, None, 1002),
            ('session-1', 'assistant', 'Second call', None, None, 1003),
        ])
        writer.execute('''UPDATE messages SET input_tokens=123, output_tokens=45,
            cache_read_tokens=0, cache_write_tokens=7, reasoning_tokens=11
            WHERE timestamp=1002''')
        writer.execute('''UPDATE messages SET input_tokens=9, output_tokens=0,
            cache_read_tokens=2, cache_write_tokens=NULL, reasoning_tokens=0
            WHERE timestamp=1003''')
        writer.commit()
        original = path.read_bytes()

        conversation, = extract_hermes.extract_database(path)
        messages = conversation['messages']
        self.assertEqual({field: messages[0][field] for field in fields},
                         dict.fromkeys(fields))
        self.assertEqual({field: messages[1][field] for field in fields}, {
            'input_tokens': 123, 'output_tokens': 45, 'cache_read_tokens': 0,
            'cache_write_tokens': 7, 'reasoning_tokens': 11,
        })
        self.assertEqual({field: messages[2][field] for field in fields}, {
            'input_tokens': 9, 'output_tokens': 0, 'cache_read_tokens': 2,
            'cache_write_tokens': None, 'reasoning_tokens': 0,
        })
        self.assertEqual(conversation['metadata']['input_tokens'], 100)
        self.assertEqual(conversation['metadata']['output_tokens'], 30)
        self.assertEqual(path.read_bytes(), original)

    def test_missing_message_token_columns_remain_absent(self):
        path, writer = self.database()
        self.add_messages(writer, [
            ('session-1', 'assistant', 'No recorded call usage', None, None, 1002),
        ])
        conversation, = extract_hermes.extract_database(path)
        self.assertEqual(conversation['messages'], [{
            'role': 'assistant', 'content': 'No recorded call usage',
            'timestamp': '1970-01-01T00:16:42Z',
        }])
        self.assertEqual(conversation['metadata']['input_tokens'], 100)

    def test_discovers_all_profiles_without_collapsing_equal_session_ids(self):
        default, _ = self.database()
        alpha, _ = self.database('profiles/alpha/state.db')
        beta, _ = self.database('profiles/beta/state.db')
        self.database('profiles/beta/nested/state.db')
        databases = extract_hermes.find_databases(self.root)
        self.assertEqual(databases, [(default.resolve(), 'default'),
                                     (alpha.resolve(), 'alpha'), (beta.resolve(), 'beta')])
        records = [record for path, profile in databases
                   for record in extract_hermes.extract_database(path, profile)]
        self.assertEqual([r['session_id'] for r in records], ['session-1'] * 3)
        self.assertEqual({r['metadata']['profile'] for r in records}, {'default', 'alpha', 'beta'})
        self.assertEqual(len({r['source_file'] for r in records}), 3)

    def test_read_only_snapshot_includes_live_wal_and_escaped_path(self):
        path, writer = self.database('uri #?% space/state.db', wal=True)
        self.add_messages(writer, [('session-1', 'assistant', 'committed WAL turn', None, None, 1002)])
        files = [path, Path(str(path) + '-wal')]
        before = [file.read_bytes() for file in files]
        connect = sqlite3.connect
        source_connections = []

        def check_read_only(database, *args, **kwargs):
            connection = connect(database, *args, **kwargs)
            if database != ':memory:':
                source_connections.append(database)
                try:
                    with self.assertRaisesRegex(sqlite3.OperationalError, 'readonly'):
                        try:
                            connection.execute('DELETE FROM sessions')
                        finally:
                            connection.rollback()
                except BaseException:
                    connection.close()
                    raise
            return connection

        with patch.object(extract_hermes.sqlite3, 'connect', side_effect=check_read_only):
            conversation, = extract_hermes.extract_database(path)
        self.assertEqual(len(source_connections), 1)
        self.assertEqual(conversation['messages'][0]['content'], 'committed WAL turn')
        self.assertEqual(before, [file.read_bytes() for file in files])

    def test_malformed_calls_do_not_discard_other_messages(self):
        path, writer = self.database()
        valid_call = {'id': 'call-1', 'type': 'function',
                      'function': {'name': 'read', 'arguments': '{}'}}
        mixed = json.dumps([17, valid_call])
        self.add_messages(writer, [
            ('session-1', 'assistant', 'kept', '{broken', None, 1001),
            ('session-1', 'assistant', None, mixed, None, 1002),
            ('session-1', 'assistant', 'last', '42', None, 1003),
        ])
        conversation, = extract_hermes.extract_database(path)
        messages = conversation['messages']
        self.assertEqual([m['content'] for m in messages], ['kept', '', 'last'])
        self.assertEqual(messages[0]['tool_calls_raw'], '{broken')
        self.assertEqual(messages[1]['tool_calls'], [valid_call])
        self.assertEqual(messages[1]['tool_calls_raw'], mixed)
        self.assertEqual(messages[2]['tool_calls_raw'], '42')

    def test_preserves_content_blocks_and_optional_recorded_paths_and_call_id(self):
        repository = self.root / 'project'
        path, writer = self.database(optional=True, cwd=str(repository / 'src'),
                                     git_repo_root=str(repository))
        blocks = [{'type': 'text', 'text': 'inspect image'},
                  {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        self.add_messages(writer, [
            ('session-1', 'user', json.dumps(blocks), None, None, 1001),
            ('session-1', 'tool', 'result', None, 'read', 1002),
        ])
        writer.execute("UPDATE messages SET tool_call_id='call-recorded' WHERE role='tool'")
        writer.commit()
        conversation, = extract_hermes.extract_database(path)
        self.assertEqual(conversation['messages'][0]['content'], 'inspect image')
        self.assertEqual(conversation['messages'][0]['content_blocks'], blocks)
        self.assertEqual(conversation['messages'][1]['tool_call_id'], 'call-recorded')
        self.assertEqual(conversation['cwd'], str(repository / 'src'))
        self.assertEqual(conversation['git_repo_root'], str(repository))
        self.assertEqual(conversation['metadata']['cwd'], str(repository / 'src'))

    def test_empty_session_keeps_unknown_counts_and_open_window(self):
        path, _ = self.database(input_tokens=None, ended_at=None, tool_call_count=0)
        conversation, = extract_hermes.extract_database(path)
        self.assertEqual(conversation['messages'], [])
        self.assertIsNone(conversation['metadata']['input_tokens'])
        self.assertEqual(conversation['metadata']['tool_call_count'], 0)
        self.assertIsNone(conversation['ended_at'])

    def test_missing_or_invalid_database_does_not_create_or_change_source(self):
        missing = self.root / 'missing' / 'state.db'
        self.assertEqual(extract_hermes.extract_database(missing), [])
        self.assertFalse(missing.parent.exists())
        broken = self.root / 'broken.db'
        broken.write_bytes(b'not a sqlite database')
        with patch('sys.stderr', new_callable=io.StringIO) as errors:
            self.assertEqual(extract_hermes.extract_database(broken), [])
        self.assertIn('state.db unreadable', errors.getvalue())
        self.assertEqual(broken.read_bytes(), b'not a sqlite database')

    def test_cli_uses_explicit_root_and_writes_filter_compatible_jsonl(self):
        path, writer = self.database(source='private-channel')
        self.add_messages(writer, [('session-1', 'user', 'private-marker', None, None, 1001)])
        original = path.read_bytes()
        output = self.root / 'output'
        with patch('sys.stdout', new_callable=io.StringIO) as stdout:
            result = extract_hermes.main(['--hermes-dir', str(self.root), '--output-dir', str(output)])
        self.assertEqual(result, 0)
        self.assertIn('Total conversations: 1', stdout.getvalue())
        exported, = output.glob('hermes_conversations_*.jsonl')
        raw = json.loads(exported.read_text(encoding='utf-8'))
        self.assertEqual(raw['metadata']['source'], 'private-channel')
        filtered = self.root / 'filtered.jsonl'
        stats = filter_jsonl(exported, filtered, lambda text: text.replace('private-', 'redacted-'))
        record = json.loads(filtered.read_text(encoding='utf-8'))
        self.assertEqual(record['messages'][0]['content'], 'redacted-marker')
        self.assertEqual(record['metadata']['source'], 'redacted-channel')
        self.assertEqual(stats.records, 1)
        self.assertEqual(path.read_bytes(), original)

    def test_same_second_exports_preserve_both_snapshots(self):
        _, writer = self.database()
        self.add_messages(writer, [
            ('session-1', 'assistant', 'First snapshot', None, None, 1002),
        ])
        output = self.root / 'output'
        arguments = ['--hermes-dir', str(self.root), '--output-dir', str(output)]
        with patch.object(extract_hermes, 'datetime', wraps=datetime) as clock, \
                patch('sys.stdout', new_callable=io.StringIO):
            clock.now.side_effect = [datetime(2026, 1, 1, 10, 0, 0, 100),
                                     datetime(2026, 1, 1, 10, 0, 0, 200)]
            self.assertEqual(extract_hermes.main(arguments), 0)
            first, = output.glob('hermes_conversations_*.jsonl')
            original = first.read_bytes()
            writer.execute("UPDATE messages SET content='Second snapshot'")
            writer.commit()
            self.assertEqual(extract_hermes.main(arguments), 0)

        exports = sorted(output.glob('hermes_conversations_*.jsonl'))
        self.assertEqual(len(exports), 2)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(
            [json.loads(path.read_text(encoding='utf-8'))['messages'][0]['content']
             for path in exports],
            ['First snapshot', 'Second snapshot'],
        )

    def test_exact_export_name_collision_fails_without_overwriting_snapshot(self):
        _, writer = self.database()
        self.add_messages(writer, [
            ('session-1', 'assistant', 'Original snapshot', None, None, 1002),
        ])
        output = self.root / 'output'
        arguments = ['--hermes-dir', str(self.root), '--output-dir', str(output)]
        with patch.object(extract_hermes, 'datetime', wraps=datetime) as clock, \
                patch('sys.stdout', new_callable=io.StringIO), \
                patch('sys.stderr', new_callable=io.StringIO) as errors:
            clock.now.return_value = datetime(2026, 1, 1, 10, 0, 0, 100)
            self.assertEqual(extract_hermes.main(arguments), 0)
            exported, = output.glob('hermes_conversations_*.jsonl')
            original = exported.read_bytes()
            writer.execute("UPDATE messages SET content='Changed snapshot'")
            writer.commit()
            self.assertEqual(extract_hermes.main(arguments), 1)

        self.assertEqual(list(output.glob('hermes_conversations_*.jsonl')), [exported])
        self.assertEqual(exported.read_bytes(), original)
        self.assertIn('error:', errors.getvalue())

    def test_cli_uses_environment_root_without_home_discovery(self):
        self.database()
        with patch.dict(os.environ, {'HERMES_HOME': str(self.root)}), \
                patch('sys.stdout', new_callable=io.StringIO):
            result = extract_hermes.main(['--output-dir', str(self.root / 'output')])
        self.assertEqual(result, 0)
        self.assertEqual(len(list((self.root / 'output').glob('hermes_conversations_*.jsonl'))), 1)

    def test_default_discovery_uses_only_synthetic_home(self):
        self.database('.hermes/state.db')
        with patch.object(Path, 'home', return_value=self.root), \
                patch.dict(os.environ, {'HERMES_HOME': ''}), \
                patch('sys.stdout', new_callable=io.StringIO):
            result = extract_hermes.main(['--output-dir', str(self.root / 'output')])
        self.assertEqual(result, 0)
        exported, = (self.root / 'output').glob('hermes_conversations_*.jsonl')
        self.assertEqual(json.loads(exported.read_text())['metadata']['profile'], 'default')


if __name__ == '__main__':
    unittest.main()

