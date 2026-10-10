#!/usr/bin/env python3
"""
Extract ALL Claude Code chat data from all projects
Includes: messages, code context, diffs, file references
Sub-agent transcripts are exported as conversations of their own
A file or directory name that is not valid text is written as bytes://<percent-encoded bytes>
Auto-discovers Claude Code installations on the device
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import os
import sys
from urllib.parse import quote

def find_claude_installations():
    """Find all Claude Code installation directories"""
    system = platform.system()
    home = Path.home()

    # Common installation locations by OS
    locations = []

    if system == "Darwin":  # macOS
        base_dirs = [
            home / "Library/Application Support",
            home / ".config"
        ]
    elif system == "Linux":
        base_dirs = [
            home / ".config",
            home / ".local/share"
        ]
    elif system == "Windows":
        base_dirs = [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')),
            Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local'))
        ]
    else:
        base_dirs = [home / ".config"]

    # Search for Claude-related directories
    claude_patterns = [
        'claude', 'claude-code', 'claude-local', 'claude-m2', 'claude-zai',
        '.claude', '.claude-code', '.claude-local', '.claude-m2', '.claude-zai'
    ]

    # An explicitly configured directory is where the sessions actually are
    configured = os.environ.get('CLAUDE_CONFIG_DIR')
    if configured and Path(configured).exists():
        locations.append(Path(configured))

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        # Check direct children
        for pattern in claude_patterns:
            claude_dir = base_dir / pattern
            if claude_dir.exists():
                locations.append(claude_dir)

    # Also check home directory directly, whether or not the directories above exist
    for pattern in claude_patterns:
        home_dir = home / pattern
        if home_dir.exists():
            locations.append(home_dir)

    return list(dict.fromkeys(locations))  # Remove duplicates, keep discovery order


ESCAPED_NAME = 'bytes://'


def _name(text):
    """A file or directory name, or a path, as text every output can hold

    A name is bytes and not every name is text. One that is valid text is returned unchanged.
    One that is not is returned as 'bytes://' followed by its bytes percent-encoded ('/' is
    kept, '%' becomes '%25'): agent-<byte FF>.jsonl gives the agent id 'bytes://%FF'.
    urllib.parse.unquote_to_bytes(value[len('bytes://'):]) gives the bytes back.

    No name returned unchanged holds '://' (a file name has no '/', and a path never repeats
    one after its start), so an escaped name is never mistaken for a real one, and the same
    name is escaped the same way wherever it is written.

    The session id a transcript's entries give is written the same way, so that it still
    equals the id of the session it names; one that is not text and not a name either is
    escaped from its UTF-8 form with the unpaired surrogates kept.
    """
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        try:
            raw = os.fsencode(text)
        except UnicodeEncodeError:
            raw = text.encode('utf-8', 'surrogatepass')
        return ESCAPED_NAME + quote(raw)
    return text


def _say(text):
    """Print a line that can hold a path; what the terminal cannot show is escaped, not raised"""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(text.encode(encoding, 'backslashreplace').decode(encoding))


ENTRY_FLAGS = {'isSidechain': 'is_sidechain', 'isMeta': 'is_meta', 'isCompactSummary': 'is_compact_summary'}


def _identity(obj):
    """Source entry identity, so an exported message can be traced back to the transcript"""
    identity = {'id': obj.get('uuid'), 'parent_id': obj.get('parentUuid')}
    # A sub-agent turn, an injected note and a compaction summary are not ordinary turns
    identity.update({name: True for key, name in ENTRY_FLAGS.items() if obj.get(key)})
    return identity


def _block_text(content):
    """Text of a tool result, which is either a string or a list of text blocks"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(item if isinstance(item, str) else item.get('text', '') for item in content
                         if isinstance(item, str) or (isinstance(item, dict) and item.get('type') == 'text'))
    return ''


def _omitted_blocks(content):
    """What a tool result held besides text (an image, a document), by type and size, without its bytes"""
    blocks = []
    for item in content if isinstance(content, list) else []:
        if isinstance(item, dict) and item.get('type') != 'text':
            source = item.get('source') or {}
            blocks.append({'type': item.get('type'), 'media_type': source.get('media_type'),
                           'encoded_size': len(source.get('data') or '')})
    return blocks

def _messages_digest(messages):
    """One value for everything a conversation's messages say, so two copies can be compared"""
    return hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode()).hexdigest()


def _follows_parent_chain(conversation):
    """Whether no message comes before the message its parent_id names

    A parent_id that names an entry left out of the export is followed through
    'omitted_entries' to the first exported entry behind it.
    """
    messages = conversation['messages']
    omitted = conversation['omitted_entries']
    position = {}
    for index, message in enumerate(messages):
        position.setdefault(message['id'], index)
    for index, message in enumerate(messages):
        parent = message.get('parent_id')
        passed = set()
        while isinstance(parent, str) and parent in omitted and parent not in passed:
            passed.add(parent)
            parent = omitted[parent].get('parent_id')
        if isinstance(parent, str) and position.get(parent, index) > index:
            return False
    return True


def _comparable_copy(conversation, nested):
    """What is needed to compare a sub-agent conversation with another copy of it, or None

    None when its messages carry no real entry ids (two such transcripts cannot be told
    apart) or cannot be digested; such a conversation is exported as it is, never merged.
    """
    messages = conversation['messages']
    if not all(isinstance(message.get('id'), str) and message['id'] for message in messages):
        return None
    try:
        digests = [_messages_digest([message]) for message in messages]
    except (RecursionError, TypeError, ValueError):
        return None
    return {'conversation': conversation, 'digests': digests, 'nested': nested,
            'in_chain_order': _follows_parent_chain(conversation)}


def _shared_leading(first, second):
    """How many messages two copies have in common from the start, by digest"""
    shared = 0
    for one, other in zip(first, second):
        if one != other:
            break
        shared += 1
    return shared


def _same_transcript(old, new):
    """Whether two copies of one agent's transcript in one session are the same transcript

    Returns None when they are not, else (kept, left out, field, record): the record goes
    into the kept conversation's list `field`; an exact copy has no record. `old` was found
    first.
    """
    first, second = old['digests'], new['digests']
    if first == second:
        return old, new, None, None
    shared = _shared_leading(first, second)
    if shared == min(len(first), len(second)):
        # One stops where the other goes on: a file copied while it was still being written
        kept, left_out = (old, new) if len(first) > len(second) else (new, old)
        return kept, left_out, 'prefix_source_files', {'shared_leading_messages': shared}
    if sorted(first) == sorted(second):
        # The same messages, each as many times, in another order
        if old['in_chain_order'] != new['in_chain_order']:
            kept, rule = (old if old['in_chain_order'] else new), 'parent_id_chain'
        elif old['nested'] != new['nested']:
            kept, rule = (old if old['nested'] else new), 'nested_layout'
        else:
            kept, rule = old, 'first_found'
        return kept, (new if kept is old else old), 'reordered_source_files', {'kept_by': rule}
    return None


COPY_RECORD_FIELDS = ('prefix_source_files', 'reordered_source_files')


def _leave_out(kept, left_out, field, record):
    """Record on the kept conversation the copy that is not exported, and all that copy stood for"""
    conversation, other = kept['conversation'], left_out['conversation']
    conversation.setdefault('duplicate_source_files', []).append(other['source_file'])
    conversation['duplicate_source_files'].extend(other.get('duplicate_source_files', []))
    if record is not None:
        # 'compared_with' is the kept file now; it stays true if another copy later replaces that one
        conversation.setdefault(field, []).append(
            {'source_file': other['source_file'], 'compared_with': conversation['source_file'], **record})
    for name in COPY_RECORD_FIELDS:
        if other.get(name):
            conversation.setdefault(name, []).extend(other[name])
    # The meta file sits beside only one of the copies
    for name in META_FIELDS.values():
        if name in other:
            conversation.setdefault(name, other[name])


META_FILE_LIMIT = 64 * 1024
META_FIELDS = {'agentType': 'agent_type', 'description': 'agent_description'}


def _agent_meta(agent_file):
    """What kind of sub-agent this was and what it was asked for, from the small meta file beside its transcript"""
    meta_file = agent_file.with_name(agent_file.stem + '.meta.json')
    try:
        if not meta_file.is_file() or meta_file.stat().st_size > META_FILE_LIMIT:
            return {}
        meta = json.loads(meta_file.read_text())
    except (OSError, ValueError, RecursionError):
        return {}
    if not isinstance(meta, dict):
        return {}
    # Only these two fields; the file can hold anything else
    return {name: meta[key] for key, name in META_FIELDS.items() if isinstance(meta.get(key), str)}


def _session_files(container):
    """Session transcripts in a directory, and the sub-agent transcripts that belong to them

    A sub-agent transcript is agent-<id>.jsonl, either beside the session files or under
    <session-id>/subagents/. Returns (session files, [(sub-agent file, parent session id or None)]);
    the parent of a file beside the session files is only known from its entries.
    """
    # Sessions keep the order the directory lists them in, as they always have
    sessions = [f for f in container.glob('*.jsonl') if not f.name.startswith('agent-')]
    subagents = [(f, None) for f in sorted(container.glob('agent-*.jsonl'))]
    subagents += [(f, _name(f.relative_to(container).parts[0]))
                  for f in sorted(container.glob('*/subagents/**/agent-*.jsonl'))]
    return sessions, [(f, parent) for f, parent in subagents if f.is_file()]


def _convert_session_file(jsonl_file, project_dir, project_name):
    """Convert one transcript, a session's or a sub-agent's, into a conversation

    Returns (conversation or None, the session id its entries name).
    """
    messages = []
    omitted_attachments = {}
    omitted_entries = {}
    skipped_lines = 0
    session_id = _name(jsonl_file.stem)
    project_path = None
    entry_session_id = None

    with open(jsonl_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            exported = len(messages)
            try:
                obj = json.loads(line)
                msg_type = obj.get('type')
                if entry_session_id is None and obj.get('sessionId'):
                    entry_session_id = obj['sessionId']

                if msg_type == 'user':
                    message = obj.get('message', {})
                    content = message.get('content', '')

                    # Tool output is stored as a user entry; it is not something the user said
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'tool_result':
                                tool_msg = {
                                    'role': 'tool',
                                    'tool_call_id': item.get('tool_use_id'),
                                    'content': _block_text(item.get('content')),
                                    'is_error': bool(item.get('is_error')),
                                    'timestamp': obj.get('timestamp'),
                                    **_identity(obj)
                                }
                                omitted = _omitted_blocks(item.get('content'))
                                if omitted:
                                    tool_msg['omitted_content'] = omitted
                                messages.append(tool_msg)
                        content = [item for item in content
                                   if not (isinstance(item, dict) and item.get('type') == 'tool_result')]

                    if content:
                        msg = {
                            'role': 'user',
                            'content': content,
                            'timestamp': obj.get('timestamp'),
                            **_identity(obj)
                        }

                        # Extract tool use (code context, diffs, etc.)
                        if 'toolUse' in obj:
                            msg['tool_use'] = obj['toolUse']

                        messages.append(msg)

                    # Extract working directory
                    if 'cwd' in obj:
                        project_path = obj['cwd']

                elif msg_type == 'assistant':
                    message = obj.get('message', {})
                    content = message.get('content', [])

                    # Extract text from content array
                    text_parts = []
                    code_blocks = []
                    tool_uses = []
                    thinking = []

                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    text_parts.append(item.get('text', ''))
                                elif item.get('type') == 'thinking' and item.get('thinking'):
                                    thinking.append(item['thinking'])
                                elif item.get('type') == 'tool_use':
                                    # Code execution, file edits, etc.
                                    tool_uses.append(item)
                    elif isinstance(content, str):
                        text_parts.append(content)

                    full_text = '\n'.join(text_parts)
                    if full_text or tool_uses or thinking:
                        msg = {
                            'role': 'assistant',
                            'content': full_text,
                            'model': message.get('model'),
                            'timestamp': obj.get('timestamp'),
                            **_identity(obj)
                        }

                        # One response can be written as several entries that repeat the
                        # same usage; message_id lets a consumer count it once
                        if message.get('id'):
                            msg['message_id'] = message['id']
                        if obj.get('requestId'):
                            msg['request_id'] = obj['requestId']
                        usage = message.get('usage')
                        if isinstance(usage, dict):
                            msg['usage'] = dict(usage)

                        if thinking:
                            msg['reasoning'] = '\n'.join(thinking)

                        if tool_uses:
                            msg['tool_uses'] = tool_uses

                        messages.append(msg)

                elif msg_type == 'attachment':
                    # Context a hook added is something the model read but the user never typed
                    attachment = obj.get('attachment') or {}
                    if attachment.get('type') == 'hook_additional_context':
                        messages.append({
                            'role': 'system',
                            'content': _block_text(attachment.get('content')),
                            'timestamp': obj.get('timestamp'),
                            'custom': True,
                            'custom_type': 'hook_additional_context',
                            'hook_event': attachment.get('hookEvent'),
                            **_identity(obj)
                        })
                    else:
                        # Not conversation, but say what was left out
                        kind = attachment.get('type') or 'unknown'
                        omitted_attachments[kind] = omitted_attachments.get(kind, 0) + 1

                elif msg_type == 'tool_result':
                    # Capture tool results (diffs, file reads, etc.)
                    tool_result = obj.get('toolResult', {})
                    if tool_result and messages:
                        # Add to last assistant message
                        if 'tool_results' not in messages[-1]:
                            messages[-1]['tool_results'] = []
                        messages[-1]['tool_results'].append(tool_result)

                if len(messages) == exported and obj.get('uuid'):
                    # Nothing was exported for this entry; keep its link so a parent_id
                    # that names it can still be followed
                    omitted = {'type': msg_type, 'parent_id': obj.get('parentUuid')}
                    if obj.get('logicalParentUuid'):
                        omitted['logical_parent_id'] = obj['logicalParentUuid']
                    omitted_entries[obj['uuid']] = omitted

            except (json.JSONDecodeError, AttributeError, TypeError):
                # A line that is not a well-formed entry costs that line, not the session
                del messages[exported:]
                skipped_lines += 1
                continue

    if not messages:
        return None, entry_session_id

    return {
        'messages': messages,
        'source': 'claude-code',
        'session_id': session_id,
        'project_path': project_path,
        'project_name': project_name,
        'omitted_attachments': omitted_attachments,
        'omitted_entries': omitted_entries,
        'skipped_lines': skipped_lines,
        'source_file': _name(str(jsonl_file)),
        'installation': _name(str(project_dir))
    }, entry_session_id


def extract_claude_project_conversations(project_dir, include_subagents=True):
    """Extract conversations from a Claude project directory with full context

    A sub-agent's transcript is exported as a conversation of its own, marked 'subagent' and
    linked to its session by 'parent_session'; the session lists its sub-agents in
    'subagent_sessions' whether or not they are exported.

    One transcript can be found in several files. Copies are compared message by message,
    by content, and only when every message has an entry id:

    - Files for the same session and agent that hold the same transcript are exported once,
      and the files left out are listed in 'duplicate_source_files'. That covers a copy that
      is the same, a copy that stops early ('prefix_source_files' on the longer one that is
      kept: each file and its 'shared_leading_messages'), and a copy with the same messages,
      each as many times, in another order ('reordered_source_files': each file and
      'kept_by', the rule that chose the kept one: 'parent_id_chain' when only it has no
      message before its parent, else 'nested_layout' for the copy under <session-id>/subagents/,
      else 'first_found'). Each of those records names in 'compared_with' the file it was
      compared with, which is the exported one unless that file was itself left out later.
    - Files for the same session and agent that begin alike and then differ are all
      exported; each lists the others in 'shared_prefix_with' ('source_file' and
      'shared_leading_messages').
    - The same transcript (the same messages, in any order) for the same agent under different
      sessions of a project is exported for each; each lists the others in
      'same_transcript_in' ('parent_session' and 'source_file').

    To count every message once: skip a conversation when 'same_transcript_in' names a
    'source_file' that sorts before its own; otherwise skip its first n messages, n being the
    largest 'shared_leading_messages' in 'shared_prefix_with' among files that sort before it.

    Every name taken from a file or directory ('session_id', 'agent', 'parent_session',
    'subagent_sessions', 'project_name', 'source_file', 'duplicate_source_files',
    'installation', and 'source_file' and 'parent_session' in the fields above) is written by
    _name: unchanged when it is text, escaped when it is not.
    """
    project_dir = Path(project_dir)
    conversations = []
    subagent_conversations = []

    if (project_dir / 'projects').exists():
        # New structure: projects/project-name/session.jsonl
        containers = [proj for proj in (project_dir / 'projects').iterdir() if proj.is_dir()]
    else:
        # Old structure: direct JSONL files
        containers = [project_dir]

    def convert(jsonl_file, project_name):
        try:
            return _convert_session_file(jsonl_file, project_dir, project_name)
        except Exception as e:
            _say(f"Error processing {jsonl_file}: {e}")
            return None, None

    for container in containers:
        project_name = _name(container.name) if container.name != 'projects' else None
        session_files, subagent_files = _session_files(container)

        sessions = {}
        for jsonl_file in session_files:
            conversation, _ = convert(jsonl_file, project_name)
            if conversation:
                conversations.append(conversation)
                sessions[conversation['session_id']] = conversation

        agents_of = {}
        copies = {}  # (parent session, agent) -> the exported copies that can be compared
        for agent_file, parent_session in subagent_files:
            nested = parent_session is not None
            agent = _name(agent_file.stem[len('agent-'):])
            conversation, entry_session_id = convert(agent_file, project_name)
            # Only a string can name a session; anything else in that field is noise
            if not parent_session and isinstance(entry_session_id, str):
                parent_session = _name(entry_session_id.strip()) or None
            if parent_session:
                agents_of.setdefault(parent_session, []).append(agent)
            if conversation:
                conversation.update(subagent=True, agent=agent, parent_session=parent_session,
                                    **_agent_meta(agent_file))
                # The same transcript can sit in both layouts; exported twice, its usage counts twice
                copy = _comparable_copy(conversation, nested)
                if copy is not None:
                    kept_copies = copies.setdefault((parent_session, agent), [])
                    for old in list(kept_copies):
                        same = _same_transcript(old, copy)
                        if same is None:
                            continue
                        kept, left_out, field, record = same
                        _leave_out(kept, left_out, field, record)
                        if kept is old:
                            copy = None
                            break
                        kept_copies[:] = [c for c in kept_copies if c is not old]
                        subagent_conversations[:] = [c for c in subagent_conversations
                                                     if c is not old['conversation']]
                    if copy is None:
                        continue
                    kept_copies.append(copy)
                subagent_conversations.append(conversation)

        for kept_copies in copies.values():
            # Copies that begin alike and then differ are both exported; each says how far they agree
            for index, one in enumerate(kept_copies):
                for other in kept_copies[index + 1:]:
                    shared = _shared_leading(one['digests'], other['digests'])
                    if not shared:
                        continue
                    for here, there in ((one, other), (other, one)):
                        here['conversation'].setdefault('shared_prefix_with', []).append(
                            {'source_file': there['conversation']['source_file'],
                             'shared_leading_messages': shared})

        # One transcript under two sessions is exported for each; each says where the other is
        same_everywhere = {}
        for (_, agent), kept_copies in copies.items():
            for copy in kept_copies:
                # As within a session, the same messages in another order are the same transcript
                same_everywhere.setdefault((agent, tuple(sorted(copy['digests']))), []).append(copy['conversation'])
        for group in same_everywhere.values():
            for conversation in group:
                others = [{'parent_session': other['parent_session'], 'source_file': other['source_file']}
                          for other in group if other is not conversation]
                if others:
                    conversation['same_transcript_in'] = others

        # A sub-agent's usage is not in its session's transcript; say that it exists
        for parent_session, agents in agents_of.items():
            if parent_session in sessions:
                sessions[parent_session]['subagent_sessions'] = sorted(set(agents))

    return conversations + (subagent_conversations if include_subagents else [])


def _output_line(conversation, encoding):
    """One conversation as a line of the output file, or None when it cannot be written

    A session is written as it always was, and a failure there still stops the export. A
    sub-agent conversation must never do that, because the sessions around it were exported
    before sub-agents were: what the file's encoding cannot hold is escaped, and what is
    nested too deeply to serialise is left out. Names are not what fails here: _name has
    already made every one of them text, in a session and in a sub-agent alike.
    """
    if not conversation.get('subagent'):
        return json.dumps(conversation, ensure_ascii=False) + '\n'
    try:
        line = json.dumps(conversation, ensure_ascii=False) + '\n'
        line.encode(encoding)
    except RecursionError:
        return None
    except UnicodeEncodeError:
        line = json.dumps(conversation) + '\n'
    return line


def main():
    print("="*80)
    print("CLAUDE CODE COMPLETE DATA EXTRACTION")
    print("="*80)
    print()

    # Find all Claude installations
    print("🔍 Searching for Claude Code installations...")
    installations = find_claude_installations()

    if not installations:
        print("❌ No Claude Code installations found!")
        return

    print(f"✅ Found {len(installations)} installation(s):")
    for inst in installations:
        _say(f"   - {inst}")
    print()

    # Extract from all installations
    all_conversations = []
    installation_stats = {}

    for installation in installations:
        _say(f"📂 Processing: {installation}")

        conversations = extract_claude_project_conversations(installation)

        if conversations:
            all_conversations.extend(conversations)
            installation_stats[str(installation)] = len(conversations)
            print(f"   ✅ {len(conversations)} conversations")
        else:
            print(f"   ⚠️  No conversations found")

    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Total conversations: {len(all_conversations):,}")

    if not all_conversations:
        print("No conversations found!")
        return

    # Statistics
    total_messages = sum(len(c['messages']) for c in all_conversations)
    with_tools = sum(1 for c in all_conversations
                     if any('tool_use' in m or 'tool_uses' in m or 'tool_results' in m
                           for m in c['messages']))
    subagents = sum(1 for c in all_conversations if c.get('subagent'))
    complete = sum(1 for c in all_conversations
                   if any(m['role'] == 'assistant' for m in c['messages']))

    print(f"Complete conversations: {complete:,}")
    print(f"Sub-agent sessions: {subagents:,}")
    print(f"Total messages: {total_messages:,}")
    print(f"With tool use/diffs: {with_tools:,}")
    print()

    print("Breakdown by installation:")
    for inst, count in sorted(installation_stats.items(), key=lambda x: -x[1]):
        _say(f"  {Path(inst).name:20} {count:5,} conversations")
    print()

    # Save to organized JSONL
    output_dir = Path('extracted_data')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'claude_code_conversations_{timestamp}.jsonl'

    not_written = 0
    with open(output_file, 'w') as f:
        for conv in all_conversations:
            line = _output_line(conv, f.encoding)
            if line is None:
                not_written += 1
                # source_file is text by now, whatever the file is called
                print(f"⚠️  Not written, nested too deeply to serialise: {conv['source_file']}")
                continue
            f.write(line)

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"✅ Saved to: {output_file}")
    if not_written:
        print(f"   Sub-agent sessions not written: {not_written:,}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Format: JSONL (one conversation per line)")

if __name__ == '__main__':
    main()
