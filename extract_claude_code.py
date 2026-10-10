#!/usr/bin/env python3
"""
Extract ALL Claude Code chat data from all projects
Includes: messages, code context, diffs, file references
Sub-agent transcripts are exported as conversations of their own
Auto-discovers Claude Code installations on the device
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import os

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
    sessions = sorted(f for f in container.glob('*.jsonl') if not f.name.startswith('agent-'))
    subagents = [(f, None) for f in sorted(container.glob('agent-*.jsonl'))]
    subagents += [(f, f.relative_to(container).parts[0])
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
    session_id = jsonl_file.stem
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
        'source_file': str(jsonl_file),
        'installation': str(project_dir)
    }, entry_session_id


def extract_claude_project_conversations(project_dir, include_subagents=True):
    """Extract conversations from a Claude project directory with full context

    A sub-agent's transcript is exported as a conversation of its own, marked 'subagent' and
    linked to its session by 'parent_session'; the session lists its sub-agents in
    'subagent_sessions' whether or not they are exported.
    """
    project_dir = Path(project_dir)
    conversations = []
    subagent_conversations = []

    if (project_dir / 'projects').exists():
        # New structure: projects/project-name/session.jsonl
        containers = sorted(proj for proj in (project_dir / 'projects').iterdir() if proj.is_dir())
    else:
        # Old structure: direct JSONL files
        containers = [project_dir]

    def convert(jsonl_file, project_name):
        try:
            return _convert_session_file(jsonl_file, project_dir, project_name)
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            return None, None

    for container in containers:
        project_name = container.name if container.name != 'projects' else None
        session_files, subagent_files = _session_files(container)

        sessions = {}
        for jsonl_file in session_files:
            conversation, _ = convert(jsonl_file, project_name)
            if conversation:
                conversations.append(conversation)
                sessions[conversation['session_id']] = conversation

        agents_of = {}
        exported = {}
        for agent_file, parent_session in subagent_files:
            agent = agent_file.stem[len('agent-'):]
            conversation, entry_session_id = convert(agent_file, project_name)
            # Only a string can name a session; anything else in that field is noise
            if not parent_session and isinstance(entry_session_id, str):
                parent_session = entry_session_id.strip() or None
            if parent_session:
                agents_of.setdefault(parent_session, []).append(agent)
            if conversation:
                # The same transcript can sit in both layouts; exported twice, its usage counts twice
                entry_ids = tuple(message.get('id') for message in conversation['messages'])
                # Without real entry ids two transcripts cannot be told apart, so both are kept
                identified = bool(entry_ids) and all(isinstance(i, str) and i for i in entry_ids)
                # Same ids are not enough: a copy can differ in what the entries say
                key = (parent_session, agent, _messages_digest(conversation['messages'])) if identified else None
                first = exported.get(key) if key else None
                if first is not None:
                    first.setdefault('duplicate_source_files', []).append(conversation['source_file'])
                    # The meta file sits beside only one of the copies
                    for name, value in _agent_meta(agent_file).items():
                        first.setdefault(name, value)
                    continue
                if key:
                    exported[key] = conversation
                conversation.update(subagent=True, agent=agent, parent_session=parent_session,
                                    **_agent_meta(agent_file))
                subagent_conversations.append(conversation)

        # A sub-agent's usage is not in its session's transcript; say that it exists
        for parent_session, agents in agents_of.items():
            if parent_session in sessions:
                sessions[parent_session]['subagent_sessions'] = sorted(set(agents))

    return conversations + (subagent_conversations if include_subagents else [])


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
        print(f"   - {inst}")
    print()

    # Extract from all installations
    all_conversations = []
    installation_stats = {}

    for installation in installations:
        print(f"📂 Processing: {installation}")

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
        print(f"  {Path(inst).name:20} {count:5,} conversations")
    print()

    # Save to organized JSONL
    output_dir = Path('extracted_data')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'claude_code_conversations_{timestamp}.jsonl'

    with open(output_file, 'w') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"✅ Saved to: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Format: JSONL (one conversation per line)")

if __name__ == '__main__':
    main()
