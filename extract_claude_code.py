#!/usr/bin/env python3
"""
Extract ALL Claude Code chat data from all projects
Includes: messages, code context, diffs, file references
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


USAGE_KEYS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def _identity(obj):
    """Source entry identity, so an exported message can be traced back to the transcript"""
    return {'id': obj.get('uuid'), 'parent_id': obj.get('parentUuid')}


def _block_text(content):
    """Text of a tool result, which is either a string or a list of text blocks"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(item if isinstance(item, str) else item.get('text', '') for item in content
                         if isinstance(item, str) or (isinstance(item, dict) and item.get('type') == 'text'))
    return ''

def extract_claude_project_conversations(project_dir):
    """Extract conversations from a Claude project directory with full context"""
    conversations = []

    # Find all JSONL session files
    jsonl_files = []
    if (project_dir / 'projects').exists():
        # New structure: projects/project-name/session.jsonl
        for proj in (project_dir / 'projects').iterdir():
            if proj.is_dir():
                jsonl_files.extend(list(proj.glob('*.jsonl')))
    else:
        # Old structure: direct JSONL files
        jsonl_files = list(project_dir.glob('*.jsonl'))

    # Filter out agent files
    jsonl_files = [f for f in jsonl_files if not f.name.startswith('agent-')]

    for jsonl_file in jsonl_files:
        try:
            messages = []
            omitted_attachments = {}
            session_id = jsonl_file.stem
            project_path = None
            project_name = jsonl_file.parent.name if jsonl_file.parent.name != 'projects' else None

            with open(jsonl_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        obj = json.loads(line)
                        msg_type = obj.get('type')

                        if msg_type == 'user':
                            message = obj.get('message', {})
                            content = message.get('content', '')

                            # Tool output is stored as a user entry; it is not something the user said
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'tool_result':
                                        messages.append({
                                            'role': 'tool',
                                            'tool_call_id': item.get('tool_use_id'),
                                            'content': _block_text(item.get('content')),
                                            'is_error': bool(item.get('is_error')),
                                            'timestamp': obj.get('timestamp'),
                                            **_identity(obj)
                                        })
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
                            if full_text or tool_uses:
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
                                    msg['usage'] = {k: usage[k] for k in USAGE_KEYS if k in usage}

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

                    except json.JSONDecodeError:
                        continue

            if messages:
                conversations.append({
                    'messages': messages,
                    'source': 'claude-code',
                    'session_id': session_id,
                    'project_path': project_path,
                    'project_name': project_name,
                    'omitted_attachments': omitted_attachments,
                    'source_file': str(jsonl_file),
                    'installation': str(project_dir)
                })

        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue

    return conversations

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
    complete = sum(1 for c in all_conversations
                   if any(m['role'] == 'assistant' for m in c['messages']))

    print(f"Complete conversations: {complete:,}")
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
