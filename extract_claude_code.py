#!/usr/bin/env python3
"""
Extract ALL Claude Code chat data from all projects
Includes: messages, code context, diffs, file references
Sub-agent transcripts are exported as conversations of their own
A tool's structured output is kept with its result in a bounded form: 'structured' on a tool message
A file or directory name that is not valid text is written as bytes://<percent-encoded bytes>
Auto-discovers Claude Code installations on the device
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import math
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
    """Print a line; what the terminal cannot show (an emoji, a path that is not text) is escaped, not raised"""
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


STRUCTURED_STRING_LIMIT = 2000   # characters, of a string and of a key
STRUCTURED_LIST_LIMIT = 100      # items
STRUCTURED_DEPTH_LIMIT = 32      # containers inside one another
# JSON is measured as json.dumps writes it by default, with everything outside ASCII as \u
# escapes: no line of the output is written in a longer form than that
STRUCTURED_SIZE_LIMIT = 16000    # characters of JSON, after the limits above
SUBAGENT_TOTAL_LIMIT = 1000      # characters of JSON of each total in 'subagent_ref'
STOP_REASON_LIMIT = 200          # characters of JSON of 'stop_reason'
ID_LIMIT = 200                   # characters of JSON, inside the quotes, of an id written outside 'structured'
OMITTED_KEY = 'omitted-key-'


def _writable(text):
    """Whether a string can be written as UTF-8; half of a surrogate pair cannot"""
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True


def _id(value):
    """An id as an added field may hold it, UTF-8 text of at most ID_LIMIT characters of JSON, else None

    An empty id is returned as it is: it is false, like None, so it is never written either.
    """
    # The JSON is never shorter than the text, so text over the limit is refused without writing it out
    return value if (isinstance(value, str) and len(value) <= ID_LIMIT and len(json.dumps(value)) - 2 <= ID_LIMIT
                     and _writable(value)) else None


def _canonical(value):
    """One JSON form of a value as bytes: sorted keys, no spaces, ASCII"""
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _left_out(value=None):
    """What is written in place of a value that is not embedded: its kind, size and SHA-256

    The hash is of a string's UTF-8 bytes, or of the canonical JSON of a list or object, so two
    exports of one input give the same marker. A value that has no such bytes, and with no
    argument anything JSON output cannot hold, gives {'omitted': True, 'kind': 'unserialisable'}.
    """
    try:
        raw = value.encode('utf-8') if isinstance(value, str) else _canonical(value)
        kind = {str: 'string', list: 'list', dict: 'object'}[type(value)]
    except (KeyError, RecursionError, ValueError):
        return {'omitted': True, 'kind': 'unserialisable'}
    return {'omitted': True, 'kind': kind, 'length': len(value), 'sha256': hashlib.sha256(raw).hexdigest()}


def _bounded(value, depth=0, image=False):
    """A value from a transcript with every long string and long list replaced by _left_out

    A key over STRUCTURED_STRING_LIMIT characters, or one that is not UTF-8, is replaced by
    OMITTED_KEY and the SHA-256 (64 hex digits) of its UTF-8 bytes, unpaired surrogates kept.
    """
    if isinstance(value, str):
        if image or len(value) > STRUCTURED_STRING_LIMIT:
            return _left_out(value)
        return value if _writable(value) else _left_out()
    if isinstance(value, float) and not math.isfinite(value):
        return _left_out()  # NaN and Infinity are not JSON
    if isinstance(value, (list, dict)):
        if depth >= STRUCTURED_DEPTH_LIMIT:
            raise ValueError('nested too deeply')
        if isinstance(value, list):
            if len(value) > STRUCTURED_LIST_LIMIT:
                return _left_out(value)
            return [_bounded(item, depth + 1) for item in value]
        # Image data is never embedded, however short: the 'base64' of a file that was read,
        # and the 'data' of a base64 source
        bounded = {}
        for key, item in value.items():
            name = key
            if len(key) > STRUCTURED_STRING_LIMIT or not _writable(key):
                name = OMITTED_KEY + hashlib.sha256(key.encode('utf-8', 'surrogatepass')).hexdigest()
                if name in value:
                    raise ValueError('a replaced key is already a key')
            bounded[name] = _bounded(item, depth + 1,
                                     image=key == 'base64' or (key == 'data' and value.get('type') == 'base64'))
        return bounded
    return value


def _structured(value, size_limit=STRUCTURED_SIZE_LIMIT):
    """A tool's structured output (toolUseResult), or any value of an entry, in a bounded form

    Scalars and small containers are kept as they are. A string over STRUCTURED_STRING_LIMIT
    characters, a list over STRUCTURED_LIST_LIMIT items and image data of any length are
    replaced where they stand by _left_out, as is a value JSON cannot hold (NaN, Infinity,
    text that is not UTF-8). If what is left is still over `size_limit` characters of JSON,
    the whole value is replaced. A value nested over STRUCTURED_DEPTH_LIMIT containers deep, or with a
    replaced key that another key already is, is replaced whole by
    {'omitted': True, 'kind': 'unserialisable'}.
    """
    try:
        bounded = _bounded(value)
    except ValueError:
        return _left_out()
    return bounded if len(json.dumps(bounded)) <= size_limit else _left_out(value)


SUBAGENT_TOTALS = {'totalTokens': 'total_tokens', 'usage': 'usage'}


def _subagent_ref(result):
    """What a sub-agent launch result says the sub-agent used, or None when it names no agent or no total

    'agent' is the id the sub-agent's conversation is exported under ('agent' there, and in the
    session's 'subagent_sessions'), so the totals reported to the parent can be set against the
    usage of the sub-agent's own messages. They are never added to the parent's usage. An
    agentId that _id refuses gives no reference; a total over SUBAGENT_TOTAL_LIMIT is left out.
    """
    agent = _id(result.get('agentId')) if isinstance(result, dict) else None
    totals = {name: _structured(result[key], SUBAGENT_TOTAL_LIMIT)
              for key, name in SUBAGENT_TOTALS.items() if agent and key in result}
    return {'agent': agent, **totals} if totals else None


def _link_calling_entries(unlinked, messages):
    """Give each tool message whose entry did not name its caller the one exported entry that made the call

    Nothing is written when no exported assistant message has a tool_use with that id, when
    more than one has, or when the one that has carries no entry id that _id accepts.
    The messages are read only when there is such a tool message, and no more than two
    callers of one id are kept: two already say it has more than one.
    """
    callers = {}
    for message in messages if unlinked else ():
        for tool_use in message.get('tool_uses', ()):  # only an assistant message has them
            if isinstance(tool_use.get('id'), str):
                made_by = callers.setdefault(tool_use['id'], [])
                if len(made_by) < 2 and message['id'] not in made_by:
                    made_by.append(message['id'])
    for tool_msg in unlinked:
        made_by = callers.get(tool_msg['tool_call_id'], ()) if isinstance(tool_msg['tool_call_id'], str) else ()
        if len(made_by) == 1 and _id(made_by[0]):
            tool_msg['source_assistant_id'] = made_by[0]


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
    """What is needed to compare a sub-agent conversation with the others of its project, or None

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
            'owner': (conversation['parent_session'], conversation['agent']),
            'in_chain_order': _follows_parent_chain(conversation)}


def _in_same_order(part, whole):
    """Whether every message of one copy is met in the other in the order the copy has them"""
    rest = iter(whole)
    return all(digest in rest for digest in part)


def _reconcile_copies(copies):
    """Decide, for all comparable sub-agent conversations of one project at once, which are exported

    `copies` are in the order their files were read, which is a sorted order. Each is compared
    by the multiset of its message digests with every other that shares a message. Returns the
    conversations that are not exported; the fields this writes on the others are described on
    extract_claude_project_conversations. Nothing here depends on the order of `copies` except
    where a tie is said to be broken by it.
    """
    holders = {}
    for index, copy in enumerate(copies):
        copy.update(index=index, bag=Counter(copy['digests']), shared={})
        for digest in copy['bag']:
            holders.setdefault(digest, []).append(copy)
    for copy in copies:
        for digest, times in copy['bag'].items():
            for other in holders[digest]:
                if other is not copy:
                    copy['shared'][other['index']] = (copy['shared'].get(other['index'], 0)
                                                      + min(times, other['bag'][digest]))

    # Files of one session and agent with the same messages in the same order are one choice
    same_order = {}
    for copy in copies:
        same_order.setdefault((copy['owner'], tuple(copy['digests'])), []).append(copy)
    for alike in same_order.values():
        nested = any(copy['nested'] for copy in alike)
        for copy in alike:
            # More messages first; of equals, the one in the order of its parent ids, then one
            # found under <session-id>/subagents/, then the first read
            copy['rank'] = (-len(copy['digests']), not copy['in_chain_order'], not nested, copy['index'])

    for copy in copies:
        # Everything this copy has, the other has too
        copy['held_by'] = [copies[index] for index, shared in sorted(copy['shared'].items())
                           if shared == len(copy['digests'])]
        copy['left_out'] = any(other['owner'] == copy['owner'] and other['rank'] < copy['rank']
                               for other in copy['held_by'])

    for copy in copies:
        if not copy['left_out']:
            continue
        exported = [other for other in copy['held_by'] if not other['left_out']]
        # A session never loses its sub-agent: only a conversation of the same session and agent stands for it
        kept = min((other for other in exported if other['owner'] == copy['owner']),
                   key=lambda other: (-len(other['digests']), other['index']))
        conversation, left_out = kept['conversation'], copy['conversation']
        record = {'source_file': left_out['source_file'], 'shared_messages': len(copy['digests']),
                  'order_matched': _in_same_order(copy['digests'], kept['digests'])}
        also_in = [other['conversation']['source_file'] for other in exported if other is not kept]
        if also_in:
            record['also_in'] = also_in
        conversation.setdefault('duplicate_source_files', []).append(left_out['source_file'])
        conversation.setdefault('contained_source_files', []).append(record)
        # The meta file sits beside only one of the copies
        for name in META_FIELDS.values():
            if name in left_out:
                conversation.setdefault(name, left_out[name])

    for copy in copies:
        overlapping = [{'source_file': copies[index]['conversation']['source_file'], 'shared_messages': shared}
                       for index, shared in sorted(copy['shared'].items()) if not copies[index]['left_out']]
        if overlapping and not copy['left_out']:
            copy['conversation']['overlapping_source_files'] = overlapping

    return [copy['conversation'] for copy in copies if copy['left_out']]


META_FILE_LIMIT = 64 * 1024
META_FIELDS = {'agentType': 'agent_type', 'description': 'agent_description'}


def _agent_meta(agent_file):
    """What kind of sub-agent this was and what it was asked for, from the small meta file beside its transcript"""
    meta_file = agent_file.with_name(agent_file.stem + '.meta.json')
    try:
        if not meta_file.is_file() or meta_file.stat().st_size > META_FILE_LIMIT:
            return {}
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
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
    unlinked = []  # tool messages with a structured result whose entry names no calling entry (absent or null)
    session_id = _name(jsonl_file.stem)
    project_path = None
    entry_session_id = None

    with open(jsonl_file, 'r', encoding='utf-8') as f:  # transcripts are UTF-8, whatever the locale is
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
                        # The entry's structured result is written once, with its first tool result
                        result = obj.get('toolUseResult')
                        caller = obj.get('sourceToolAssistantUUID')
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
                                if _id(caller):
                                    tool_msg['source_assistant_id'] = caller
                                elif caller is None and obj.get('toolUseResult') is not None:
                                    unlinked.append(tool_msg)
                                if result is not None:
                                    tool_msg['structured'] = _structured(result)
                                    launched = _subagent_ref(result)
                                    if launched:
                                        tool_msg['subagent_ref'] = launched
                                    result = None
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
                        if message.get('stop_reason') is not None:
                            msg['stop_reason'] = _structured(message['stop_reason'], STOP_REASON_LIMIT)

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

    _link_calling_entries(unlinked, messages)

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

    One transcript can be found in several files, whole, cut short, in another order or carried
    on differently. So every message of a sub-agent conversation has a 'dedupe_key': the digest
    of what the message says, entry id included (the SHA-256 of the exported message without
    this field, as JSON with sorted keys in a list of one). Two messages with the same key are
    the same message, wherever they are; a line written twice in a file gives the same key twice.

    To count every sub-agent message once, count a 'dedupe_key' the first time it is seen and
    not again: usage summed over distinct keys is exact for any number of files, sessions and
    agents. That is the method to use.

    The conversations of a project that hold some of the same messages are found in one pass
    over all of its sub-agent files, read in sorted order, by comparing how many times each
    conversation has each key:

    - A conversation that has nothing another conversation of the same session and agent
      does not have (the same file again, a copy that stops early, the same messages in
      another order, or both) is not exported. The file is listed in 'duplicate_source_files'
      and described in 'contained_source_files' of an exported conversation that has all of
      it: 'source_file', 'shared_messages' (how many messages it has) and 'order_matched'
      (whether they come in the same order there). When several exported conversations of that
      session and agent have all of it, it is recorded on the one with the most messages, the
      first read of those with as many, and 'also_in' names the others, with any exported
      conversation of another session or agent that has all of it too.
      Of files with the same messages as many times, the one exported is the one with no
      message before its parent, else one whose order is found under <session-id>/subagents/,
      else the first read. Files beside the sessions are read first, then the others by path.
    - Conversations that share messages and are all exported each list the others in
      'overlapping_source_files': 'source_file' and 'shared_messages', the number of messages
      the two have in common, a message that both have twice counting twice.
    - A conversation of another session or of another agent is never left out for it, however
      much they share, and neither is one whose session is not known: a session keeps its
      sub-agent. They are exported and list each other in 'overlapping_source_files'.

    Two conversations a and b hold len(a) + len(b) - shared_messages different messages when
    neither repeats a message. The counts alone do not give that for three or more
    conversations that share with one another, because what all three hold is taken off once
    too often and no pair says how much that is; 'dedupe_key' does.

    A conversation with a message that has no entry id, or that cannot be digested, is not
    compared: none of the fields above is written for it, and each of its messages has a
    'dedupe_key' made of its file and position, which no other message has.

    Every name taken from a file or directory ('session_id', 'agent', 'parent_session',
    'subagent_sessions', 'project_name', 'source_file', 'duplicate_source_files',
    'installation', and 'source_file' and 'also_in' in the fields above) is written by
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
        found = []
        copies = []  # the sub-agent conversations of this project that can be compared
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
                # The same transcript can sit in several files; exported for each, its usage counts for each
                copy = _comparable_copy(conversation, nested)
                if copy is not None:
                    copies.append(copy)
                for index, message in enumerate(conversation['messages']):
                    # Not part of what is compared: the digest was taken before this was added
                    message['dedupe_key'] = (copy['digests'][index] if copy is not None
                                             else f"{conversation['source_file']}#{index}")
                found.append(conversation)

        left_out = {id(c) for c in _reconcile_copies(copies)}
        subagent_conversations.extend(c for c in found if id(c) not in left_out)

        # A sub-agent's usage is not in its session's transcript; say that it exists
        for parent_session, agents in agents_of.items():
            if parent_session in sessions:
                sessions[parent_session]['subagent_sessions'] = sorted(set(agents))

    return conversations + (subagent_conversations if include_subagents else [])


def _output_line(conversation, encoding):
    """One conversation as a line of the output file, or None when it cannot be written

    No conversation stops the export for what the file's encoding (UTF-8) cannot hold, half of
    a surrogate pair in an entry id, a timestamp or text: its whole line is then written with
    everything outside ASCII as \\u escapes, which reads back as the same values, and the lines
    around it are written as they always were. Names are not what fails here: _name has already
    made every one of them text, in a session and in a sub-agent alike.

    What is nested too deeply to serialise is left out when it is a sub-agent conversation,
    because the sessions around it were exported before sub-agents were. In a session it still
    stops the export, as it always did.
    """
    try:
        line = json.dumps(conversation, ensure_ascii=False) + '\n'
        line.encode(encoding)
    except RecursionError:
        if not conversation.get('subagent'):
            raise
        return None
    except UnicodeEncodeError:
        line = json.dumps(conversation) + '\n'
    return line


def main():
    _say("="*80)
    _say("CLAUDE CODE COMPLETE DATA EXTRACTION")
    _say("="*80)
    print()

    # Find all Claude installations
    _say("🔍 Searching for Claude Code installations...")
    installations = find_claude_installations()

    if not installations:
        _say("❌ No Claude Code installations found!")
        return

    _say(f"✅ Found {len(installations)} installation(s):")
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
            _say(f"   ✅ {len(conversations)} conversations")
        else:
            _say(f"   ⚠️  No conversations found")

    print()
    _say("="*80)
    _say("EXTRACTION COMPLETE")
    _say("="*80)
    _say(f"Total conversations: {len(all_conversations):,}")

    if not all_conversations:
        _say("No conversations found!")
        return

    # Statistics
    total_messages = sum(len(c['messages']) for c in all_conversations)
    with_tools = sum(1 for c in all_conversations
                     if any('tool_use' in m or 'tool_uses' in m or 'tool_results' in m
                           for m in c['messages']))
    subagents = sum(1 for c in all_conversations if c.get('subagent'))
    complete = sum(1 for c in all_conversations
                   if any(m['role'] == 'assistant' for m in c['messages']))

    _say(f"Complete conversations: {complete:,}")
    _say(f"Sub-agent sessions: {subagents:,}")
    _say(f"Total messages: {total_messages:,}")
    _say(f"With tool use/diffs: {with_tools:,}")
    print()

    _say("Breakdown by installation:")
    for inst, count in sorted(installation_stats.items(), key=lambda x: -x[1]):
        _say(f"  {Path(inst).name:20} {count:5,} conversations")
    print()

    # Save to organized JSONL
    output_dir = Path('extracted_data')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'claude_code_conversations_{timestamp}.jsonl'

    not_written = 0
    with open(output_file, 'w', encoding='utf-8') as f:  # JSONL is UTF-8, whatever the locale is
        for conv in all_conversations:
            line = _output_line(conv, f.encoding)
            if line is None:
                not_written += 1
                # source_file is text by now, whatever the file is called
                _say(f"⚠️  Not written, nested too deeply to serialise: {conv['source_file']}")
                continue
            f.write(line)

    file_size = output_file.stat().st_size / 1024 / 1024
    _say(f"✅ Saved to: {output_file}")
    if not_written:
        _say(f"   Sub-agent sessions not written: {not_written:,}")
    _say(f"   Size: {file_size:.2f} MB")
    _say(f"   Format: JSONL (one conversation per line)")

if __name__ == '__main__':
    main()
