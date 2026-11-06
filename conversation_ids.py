"""Shared stable identifiers for existing conversation extractor formats."""

SESSION_IDS = ('session_id', 'composer_id', 'conversation_id', 'chat_id', 'tab_id')


def _valid_identifier(value):
    return (isinstance(value, (str, int)) and not isinstance(value, bool)
            and value != '')


def conversation_identifiers(record):
    """Return typed native identifiers, or None when identity is incomplete.

    Native session IDs take precedence. Cursor CLI's chat ID names a group of
    agents, so its fallback identity requires both chat_id and agent_id.
    """
    for field in SESSION_IDS:
        identifier = record.get(field)
        if not _valid_identifier(identifier):
            continue
        identifiers = {field: identifier}
        if field == 'chat_id' and record.get('source') == 'cursor-agent-cli':
            agent_id = record.get('agent_id')
            if not _valid_identifier(agent_id):
                return None
            identifiers['agent_id'] = agent_id
        return identifiers
    return None
