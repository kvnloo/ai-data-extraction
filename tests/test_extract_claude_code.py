import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extract_claude_code
from extract_claude_code import extract_claude_project_conversations, find_claude_installations

TS = "2026-09-01T00:00:00.000Z"

SESSION = [
    {"type": "user", "uuid": "u1", "timestamp": TS, "cwd": "/work",
     "message": {"role": "user", "content": "How do we deploy?"}},
    {"type": "attachment", "uuid": "a1", "timestamp": TS,
     "attachment": {"type": "hook_additional_context", "hookEvent": "UserPromptSubmit", "hookName": "UserPromptSubmit",
                    "content": ["workstream memory: deploy with canary"]}},
    {"type": "attachment", "uuid": "a2", "timestamp": TS, "attachment": {"type": "date", "date": "2026-09-01"}},
    {"type": "assistant", "uuid": "s1", "timestamp": TS,
     "message": {"role": "assistant", "model": "claude-x", "content": [
         {"type": "thinking", "thinking": "Check the config."},
         {"type": "text", "text": "Checking."},
         {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "cat deploy.yml"}},
     ]}},
    {"type": "user", "uuid": "u2", "timestamp": TS,
     "message": {"role": "user", "content": [
         {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": False,
          "content": [{"type": "text", "text": "strategy: canary"}]},
     ]}},
    {"type": "assistant", "uuid": "s2", "timestamp": TS,
     "message": {"role": "assistant", "model": "claude-x", "content": [{"type": "text", "text": "Canary."}]}},
]


class ExtractClaudeCodeTests(unittest.TestCase):
    def install(self, root, rows=SESSION):
        project = Path(root) / "projects" / "-work"
        project.mkdir(parents=True)
        (project / "sess-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return Path(root)

    def conversation(self, rows=SESSION):
        with tempfile.TemporaryDirectory() as tmp:
            return extract_claude_project_conversations(self.install(tmp, rows))[0]

    def test_tool_output_is_a_tool_message_not_a_user_turn(self):
        messages = self.conversation()["messages"]
        self.assertEqual([m["role"] for m in messages if m["role"] in ("user", "tool")], ["user", "tool"])
        tool = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(tool["tool_call_id"], "toolu_1")
        self.assertEqual(tool["content"], "strategy: canary")
        self.assertFalse(tool["is_error"])
        # The only user turn is what the user typed.
        self.assertEqual([m["content"] for m in messages if m["role"] == "user"], ["How do we deploy?"])

    def test_a_failed_tool_call_keeps_its_error_flag(self):
        rows = [dict(r) for r in SESSION]
        rows[4] = {**rows[4], "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": "No such file"}]}}
        tool = next(m for m in self.conversation(rows)["messages"] if m["role"] == "tool")
        self.assertTrue(tool["is_error"])
        self.assertEqual(tool["content"], "No such file")

    def test_context_a_hook_gave_the_model_is_kept_and_marked(self):
        messages = self.conversation()["messages"]
        injected = [m for m in messages if m.get("custom")]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["role"], "system")
        self.assertEqual(injected[0]["custom_type"], "hook_additional_context")
        self.assertEqual(injected[0]["hook_event"], "UserPromptSubmit")
        self.assertIn("deploy with canary", injected[0]["content"])
        # It sits where the model saw it: after the prompt, before the answer.
        self.assertEqual([m["role"] for m in messages][:3], ["user", "system", "assistant"])
        # Other attachments (date, environment, ...) are not conversation.
        self.assertEqual({m.get("custom_type") for m in messages if m.get("custom")}, {"hook_additional_context"})

    def test_assistant_reasoning_is_kept_beside_the_answer(self):
        first = next(m for m in self.conversation()["messages"] if m["role"] == "assistant")
        self.assertEqual(first["reasoning"], "Check the config.")
        self.assertEqual(first["content"], "Checking.")
        self.assertEqual(first["tool_uses"][0]["name"], "Bash")

    def test_home_installation_is_found_without_xdg_directories(self):
        with tempfile.TemporaryDirectory() as home:
            self.install(Path(home) / ".claude")
            with mock.patch.object(extract_claude_code.Path, "home", return_value=Path(home)), \
                    mock.patch.dict(os.environ, {}, clear=True):
                found = find_claude_installations()
            self.assertEqual([p.name for p in found], [".claude"])

    def test_configured_directory_is_found(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as elsewhere:
            custom = self.install(Path(elsewhere) / "claude-home")
            with mock.patch.object(extract_claude_code.Path, "home", return_value=Path(home)), \
                    mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(custom)}, clear=True):
                found = find_claude_installations()
            self.assertEqual(found, [custom])


if __name__ == "__main__":
    unittest.main()


USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 7}

IDENTIFIED = [
    {"type": "user", "uuid": "u1", "parentUuid": None, "timestamp": TS, "cwd": "/work",
     "message": {"role": "user", "content": "go"}},
    {"type": "attachment", "uuid": "a1", "parentUuid": "u1", "timestamp": TS,
     "attachment": {"type": "hook_additional_context", "hookEvent": "UserPromptSubmit", "content": ["memory"]}},
    {"type": "attachment", "uuid": "a2", "parentUuid": "a1", "timestamp": TS, "attachment": {"type": "date"}},
    {"type": "attachment", "uuid": "a3", "parentUuid": "a2", "timestamp": TS, "attachment": {"type": "date"}},
    # One model response written as two entries: both carry the same message id and usage.
    {"type": "assistant", "uuid": "s1", "parentUuid": "a3", "timestamp": TS, "requestId": "req_1",
     "message": {"id": "msg_1", "role": "assistant", "model": "claude-x", "usage": USAGE,
                 "content": [{"type": "text", "text": "Looking."}]}},
    {"type": "assistant", "uuid": "s2", "parentUuid": "s1", "timestamp": TS, "requestId": "req_1",
     "message": {"id": "msg_1", "role": "assistant", "model": "claude-x", "usage": USAGE,
                 "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]}},
    {"type": "user", "uuid": "u2", "parentUuid": "s2", "timestamp": TS,
     "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.py"}]}},
]


class ExtractClaudeCodeProvenanceTests(ExtractClaudeCodeTests):
    def test_every_message_names_its_source_entry_and_parent(self):
        messages = self.conversation(IDENTIFIED)["messages"]
        self.assertEqual([m["id"] for m in messages], ["u1", "a1", "s1", "s2", "u2"])
        self.assertEqual([m["parent_id"] for m in messages], [None, "u1", "a3", "s1", "s2"])

    def test_usage_is_exported_with_the_ids_needed_to_count_it_once(self):
        assistant = [m for m in self.conversation(IDENTIFIED)["messages"] if m["role"] == "assistant"]
        self.assertEqual([m["usage"] for m in assistant], [USAGE, USAGE])
        # Two entries, one response: a consumer must dedupe on message_id before summing.
        self.assertEqual({m["message_id"] for m in assistant}, {"msg_1"})
        self.assertEqual({m["request_id"] for m in assistant}, {"req_1"})

    def test_attachments_left_out_are_counted_by_type(self):
        conversation = self.conversation(IDENTIFIED)
        self.assertEqual(conversation["omitted_attachments"], {"date": 2})
        self.assertNotIn("hook_additional_context", conversation["omitted_attachments"])

    def test_entries_without_identity_export_as_before(self):
        messages = self.conversation()["messages"]  # SESSION has uuids but no parents, usage or message ids
        assistant = next(m for m in messages if m["role"] == "assistant")
        self.assertNotIn("usage", assistant)
        self.assertNotIn("message_id", assistant)
        self.assertIsNone(assistant["parent_id"])


FULL_USAGE = {**USAGE, "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 2},
              "server_tool_use": {"web_search_requests": 2}, "service_tier": "standard"}
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD" * 10}}


def user(uuid, parent, content, **extra):
    return {"type": "user", "uuid": uuid, "parentUuid": parent, "timestamp": TS, "cwd": "/work",
            "message": {"role": "user", "content": content}, **extra}


def assistant(uuid, parent, content, **extra):
    return {"type": "assistant", "uuid": uuid, "parentUuid": parent, "timestamp": TS, "requestId": "req_1",
            "message": {"id": "msg_1", "role": "assistant", "model": "claude-x", "usage": FULL_USAGE,
                        "content": content}, **extra}


class ExtractClaudeCodeProvenanceAuditTests(ExtractClaudeCodeTests):
    def test_a_response_entry_that_is_only_reasoning_is_still_exported(self):
        # Each content block of a response is its own entry; the reasoning one has no text and no tool call.
        messages = self.conversation([
            user("u1", None, "go"),
            assistant("s1", "u1", [{"type": "thinking", "thinking": "Plan it."}]),
            assistant("s2", "s1", [{"type": "text", "text": "Done."}]),
        ])["messages"]
        self.assertEqual([m["id"] for m in messages], ["u1", "s1", "s2"])
        self.assertEqual(messages[1]["reasoning"], "Plan it.")
        self.assertEqual(messages[1]["usage"], FULL_USAGE)

    def test_usage_keeps_every_field_the_transcript_recorded(self):
        message = self.conversation([user("u1", None, "go"),
                                     assistant("s1", "u1", [{"type": "text", "text": "Done."}])])["messages"][1]
        self.assertEqual(message["usage"], FULL_USAGE)

    def test_parent_ids_can_be_followed_through_entries_left_out(self):
        conversation = self.conversation([
            user("u1", None, "go"),
            {"type": "attachment", "uuid": "a1", "parentUuid": "u1", "timestamp": TS, "attachment": {"type": "date"}},
            assistant("s1", "a1", [{"type": "text", "text": "Done."}]),
            {"type": "system", "uuid": "b1", "parentUuid": None, "logicalParentUuid": "s1", "timestamp": TS,
             "subtype": "compact_boundary", "content": "Conversation compacted"},
            user("u2", "b1", "next"),
        ])
        self.assertEqual(conversation["omitted_entries"], {
            "a1": {"type": "attachment", "parent_id": "u1"},
            "b1": {"type": "system", "parent_id": None, "logical_parent_id": "s1"},
        })
        # Every parent an exported message names is either exported or accounted for.
        known = {m["id"] for m in conversation["messages"]} | set(conversation["omitted_entries"]) | {None}
        self.assertLessEqual({m["parent_id"] for m in conversation["messages"]}, known)
        self.assertEqual(conversation["omitted_attachments"], {"date": 1})

    def test_sidechain_meta_and_compact_summary_entries_are_marked(self):
        messages = self.conversation([
            user("u1", None, "go"),
            user("c1", "u1", "This session is being continued...", isCompactSummary=True),
            user("m1", "c1", "<local-command-caveat>generated</local-command-caveat>", isMeta=True),
            user("k1", None, "sub-agent prompt", isSidechain=True),
            assistant("k2", "k1", [{"type": "text", "text": "sub-agent answer"}], isSidechain=True),
        ])["messages"]
        flags = ("is_compact_summary", "is_meta", "is_sidechain")
        self.assertEqual([[f for f in flags if m.get(f)] for m in messages],
                         [[], ["is_compact_summary"], ["is_meta"], ["is_sidechain"], ["is_sidechain"]])
        # An ordinary entry gains no keys.
        self.assertFalse(set(flags) & set(messages[0]))

    def test_tool_output_that_is_not_text_is_described_without_its_bytes(self):
        conversation = self.conversation([
            user("u1", None, "go"),
            user("u2", "u1", [{"type": "tool_result", "tool_use_id": "toolu_1",
                               "content": [{"type": "text", "text": "shot"}, IMAGE]}]),
        ])
        tool = conversation["messages"][1]
        self.assertEqual(tool["content"], "shot")
        self.assertEqual(tool["omitted_content"], [{"type": "image", "media_type": "image/png", "encoded_size": 40}])
        self.assertNotIn(IMAGE["source"]["data"], json.dumps(tool))
        # Text-only tool output exports as before.
        self.assertNotIn("omitted_content", next(m for m in self.conversation()["messages"] if m["role"] == "tool"))

    def test_one_unreadable_line_costs_that_line_not_the_session(self):
        rows = [user("u1", None, "go"), assistant("s1", "u1", [{"type": "text", "text": "Done."}])]
        for bad in ('{"type": "assistant", "uuid": "x", "mess', "null", "[1, 2]",
                    json.dumps({"type": "assistant", "uuid": "x", "message": None}),
                    json.dumps({"type": "attachment", "uuid": "x", "attachment": "oops"})):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                root = self.install(tmp, rows)
                session = root / "projects" / "-work" / "sess-1.jsonl"
                first, second = session.read_text().splitlines()
                session.write_text("\n".join([first, bad, second]) + "\n")
                conversations = extract_claude_project_conversations(root)
                self.assertEqual([[m["id"] for m in c["messages"]] for c in conversations], [["u1", "s1"]])
                self.assertEqual(conversations[0]["skipped_lines"], 1)
        self.assertEqual(self.conversation(rows)["skipped_lines"], 0)


SUB_USAGE = {"input_tokens": 100, "output_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 9}
PARENT_USAGE = {"input_tokens": 20, "output_tokens": 6, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 2}


def transcript(prefix, usage, session_id="sess-1", agent_id=None, replies=1):
    """A synthetic transcript: one prompt, then `replies` responses that each carry `usage` under their own id"""
    extra = {"sessionId": session_id}
    if agent_id:
        extra.update(agentId=agent_id, isSidechain=True)
    rows = [{"type": "user", "uuid": f"{prefix}-u", "parentUuid": None, "timestamp": TS, "cwd": "/work",
             "message": {"role": "user", "content": f"{prefix} prompt"}, **extra},
            {"type": "attachment", "uuid": f"{prefix}-a", "parentUuid": f"{prefix}-u", "timestamp": TS,
             "attachment": {"type": "date"}, **extra}]
    parent = f"{prefix}-a"
    for n in range(replies):
        rows.append({"type": "assistant", "uuid": f"{prefix}-s{n}", "parentUuid": parent, "timestamp": TS,
                     "requestId": f"req-{prefix}-{n}",
                     "message": {"id": f"msg-{prefix}-{n}", "role": "assistant", "model": "claude-x", "usage": usage,
                                 "content": [{"type": "text", "text": f"{prefix} reply {n}"}]}, **extra})
        parent = f"{prefix}-s{n}"
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def total_usage(conversations):
    """Sum usage across conversations, counting each response once"""
    seen, totals = set(), {}
    for conversation in conversations:
        for message in conversation["messages"]:
            if "usage" in message and message["message_id"] not in seen:
                seen.add(message["message_id"])
                for key, value in message["usage"].items():
                    totals[key] = totals.get(key, 0) + value
    return totals


class ExtractClaudeCodeSubagentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.project = self.root / "projects" / "-work"
        self.project.mkdir(parents=True)

    def parent(self, session_id="sess-1"):
        return write_rows(self.project / f"{session_id}.jsonl", transcript("main", PARENT_USAGE, session_id))

    def nested(self, agent_id, session_id="sess-1", rows=None, meta=None):
        path = self.project / session_id / "subagents" / f"agent-{agent_id}.jsonl"
        write_rows(path, transcript(agent_id, SUB_USAGE, session_id, agent_id) if rows is None else rows)
        if meta is not None:
            path.with_name(f"agent-{agent_id}.meta.json").write_text(meta if isinstance(meta, str) else json.dumps(meta))
        return path

    def beside(self, agent_id, session_id="sess-1"):
        return write_rows(self.project / f"agent-{agent_id}.jsonl", transcript(agent_id, SUB_USAGE, session_id, agent_id))

    def export(self, **kwargs):
        conversations = extract_claude_project_conversations(self.root, **kwargs)
        return ([c for c in conversations if not c.get("subagent")],
                sorted((c for c in conversations if c.get("subagent")), key=lambda c: c["agent"]))

    def test_a_session_without_sub_agents_exports_as_before(self):
        self.parent()
        mains, subs = self.export()
        self.assertEqual(subs, [])
        self.assertNotIn("subagent", mains[0])
        self.assertNotIn("parent_session", mains[0])
        self.assertNotIn("subagent_sessions", mains[0])

    def test_sub_agent_in_the_session_directory_is_its_own_conversation(self):
        self.parent()
        path = self.nested("a1b2", meta={"agentType": "Explore", "description": "Find the deploy config",
                                         "prompt": "do-not-export-this", "worktreePath": "/secret/place"})
        mains, subs = self.export()
        self.assertEqual(len(mains), 1)
        self.assertEqual(len(subs), 1)
        sub = subs[0]
        self.assertIs(sub["subagent"], True)
        self.assertEqual(sub["agent"], "a1b2")
        self.assertEqual(sub["parent_session"], "sess-1")
        self.assertEqual(sub["session_id"], "agent-a1b2")
        self.assertEqual(sub["source"], "claude-code")
        self.assertEqual(sub["source_file"], str(path))
        self.assertEqual(sub["project_name"], "-work")
        self.assertEqual(sub["project_path"], "/work")
        self.assertEqual(sub["agent_type"], "Explore")
        self.assertEqual(sub["agent_description"], "Find the deploy config")
        # Nothing else in the meta file is exported.
        self.assertNotIn("do-not-export-this", json.dumps(sub))
        self.assertNotIn("/secret/place", json.dumps(sub))
        # The parent says sub-agent usage exists, for a consumer that ignores those conversations.
        self.assertEqual(mains[0]["subagent_sessions"], ["a1b2"])

    def test_sub_agent_goes_through_the_same_conversion_as_a_main_session(self):
        self.parent()
        self.nested("a1b2")
        mains, subs = self.export()
        sub = subs[0]
        self.assertEqual(set(sub) - {"subagent", "agent", "parent_session"}, set(mains[0]) - {"subagent_sessions"})
        self.assertEqual([m["id"] for m in sub["messages"]], ["a1b2-u", "a1b2-s0"])
        self.assertEqual([m["parent_id"] for m in sub["messages"]], [None, "a1b2-a"])
        self.assertEqual(sub["messages"][1]["usage"], SUB_USAGE)
        self.assertEqual(sub["messages"][1]["message_id"], "msg-a1b2-0")
        self.assertEqual(sub["messages"][1]["request_id"], "req-a1b2-0")
        self.assertTrue(all(m["is_sidechain"] for m in sub["messages"]))
        self.assertEqual(sub["omitted_entries"], {"a1b2-a": {"type": "attachment", "parent_id": "a1b2-u"}})
        self.assertEqual(sub["omitted_attachments"], {"date": 1})
        self.assertEqual(sub["skipped_lines"], 0)

    def test_sub_agent_beside_the_session_file_takes_its_parent_from_its_entries(self):
        self.parent()
        path = self.beside("c3d4")
        mains, subs = self.export()
        self.assertEqual(len(mains), 1)  # the agent file is not mistaken for a session of its own
        self.assertEqual([(s["agent"], s["parent_session"], s["subagent"]) for s in subs], [("c3d4", "sess-1", True)])
        self.assertEqual(subs[0]["source_file"], str(path))
        self.assertEqual(subs[0]["project_name"], "-work")
        self.assertEqual(subs[0]["messages"][1]["usage"], SUB_USAGE)
        self.assertEqual(mains[0]["subagent_sessions"], ["c3d4"])

    def test_both_layouts_together_and_each_parent_lists_only_its_own(self):
        self.parent("sess-1")
        self.parent("sess-2")
        self.nested("n1", "sess-1")
        self.nested("n2", "sess-2")
        self.beside("o1", "sess-1")
        mains, subs = self.export()
        self.assertEqual([(s["agent"], s["parent_session"]) for s in subs],
                         [("n1", "sess-1"), ("n2", "sess-2"), ("o1", "sess-1")])
        self.assertEqual({m["session_id"]: m["subagent_sessions"] for m in mains},
                         {"sess-1": ["n1", "o1"], "sess-2": ["n2"]})

    def test_sub_agent_files_deeper_under_subagents_are_found(self):
        self.parent()
        deep = self.project / "sess-1" / "subagents" / "group" / "agent-deep.jsonl"
        write_rows(deep, transcript("deep", SUB_USAGE, "sess-1", "deep"))
        mains, subs = self.export()
        self.assertEqual([(s["agent"], s["parent_session"]) for s in subs], [("deep", "sess-1")])
        self.assertEqual(mains[0]["subagent_sessions"], ["deep"])

    def test_sub_agent_without_its_parent_session_file_is_still_exported(self):
        self.nested("n1", "gone-1")
        self.beside("o1", "gone-2")
        mains, subs = self.export()
        self.assertEqual(mains, [])
        self.assertEqual([(s["agent"], s["parent_session"], s["subagent"]) for s in subs],
                         [("n1", "gone-1", True), ("o1", "gone-2", True)])
        self.assertEqual(total_usage(subs), {k: 2 * v for k, v in SUB_USAGE.items()})

    def test_usage_of_parent_and_sub_agents_adds_up_to_what_the_fixtures_hold(self):
        self.parent()
        self.nested("n1", rows=transcript("n1", SUB_USAGE, "sess-1", "n1", replies=3))
        self.nested("n2")
        self.beside("o1")
        mains, subs = self.export()
        self.assertEqual(total_usage(mains), PARENT_USAGE)
        self.assertEqual(total_usage(subs), {k: 5 * v for k, v in SUB_USAGE.items()})
        expected = {k: PARENT_USAGE[k] + 5 * SUB_USAGE[k] for k in PARENT_USAGE}
        self.assertEqual(total_usage(mains + subs), expected)
        self.assertEqual(expected["input_tokens"], 520)

    def test_a_malformed_sub_agent_file_does_not_break_the_parent(self):
        self.parent()
        self.nested("good")
        self.nested("junk", rows=[]).write_text('{"type": "assistant", "mess\nnull\n[1, 2]\nnot json at all\n')
        self.nested("bytes", rows=[]).write_bytes(b"\xff\xfe\x00\x80 not utf-8 \xc3\x28\n")
        self.nested("empty", rows=[])
        (self.project / "agent-oldjunk.jsonl").write_text("{{{{\n")
        half = self.nested("half")
        half.write_text(half.read_text() + '{"type": "assistant", "uuid": "x", "message": None\n')
        mains, subs = self.export()
        self.assertEqual([[m["id"] for m in c["messages"]] for c in mains], [["main-u", "main-s0"]])
        self.assertEqual(total_usage(mains), PARENT_USAGE)
        # Files with nothing readable yield no conversation; a partly readable one keeps what it can.
        self.assertEqual([s["agent"] for s in subs], ["good", "half"])
        self.assertEqual(subs[1]["skipped_lines"], 1)
        self.assertEqual([m["id"] for m in subs[1]["messages"]], ["half-u", "half-s0"])
        # The parent still names every sub-agent file its directory holds, readable or not.
        self.assertEqual(mains[0]["subagent_sessions"], ["bytes", "empty", "good", "half", "junk"])

    def test_meta_file_is_only_read_when_it_is_small_json(self):
        self.parent()
        self.nested("big", meta=json.dumps({"agentType": "Explore", "description": "x" * 200_000}))
        self.nested("bad", meta="{not json")
        self.nested("list", meta=["agentType", "Explore"])
        self.nested("odd", meta={"agentType": {"nested": "thing"}, "description": "Only a string is exported"})
        self.nested("none")
        _, subs = self.export()
        by_agent = {s["agent"]: s for s in subs}
        self.assertEqual(sorted(by_agent), ["bad", "big", "list", "none", "odd"])
        for name in ("big", "bad", "list", "none"):
            self.assertNotIn("agent_type", by_agent[name])
            self.assertNotIn("agent_description", by_agent[name])
        self.assertNotIn("agent_type", by_agent["odd"])
        self.assertEqual(by_agent["odd"]["agent_description"], "Only a string is exported")
        # A meta file is never a session.
        self.assertFalse([s for s in subs if "meta" in s["session_id"]])

    def test_sub_agents_can_be_left_out_but_the_parent_still_says_they_exist(self):
        self.parent()
        self.nested("n1")
        self.beside("o1")
        mains, subs = self.export(include_subagents=False)
        self.assertEqual(subs, [])
        self.assertEqual(mains[0]["subagent_sessions"], ["n1", "o1"])

    def test_sub_agents_are_found_when_given_a_directory_of_session_files(self):
        # The older calling convention: a directory that holds session files directly, with no projects/ level.
        self.parent()
        self.nested("n1")
        self.beside("o1")
        conversations = extract_claude_project_conversations(self.project)
        self.assertEqual(sorted((c.get("agent"), c.get("parent_session")) for c in conversations if c.get("subagent")),
                         [("n1", "sess-1"), ("o1", "sess-1")])
        main = next(c for c in conversations if not c.get("subagent"))
        self.assertEqual(main["subagent_sessions"], ["n1", "o1"])

    def test_a_sub_agent_file_with_an_unusable_session_id_does_not_stop_the_export(self):
        self.parent()
        for agent_id, bad in (("list", ["sess-1"]), ("dict", {"id": "sess-1"}), ("int", 7)):
            rows = transcript(agent_id, SUB_USAGE, "sess-1", agent_id)
            for row in rows:
                row["sessionId"] = bad
            write_rows(self.project / f"agent-{agent_id}.jsonl", rows)

        main, subs = self.export()

        self.assertEqual(len(main), 1)
        # The transcripts are still exported; they just cannot be tied to a session
        self.assertEqual([c["agent"] for c in subs], ["dict", "int", "list"])
        self.assertEqual({c["parent_session"] for c in subs}, {None})
        self.assertNotIn("subagent_sessions", main[0])

    def test_a_corrupt_meta_file_costs_only_the_meta_fields(self):
        self.parent()
        self.nested("deep", meta="[" * 60000)

        main, subs = self.export()

        self.assertEqual([c["agent"] for c in subs], ["deep"])
        self.assertNotIn("agent_type", subs[0])
        self.assertEqual(main[0]["subagent_sessions"], ["deep"])

    def test_one_agent_found_in_both_layouts_is_listed_once_on_its_session(self):
        self.parent()
        self.nested("dup")
        self.beside("dup")

        main, _ = self.export()

        self.assertEqual(main[0]["subagent_sessions"], ["dup"])

    def test_the_same_transcript_in_both_layouts_is_exported_once(self):
        self.parent()
        self.nested("dup")
        self.beside("dup")

        main, subs = self.export()

        self.assertEqual([c["agent"] for c in subs], ["dup"])
        self.assertEqual(len(subs[0]["duplicate_source_files"]), 1)
        # Summed the naive way, one conversation at a time, usage is counted once
        naive = sum(m["usage"]["output_tokens"] for c in main + subs for m in c["messages"] if m.get("usage"))
        expected = sum(m["usage"]["output_tokens"] for c in main + subs[:1] for m in c["messages"] if m.get("usage"))
        self.assertEqual(naive, expected)

    def test_two_different_transcripts_with_one_agent_id_are_both_kept(self):
        self.parent()
        self.nested("same")
        rows = transcript("same", SUB_USAGE, "sess-1", "same")
        for row in rows:
            row["uuid"] = "other-" + row["uuid"]
            if row.get("parentUuid"):
                row["parentUuid"] = "other-" + row["parentUuid"]
        write_rows(self.project / "agent-same.jsonl", rows)

        _, subs = self.export()

        self.assertEqual(len(subs), 2)
        self.assertTrue(all("duplicate_source_files" not in c for c in subs))

    def test_a_padded_session_id_still_links_to_its_session(self):
        self.parent()
        rows = transcript("pad", SUB_USAGE, "sess-1", "pad")
        for row in rows:
            row["sessionId"] = " sess-1 "
        write_rows(self.project / "agent-pad.jsonl", rows)
        blank = transcript("blank", SUB_USAGE, "sess-1", "blank")
        for row in blank:
            row["sessionId"] = " \t"
        write_rows(self.project / "agent-blank.jsonl", blank)

        main, subs = self.export()

        by_agent = {c["agent"]: c["parent_session"] for c in subs}
        self.assertEqual(by_agent, {"pad": "sess-1", "blank": None})
        self.assertEqual(main[0]["subagent_sessions"], ["pad"])

    def test_transcripts_without_entry_ids_are_never_merged(self):
        self.parent()
        for where in (self.project / "sess-1" / "subagents" / "agent-anon.jsonl", self.project / "agent-anon.jsonl"):
            rows = transcript("anon", SUB_USAGE, "sess-1", "anon")
            for row in rows:
                row.pop("uuid", None)
                row.pop("parentUuid", None)
            write_rows(where, rows)

        _, subs = self.export()

        self.assertEqual(len(subs), 2)

    def test_an_entry_id_that_is_not_a_string_does_not_stop_the_export(self):
        self.parent()
        rows = transcript("odd", SUB_USAGE, "sess-1", "odd")
        rows[0]["uuid"] = ["not", "a", "string"]
        self.nested("odd", rows=rows)

        main, subs = self.export()

        self.assertEqual(len(main), 1)
        self.assertEqual([c["agent"] for c in subs], ["odd"])

    def test_copies_with_the_same_entry_ids_but_different_content_are_both_kept(self):
        self.parent()
        self.nested("fork")
        rows = transcript("fork", SUB_USAGE, "sess-1", "fork")
        for row in rows:
            usage = (row.get("message") or {}).get("usage")
            if usage:
                usage["output_tokens"] = 999
        write_rows(self.project / "agent-fork.jsonl", rows)

        _, subs = self.export()

        self.assertEqual(len(subs), 2)
        self.assertTrue(all("duplicate_source_files" not in c for c in subs))
        self.assertEqual(sorted(m["usage"]["output_tokens"] for c in subs for m in c["messages"] if m.get("usage"))[-1], 999)

    def test_the_nested_copy_keeps_its_meta_when_a_duplicate_sits_beside_the_session(self):
        self.parent()
        self.beside("meta")
        self.nested("meta", meta={"agentType": "explorer", "description": "look around"})

        _, subs = self.export()

        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].get("agent_type"), "explorer")

    def test_content_too_deep_to_compare_is_exported_without_comparing(self):
        from unittest import mock
        import extract_claude_code

        self.parent()
        self.nested("deep")
        self.beside("deep")
        with mock.patch.object(extract_claude_code, "_messages_digest", side_effect=RecursionError("too deep")):
            main, subs = self.export()

        # Nothing can be compared, so nothing is merged and nothing is lost
        self.assertEqual(len(main), 1)
        self.assertEqual([c["agent"] for c in subs], ["deep", "deep"])

    def test_content_that_cannot_be_compared_for_any_reason_is_still_exported(self):
        self.parent()
        self.nested("odd")
        self.beside("odd")
        for error in (TypeError("not comparable"), ValueError("not comparable")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(extract_claude_code, "_messages_digest", side_effect=error):
                    mains, subs = self.export()
                self.assertEqual(len(mains), 1)
                self.assertEqual([c["agent"] for c in subs], ["odd", "odd"])

    def test_sessions_come_out_in_the_order_the_directory_lists_them(self):
        # The order sessions had before sub-agents were exported: whatever the directory gives
        for session_id in ("sess-b", "sess-a", "sess-c"):
            self.parent(session_id)
        self.nested("x", "sess-a")
        other = self.root / "projects" / "-other"
        write_rows(other / "sess-z.jsonl", transcript("z", PARENT_USAGE, "sess-z"))
        listed = {"glob": Path.glob, "iterdir": Path.iterdir}

        def backwards(name):
            return lambda path, *args: iter(sorted(listed[name](path, *args), reverse=True))

        with mock.patch.object(Path, "glob", backwards("glob")), mock.patch.object(Path, "iterdir", backwards("iterdir")):
            mains, subs = self.export()
        self.assertEqual([c["session_id"] for c in mains], ["sess-c", "sess-b", "sess-a", "sess-z"])
        self.assertEqual([c["agent"] for c in subs], ["x"])


SURROGATE = "bad \ud800 text"


def deep(levels, leaf='"x"'):
    """JSON text for a value nested `levels` lists deep"""
    return "[" * levels + leaf + "]" * levels


def depth_that_parses_but_cannot_be_written():
    """A nesting depth this interpreter can read from a line but not serialise again, or None"""
    reads, fails = [], []
    for levels in range(1000, 120000, 1000):
        try:
            value = json.loads(deep(levels))
        except RecursionError:
            break
        reads.append(levels)
        try:
            json.dumps(value)
        except RecursionError:
            fails.append(levels)
    return fails[len(fails) // 2] if fails else None


class ExtractClaudeCodeWholeExportTests(unittest.TestCase):
    """main() over a synthetic home directory: what reaches the output file"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "home"
        self.project = self.home / ".claude" / "projects" / "-work"
        self.project.mkdir(parents=True)
        self.out = Path(tmp.name) / "out"
        self.out.mkdir()
        write_rows(self.project / "sess-1.jsonl", transcript("main", PARENT_USAGE, "sess-1"))

    def sub(self, agent_id, rows=None, session_id="sess-1", beside=False):
        path = (self.project / f"agent-{agent_id}.jsonl" if beside
                else self.project / session_id / "subagents" / f"agent-{agent_id}.jsonl")
        return write_rows(path, transcript(agent_id, SUB_USAGE, session_id, agent_id) if rows is None else rows)

    def run_main(self, encoding=None):
        """Run the whole export; returns (conversations in the output file by session id, printed text)"""
        import contextlib
        import io

        real_open = open

        def open_as(path, mode="r", **kwargs):
            if mode == "w" and encoding:
                kwargs["encoding"] = encoding
            return real_open(path, mode, **kwargs)

        cwd = os.getcwd()
        os.chdir(self.out)
        printed = io.StringIO()
        try:
            # Only the synthetic home is visible: no configured directory, no real one
            with mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=True), \
                    mock.patch.object(extract_claude_code, "open", open_as, create=True), \
                    contextlib.redirect_stdout(printed):
                extract_claude_code.main()
        finally:
            os.chdir(cwd)
        return self.written(), printed.getvalue()

    def written(self):
        files = list((self.out / "extracted_data").glob("claude_code_conversations_*.jsonl"))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_a_sub_agent_with_text_that_cannot_be_encoded_is_written_escaped(self):
        rows = transcript("sur", SUB_USAGE, "sess-1", "sur")
        rows[0]["message"]["content"] = SURROGATE
        self.sub("sur", rows)
        self.sub("fine")
        self.sub("zed")
        written, _ = self.run_main()
        self.assertEqual([c["session_id"] for c in written], ["sess-1", "agent-fine", "agent-sur", "agent-zed"])
        self.assertEqual(written[2]["messages"][0]["content"], SURROGATE)
        self.assertEqual(written[0]["subagent_sessions"], ["fine", "sur", "zed"])

    def test_a_sub_agent_whose_meta_cannot_be_encoded_is_written_escaped(self):
        path = self.sub("sur")
        path.with_name("agent-sur.meta.json").write_text(json.dumps({"agentType": SURROGATE}))
        self.sub("copy", beside=True)
        written, _ = self.run_main()
        self.assertEqual([c["session_id"] for c in written], ["sess-1", "agent-copy", "agent-sur"])
        self.assertEqual(written[2]["agent_type"], SURROGATE)

    def test_a_sub_agent_is_written_escaped_when_the_output_encoding_cannot_hold_it(self):
        rows = transcript("acc", SUB_USAGE, "sess-1", "acc")
        rows[0]["message"]["content"] = "café"
        self.sub("acc", rows)
        written, _ = self.run_main(encoding="ascii")
        self.assertEqual([c["session_id"] for c in written], ["sess-1", "agent-acc"])
        self.assertEqual(written[1]["messages"][0]["content"], "café")

    def test_text_the_output_can_hold_is_written_as_it_is(self):
        rows = transcript("acc", SUB_USAGE, "sess-1", "acc")
        rows[0]["message"]["content"] = "café"
        self.sub("acc", rows)
        self.run_main(encoding="utf-8")
        text = next((self.out / "extracted_data").glob("*.jsonl")).read_text(encoding="utf-8")
        self.assertIn("café", text)

    def test_a_session_that_cannot_be_encoded_still_stops_the_export(self):
        # Unchanged from before sub-agents were exported: a session is never dropped or rewritten quietly
        rows = transcript("main", PARENT_USAGE, "sess-1")
        rows[0]["message"]["content"] = SURROGATE
        write_rows(self.project / "sess-1.jsonl", rows)
        with self.assertRaises(UnicodeEncodeError):
            self.run_main()

    def test_a_sub_agent_that_cannot_be_serialised_costs_only_itself(self):
        self.sub("a")
        self.sub("fine")
        self.sub("z", beside=True)
        write_rows(self.project / "sess-2.jsonl", transcript("two", PARENT_USAGE, "sess-2"))
        self.sub("other", session_id="sess-2")
        real_dumps = json.dumps

        def dumps(value, **kwargs):
            # Stands in for nesting too deep to serialise, which no fixture reaches on every interpreter
            if isinstance(value, dict) and value.get("agent") in ("a", "z"):
                raise RecursionError("maximum recursion depth exceeded")
            return real_dumps(value, **kwargs)

        with mock.patch.object(extract_claude_code.json, "dumps", dumps):
            written, printed = self.run_main()
        self.assertEqual(sorted(c["session_id"] for c in written),
                         ["agent-fine", "agent-other", "sess-1", "sess-2"])
        by_id = {c["session_id"]: c for c in written}
        # The session still says the sub-agents exist, and the run says which files it left out
        self.assertEqual(by_id["sess-1"]["subagent_sessions"], ["a", "fine", "z"])
        for agent in ("a", "z"):
            self.assertIn(f"agent-{agent}.jsonl", printed)
        self.assertNotIn("agent-fine.jsonl", printed)

    def test_a_session_that_cannot_be_serialised_still_stops_the_export(self):
        real_dumps = json.dumps

        def dumps(value, **kwargs):
            if isinstance(value, dict) and value.get("session_id") == "sess-1":
                raise RecursionError("maximum recursion depth exceeded")
            return real_dumps(value, **kwargs)

        with mock.patch.object(extract_claude_code.json, "dumps", dumps):
            with self.assertRaises(RecursionError):
                self.run_main()

    def test_sub_agent_content_really_too_deep_to_serialise_does_not_stop_the_export(self):
        levels = depth_that_parses_but_cannot_be_written()
        if levels is None:
            self.skipTest("this interpreter refuses to read any nesting it cannot write")
        row = json.dumps({"type": "assistant", "uuid": "deep-s0", "parentUuid": None, "timestamp": TS,
                          "sessionId": "sess-1", "agentId": "deep",
                          "message": {"role": "assistant", "model": "claude-x", "content": [
                              {"type": "tool_use", "id": "toolu_d", "name": "Bash", "input": "DEEP"}]}})
        for agent_id in ("a-deep", "z-deep"):
            path = self.project / "sess-1" / "subagents" / f"agent-{agent_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(row.replace('"DEEP"', deep(levels)) + "\n")
        self.sub("fine")
        written, printed = self.run_main()
        self.assertEqual([c["session_id"] for c in written], ["sess-1", "agent-fine"])
        self.assertEqual(written[0]["subagent_sessions"], ["a-deep", "fine", "z-deep"])
        # Whether the reader or the writer gave up on a file, the run names it
        for agent_id in ("a-deep", "z-deep"):
            self.assertIn(f"agent-{agent_id}.jsonl", printed)

