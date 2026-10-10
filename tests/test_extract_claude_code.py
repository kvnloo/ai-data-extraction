import hashlib
import itertools
import json
import os
import random
import shutil
import tempfile
import time
import unittest
from collections import Counter
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

    def test_a_session_that_cannot_be_encoded_is_written_escaped(self):
        # It used to stop the export there, and every conversation after it was lost with it
        rows = transcript("main", PARENT_USAGE, "sess-1")
        rows[0]["message"]["content"] = SURROGATE
        write_rows(self.project / "sess-1.jsonl", rows)
        self.sub("fine")
        written, _ = self.run_main()
        self.assertEqual([c["session_id"] for c in written], ["sess-1", "agent-fine"])
        self.assertEqual(written[0]["messages"][0]["content"], SURROGATE)

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

    def test_a_structured_tool_result_that_cannot_be_written_does_not_stop_the_export(self):
        hostile = {"nan": float("nan"), "inf": float("-inf"), "cut": "half \ud83d", "text": "t" * 10_000_000}
        rows = [user("u1", None, "go", sessionId="sess-1"), call(), result(toolUseResult=hostile)]
        write_rows(self.project / "sess-2.jsonl", rows)
        self.sub("hostile", rows)
        write_rows(self.project / "sess-4.jsonl", rows[:2] + [result(toolUseResult=json.loads(deep(40)))])
        # 50,000 levels cannot be written by json.dumps here, so the line is put together as text
        line = json.dumps(result(toolUseResult="@")).replace('"@"', deep(50000))
        (self.project / "sess-3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:2]) + line + "\n")

        written, printed = self.run_main()

        text = next((self.out / "extracted_data").glob("*.jsonl")).read_text()
        self.assertTrue(text.endswith("\n"))
        for line in text.splitlines():
            json.loads(line, parse_constant=self.fail)  # no bare NaN or Infinity reached the file
        by_id = {c["session_id"]: c for c in written}
        self.assertLessEqual({"sess-1", "sess-2", "sess-4", "agent-hostile"}, set(by_id))
        expected = {"nan": UNSERIALISABLE, "inf": UNSERIALISABLE, "cut": UNSERIALISABLE,
                    "text": marker("t" * 10_000_000)}
        for session_id in ("sess-2", "agent-hostile"):
            self.assertEqual(by_id[session_id]["messages"][-1]["structured"], expected)
        self.assertEqual(by_id["sess-4"]["messages"][-1]["structured"], UNSERIALISABLE)
        self.assertLess(len(text), 20_000)
        if "sess-3" in by_id:
            # An interpreter that can read the line exports the session and says the result was left out
            self.assertEqual(by_id["sess-3"]["messages"][-1]["structured"], UNSERIALISABLE)
        else:
            # One that cannot loses that session, as it did before, and nothing else
            self.assertIn("sess-3.jsonl", printed)



BAD = b"\xff"  # a byte that is not text in any UTF-8 name


def name_bytes(value):
    """The bytes behind a name the export wrote in its escaped form"""
    from urllib.parse import unquote_to_bytes
    if not value.startswith("bytes://"):
        raise AssertionError(f"not an escaped name: {value!r}")
    return unquote_to_bytes(value[len("bytes://"):])


class ExtractClaudeCodeNamesThatAreNotTextTests(unittest.TestCase):
    """The script itself, run over a synthetic home whose files have names that are not valid UTF-8

    Paths are built as bytes so the fixture is the same whatever locale the tests run under.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.fsencode(tmp.name)
        self.home = self.root + b"/home"
        self.project = self.home + b"/.claude/projects/-work"
        self.out = self.root + b"/out"
        os.makedirs(self.out)
        self.put(self.project + b"/sess-1.jsonl", transcript("main", PARENT_USAGE, "sess-1"))

    def put(self, path, rows=(), raw=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw if raw is not None else "".join(json.dumps(r) + "\n" for r in rows).encode())
        return path

    def sub(self, name, session=b"sess-1", rows=None, agent_id="named"):
        path = self.project + b"/" + session + b"/subagents/" + name
        return self.put(path, transcript(agent_id, SUB_USAGE, "sess-1", agent_id) if rows is None else rows)

    def export(self, stdout="utf-8:strict", home=None, expect=0, **env):
        """Run the export as a program; returns (lines of the output file as bytes, what it printed)"""
        import subprocess
        import sys

        script = Path(extract_claude_code.__file__).resolve()
        environ = {b"HOME": home or self.home, b"PYTHONUTF8": b"1", b"PYTHONIOENCODING": os.fsencode(stdout)}
        environ.update({os.fsencode(k): os.fsencode(v) for k, v in env.items()})
        for old in Path(os.fsdecode(self.out)).glob("extracted_data/*.jsonl"):
            old.unlink()
        # Nothing of the caller's environment reaches the run: no configured directory, no real home
        done = subprocess.run([sys.executable, "-s", "-B", str(script)], cwd=self.out, env=environ,
                              capture_output=True)
        self.assertEqual(done.returncode, expect, done.stderr.decode(errors="replace"))
        files = list(Path(os.fsdecode(self.out)).glob("extracted_data/*.jsonl"))
        lines = files[0].read_bytes().splitlines(keepends=True) if files else []
        return lines, done.stdout

    def conversations(self, lines):
        for line in lines:
            self.assertTrue(line.endswith(b"\n"))
        return [json.loads(line.decode("utf-8")) for line in lines]

    def test_an_empty_sub_agent_file_with_a_name_that_is_not_text_leaves_the_session_as_it_was(self):
        before, _ = self.export()
        self.sub(b"agent-" + BAD + b".jsonl", rows=[])
        lines, _ = self.export()
        self.assertEqual(len(lines), 1)
        parent = json.loads(lines[0].decode("utf-8"))
        self.assertEqual(parent.pop("subagent_sessions"), ["bytes://%FF"])
        self.assertEqual((json.dumps(parent, ensure_ascii=False) + "\n").encode("utf-8"), before[0])

    def test_a_sub_agent_named_with_bytes_is_listed_and_exported_under_one_escaped_name(self):
        path = self.sub(b"agent-" + BAD + b".jsonl")
        parent, agent = self.conversations(self.export()[0])
        self.assertEqual(parent["subagent_sessions"], ["bytes://%FF"])
        self.assertEqual(agent["agent"], "bytes://%FF")
        self.assertEqual(agent["parent_session"], parent["session_id"])
        self.assertEqual(agent["session_id"], "bytes://agent-%FF")
        self.assertEqual(name_bytes(agent["agent"]), BAD)
        self.assertEqual(name_bytes(agent["source_file"]), path)
        # A path keeps its separators, so it still reads as one
        self.assertTrue(agent["source_file"].startswith("bytes:///"))
        self.assertTrue(agent["source_file"].endswith("/sess-1/subagents/agent-%FF.jsonl"))
        self.assertEqual(len(agent["messages"]), 2)

    def test_only_the_name_is_escaped_not_the_rest_of_the_line(self):
        rows = transcript("named", SUB_USAGE, "sess-1", "named")
        rows[0]["message"]["content"] = "caf\u00e9 \u2603"
        self.sub(b"agent-" + BAD + b".jsonl", rows=rows)
        lines, _ = self.export()
        self.assertIn("caf\u00e9 \u2603".encode("utf-8"), lines[1])
        self.assertNotIn(b"\\udc", lines[1])

    def test_a_sub_agent_named_with_bytes_beside_the_session_files_is_listed_too(self):
        path = self.put(self.project + b"/agent-" + BAD + b".jsonl", transcript("named", SUB_USAGE, "sess-1", "named"))
        parent, agent = self.conversations(self.export()[0])
        self.assertEqual(parent["subagent_sessions"], [agent["agent"]])
        self.assertEqual(name_bytes(agent["source_file"]), path)

    def test_an_escaped_name_is_never_the_same_as_a_real_one(self):
        self.sub(b"agent-" + BAD + b".jsonl", agent_id="one")
        self.sub(b"agent-%FF.jsonl", agent_id="two")
        self.sub(b"agent-bytes:%FF.jsonl", agent_id="three")
        parent, *agents = self.conversations(self.export()[0])
        self.assertEqual(parent["subagent_sessions"], ["%FF", "bytes:%FF", "bytes://%FF"])
        self.assertEqual(sorted(a["agent"] for a in agents), parent["subagent_sessions"])

    def test_a_name_that_is_text_is_written_as_it_is(self):
        path = self.sub("agent-caf\u00e9 50%.jsonl".encode("utf-8"))
        parent, agent = self.conversations(self.export()[0])
        self.assertEqual(parent["subagent_sessions"], ["caf\u00e9 50%"])
        self.assertEqual(agent["agent"], "caf\u00e9 50%")
        self.assertEqual(agent["source_file"].encode("utf-8"), path)

    def test_percent_and_other_bytes_in_an_escaped_name_come_back_exactly(self):
        name = b"agent-50%41 caf\xc3\xa9 " + BAD + b"\x80.jsonl"
        path = self.sub(name)
        parent, agent = self.conversations(self.export()[0])
        self.assertEqual(name_bytes(agent["agent"]), name[len(b"agent-"):-len(b".jsonl")])
        self.assertEqual(name_bytes(agent["source_file"]), path)
        self.assertEqual(parent["subagent_sessions"], [agent["agent"]])

    def test_a_directory_named_with_bytes_under_subagents_does_not_stop_the_export(self):
        path = self.sub(b"deep-" + BAD + b"/agent-inner.jsonl")
        self.put(path[:-len(b".jsonl")] + b".meta.json", raw=json.dumps({"agentType": "Explore"}).encode())
        parent, agent = self.conversations(self.export()[0])
        self.assertEqual(parent["subagent_sessions"], ["inner"])
        self.assertEqual(agent["agent"], "inner")
        self.assertEqual(agent["agent_type"], "Explore")
        self.assertEqual(name_bytes(agent["source_file"]), path)

    def test_both_copies_of_a_sub_agent_named_with_bytes_are_named_in_the_one_conversation(self):
        nested = self.sub(b"agent-" + BAD + b".jsonl")
        beside = self.put(self.project + b"/agent-" + BAD + b".jsonl", transcript("named", SUB_USAGE, "sess-1", "named"))
        parent, agent = self.conversations(self.export()[0])
        self.assertEqual(parent["subagent_sessions"], ["bytes://%FF"])
        self.assertEqual({name_bytes(agent["source_file"])} | {name_bytes(f) for f in agent["duplicate_source_files"]},
                         {nested, beside})

    def test_an_unreadable_file_named_with_bytes_is_reported_and_costs_only_itself(self):
        self.put(self.project + b"/sess-1/subagents/agent-" + BAD + b".jsonl", raw=b'{"type": "user"}\n\xff\xfe\n')
        self.sub(b"agent-z.jsonl", agent_id="z")
        for stdout in ("utf-8:strict", "utf-8:surrogateescape"):
            lines, printed = self.export(stdout=stdout)
            parent, agent = self.conversations(lines)
            self.assertEqual(parent["subagent_sessions"], ["bytes://%FF", "z"])
            self.assertEqual(agent["agent"], "z")
            self.assertIn(b"Error processing", printed)
            self.assertIn(b"agent-", printed)

    def test_an_unreadable_session_file_named_with_bytes_is_reported_and_costs_only_itself(self):
        self.put(self.project + b"/sess-" + BAD + b".jsonl", raw=b"\xff\xfe\n")
        lines, printed = self.export()
        self.assertEqual([c["session_id"] for c in self.conversations(lines)], ["sess-1"])
        self.assertIn(b"Error processing", printed)

    def test_a_sub_agent_too_deep_to_write_under_a_directory_named_with_bytes_is_reported(self):
        levels = depth_that_parses_but_cannot_be_written()
        if levels is None:
            self.skipTest("this interpreter refuses to read any nesting it cannot write")
        row = json.dumps({"type": "assistant", "uuid": "deep-s0", "parentUuid": None, "timestamp": TS,
                          "sessionId": "sess-1", "agentId": "deep",
                          "message": {"role": "assistant", "model": "claude-x", "content": [
                              {"type": "tool_use", "id": "toolu_d", "name": "Bash", "input": "DEEP"}]}})
        self.put(self.project + b"/sess-1/subagents/d-" + BAD + b"/agent-a-deep.jsonl",
                 raw=row.replace('"DEEP"', deep(levels)).encode() + b"\n")
        self.sub(b"agent-z.jsonl", agent_id="z")
        lines, printed = self.export()
        parent, agent = self.conversations(lines)
        self.assertEqual(parent["subagent_sessions"], ["a-deep", "z"])
        self.assertEqual(agent["agent"], "z")
        self.assertIn(b"agent-a-deep.jsonl", printed)

    def test_a_session_file_named_with_bytes_is_exported_and_its_sub_agents_still_join_it(self):
        path = self.put(self.project + b"/sess-" + BAD + b".jsonl", transcript("other", PARENT_USAGE, "sess-other"))
        self.sub(b"agent-a.jsonl", session=b"sess-" + BAD, agent_id="a")
        by_id = {c["session_id"]: c for c in self.conversations(self.export()[0])}
        self.assertEqual(sorted(by_id), ["agent-a", "bytes://sess-%FF", "sess-1"])
        session = by_id["bytes://sess-%FF"]
        self.assertEqual(name_bytes(session["source_file"]), path)
        self.assertEqual(session["subagent_sessions"], ["a"])
        self.assertEqual(by_id["agent-a"]["parent_session"], "bytes://sess-%FF")
        self.assertNotIn("subagent_sessions", by_id["sess-1"])

    def test_a_session_id_in_the_entries_that_is_not_text_is_escaped_like_the_name_it_matches(self):
        self.put(self.project + b"/sess-" + BAD + b".jsonl", transcript("other", PARENT_USAGE, "sess-other"))
        self.put(self.project + b"/agent-a.jsonl", transcript("a", SUB_USAGE, " sess-\udcff ", "a"))
        self.put(self.project + b"/agent-b.jsonl", transcript("b", SUB_USAGE, "lone \ud800", "b"))
        lines, _ = self.export()
        by_id = {c["session_id"]: c for c in self.conversations(lines)}
        self.assertEqual(by_id["bytes://sess-%FF"]["subagent_sessions"], ["a"])
        self.assertEqual(by_id["agent-a"]["parent_session"], "bytes://sess-%FF")
        self.assertEqual(by_id["agent-b"]["parent_session"], "bytes://lone%20%ED%A0%80")
        self.assertNotIn(b"\\ud", b"".join(lines))

    def test_a_project_directory_named_with_bytes_is_exported(self):
        project = self.home + b"/.claude/projects/proj-" + BAD
        path = self.put(project + b"/sess-2.jsonl", transcript("two", PARENT_USAGE, "sess-2"))
        self.put(project + b"/sess-2/subagents/agent-a.jsonl", transcript("a", SUB_USAGE, "sess-2", "a"))
        by_id = {c["session_id"]: c for c in self.conversations(self.export()[0])}
        self.assertEqual(sorted(by_id), ["agent-a", "sess-1", "sess-2"])
        self.assertEqual(by_id["sess-2"]["project_name"], "bytes://proj-%FF")
        self.assertEqual(by_id["agent-a"]["project_name"], "bytes://proj-%FF")
        self.assertEqual(name_bytes(by_id["sess-2"]["source_file"]), path)
        self.assertEqual(by_id["sess-1"]["project_name"], "-work")

    def test_a_home_directory_named_with_bytes_is_exported_and_printed(self):
        home = self.root + b"/home-" + BAD
        path = self.put(home + b"/.claude/projects/-work/sess-3.jsonl", transcript("three", PARENT_USAGE, "sess-3"))
        for stdout in ("utf-8:strict", "utf-8:surrogateescape"):
            lines, printed = self.export(stdout=stdout, home=home)
            (session,) = self.conversations(lines)
            self.assertEqual(name_bytes(session["installation"]), home + b"/.claude")
            self.assertEqual(name_bytes(session["source_file"]), path)
            self.assertEqual(printed.count(b"home-"), 2)

    def test_a_configured_directory_named_with_bytes_is_exported_and_printed(self):
        configured = self.root + b"/cfg-" + BAD
        self.put(configured + b"/projects/-work/sess-4.jsonl", transcript("four", PARENT_USAGE, "sess-4"))
        lines, printed = self.export(CLAUDE_CONFIG_DIR=os.fsdecode(configured))
        by_id = {c["session_id"]: c for c in self.conversations(lines)}
        self.assertEqual(sorted(by_id), ["sess-1", "sess-4"])
        self.assertEqual(name_bytes(by_id["sess-4"]["installation"]), configured)
        # Found, processed, and counted in the breakdown
        self.assertEqual(printed.count(b"cfg-"), 3)

    def test_names_are_escaped_when_the_locale_cannot_read_them_as_text(self):
        # Without a UTF-8 locale every byte above 127 in a name is one the output file cannot hold
        path = self.sub("agent-caf\u00e9.jsonl".encode("utf-8"))
        lines, _ = self.export(stdout="utf-8:strict", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C")
        parent, agent = self.conversations(lines)
        self.assertEqual(parent["subagent_sessions"], ["bytes://caf%C3%A9"])
        self.assertEqual(agent["agent"], "bytes://caf%C3%A9")
        self.assertEqual(name_bytes(agent["source_file"]), path)


class SayTests(unittest.TestCase):
    """A printed line is shown escaped when the terminal cannot show it, never raised"""

    def said(self, text, encoding, errors="strict"):
        import contextlib
        import io

        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding=encoding, errors=errors)
        with contextlib.redirect_stdout(stream):
            extract_claude_code._say(text)
        stream.flush()
        return raw.getvalue()

    def test_text_the_terminal_can_show_is_printed_as_it_is(self):
        self.assertEqual(self.said("caf\u00e9", "utf-8"), "caf\u00e9\n".encode("utf-8"))

    def test_a_name_that_is_not_text_is_printed_escaped(self):
        self.assertEqual(self.said("x \udcff y", "utf-8"), b"x \\udcff y\n")

    def test_what_an_ascii_terminal_cannot_show_is_printed_escaped(self):
        self.assertEqual(self.said("\u26a0 caf\u00e9 \udcff", "ascii"), b"\\u26a0 caf\\xe9 \\udcff\n")

    def test_a_terminal_that_passes_bytes_through_gets_the_name_itself(self):
        self.assertEqual(self.said("x \udcff", "utf-8", "surrogateescape"), b"x \xff\n")

    def test_output_that_names_no_encoding_gets_plain_ascii(self):
        import contextlib

        class Plain:
            """Something to print to that takes ASCII only and does not say so"""
            def __init__(self):
                self.text = ""

            def write(self, text):
                text.encode("ascii")
                self.text += text

            def flush(self):
                pass

        stream = Plain()
        with contextlib.redirect_stdout(stream):
            extract_claude_code._say("caf\u00e9 \udcff")
        self.assertEqual(stream.text, "caf\\xe9 \\udcff\n")

    def test_an_error_while_reading_a_file_is_reported_on_an_ascii_terminal(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            project = os.fsencode(tmp) + b"/projects/-work"
            # A directory where a transcript should be cannot be read, whatever the locale
            os.makedirs(project + b"/sess-" + BAD + b".jsonl")
            raw = io.BytesIO()
            stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
            with contextlib.redirect_stdout(stream):
                found = extract_claude_project_conversations(os.fsdecode(os.fsencode(tmp)))
            stream.flush()
        self.assertEqual(found, [])
        self.assertIn(b"Error processing", raw.getvalue())
        self.assertIn(b"sess-\\udcff.jsonl", raw.getvalue())


# Its own figures: an earlier test changes SUB_USAGE in place
COPY_USAGE = {"input_tokens": 100, "output_tokens": 40}


def reply(uuid, parent, usage=None, session_id="sess-1", agent_id="x"):
    """One more response for a synthetic sub-agent transcript"""
    return {"type": "assistant", "uuid": uuid, "parentUuid": parent, "timestamp": TS, "requestId": f"req-{uuid}",
            "sessionId": session_id, "agentId": agent_id, "isSidechain": True,
            "message": {"id": f"msg-{uuid}", "role": "assistant", "model": "claude-x", "usage": usage or dict(COPY_USAGE),
                        "content": [{"type": "text", "text": f"{uuid} says"}]}}


def output_of(messages):
    return sum(m["usage"]["output_tokens"] for m in messages if m.get("usage"))


def output_per_conversation(conversations):
    """Output tokens summed the plain way: every conversation, every message"""
    return sum(output_of(c["messages"]) for c in conversations)


def output_by_dedupe_key(conversations):
    """Output tokens summed the way extract_claude_project_conversations documents: once for each 'dedupe_key'"""
    first = {}
    for conversation in conversations:
        for message in conversation["messages"]:
            first.setdefault(message["dedupe_key"], message)
    return output_of(first.values())


def message_identities(conversation):
    """What tells each message of an exported conversation apart, worked out here and not read from 'dedupe_key'"""
    messages = [{k: v for k, v in m.items() if k != "dedupe_key"} for m in conversation["messages"]]
    if all(isinstance(m.get("id"), str) and m["id"] for m in messages):
        return [json.dumps(m, sort_keys=True) for m in messages]
    # without entry ids no message can be said to be another
    return [(conversation["source_file"], index) for index in range(len(messages))]


def output_by_message(conversations):
    """Output tokens summed once for each different message, without trusting 'dedupe_key'"""
    first = {}
    for conversation in conversations:
        for identity, message in zip(message_identities(conversation), conversation["messages"]):
            first.setdefault(identity, message)
    return output_of(first.values())


def output_written(*files):
    """Output tokens of every different response in the rows that were written: the true total"""
    usage = {}
    for rows in files:
        for row in rows:
            if row.get("type") == "assistant":
                usage[json.dumps(row, sort_keys=True)] = row["message"]["usage"]["output_tokens"]
    return sum(usage.values())


COPY_FIELDS = ("duplicate_source_files", "contained_source_files", "overlapping_source_files")
# What the copy fields were called before one grouping replaced the comparison of two files at a time
FORMER_COPY_FIELDS = ("prefix_source_files", "reordered_source_files", "shared_prefix_with", "same_transcript_in")


def copy_fields(conversation):
    return [name for name in COPY_FIELDS + FORMER_COPY_FIELDS if name in conversation]


def overlaps(conversation):
    return {r["source_file"]: r["shared_messages"] for r in conversation.get("overlapping_source_files", [])}


class SubagentCopyTests(unittest.TestCase):
    """Files that hold one sub-agent transcript, when they are not the same byte for byte"""

    setUp = ExtractClaudeCodeSubagentTests.setUp
    parent = ExtractClaudeCodeSubagentTests.parent
    export = ExtractClaudeCodeSubagentTests.export

    def rows(self, replies=3, session_id="sess-1"):
        return transcript("x", dict(COPY_USAGE), session_id, "x", replies=replies)

    def nested(self, rows, session_id="sess-1", under="", agent="x"):
        return write_rows(self.project / session_id / "subagents" / under / f"agent-{agent}.jsonl", rows)

    def beside(self, rows):
        return write_rows(self.project / "agent-x.jsonl", rows)

    def reconciles(self, subs, conversations, per_conversation, true_total):
        self.assertEqual(len(subs), conversations)
        self.assertEqual(output_per_conversation(subs), per_conversation)
        self.assertEqual(output_by_dedupe_key(subs), true_total)
        self.assertEqual(output_by_message(subs), true_total)

    def test_a_copy_that_stops_early_is_left_out_and_the_complete_one_says_so(self):
        self.parent()
        rows = self.rows(replies=3)  # prompt, a note that is not exported, three responses
        short = self.beside(rows[:3])
        full = self.nested(rows)

        mains, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
        self.assertEqual(subs[0]["source_file"], str(full))
        self.assertEqual([m["id"] for m in subs[0]["messages"]], ["x-u", "x-s0", "x-s1", "x-s2"])
        self.assertEqual(subs[0]["duplicate_source_files"], [str(short)])
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 2, "order_matched": True}])
        self.assertEqual(mains[0]["subagent_sessions"], ["x"])

    def test_the_complete_copy_is_kept_whichever_file_is_read_first(self):
        self.parent()
        rows = self.rows(replies=3)
        full = self.beside(rows)  # the file beside the sessions is read first
        short = self.nested(rows[:4])

        _, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
        self.assertEqual(subs[0]["source_file"], str(full))
        self.assertEqual(subs[0]["duplicate_source_files"], [str(short)])
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 3, "order_matched": True}])

    def test_a_copy_cut_in_the_middle_of_a_line_is_a_shorter_copy(self):
        self.parent()
        full = self.nested(self.rows(replies=3))
        text = full.read_text()
        cut = self.beside([])
        cut.write_text(text[:len(text) - 40])

        _, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
        self.assertEqual(subs[0]["source_file"], str(full))
        self.assertEqual(subs[0]["skipped_lines"], 0)
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(cut), "shared_messages": 3, "order_matched": True}])

    def test_three_copies_of_growing_length_leave_the_longest(self):
        self.parent()
        rows = self.rows(replies=3)
        first = self.beside(rows[:3])
        second = self.nested(rows[:4])
        third = self.nested(rows, under="later")

        _, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
        self.assertEqual(subs[0]["source_file"], str(third))
        self.assertEqual(subs[0]["duplicate_source_files"], [str(first), str(second)])
        # each is described against the file that is exported, not against one that is itself left out
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(first), "shared_messages": 2, "order_matched": True},
                          {"source_file": str(second), "shared_messages": 3, "order_matched": True}])

    def test_the_shorter_copy_still_gives_its_meta_file(self):
        self.parent()
        rows = self.rows(replies=2)
        self.beside(rows)
        short = self.nested(rows[:3])
        short.with_name("agent-x.meta.json").write_text(json.dumps({"agentType": "explorer"}))

        _, subs = self.export()

        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["agent_type"], "explorer")

    def test_copies_that_part_ways_are_both_kept_and_each_names_the_other(self):
        self.parent()
        rows = self.rows(replies=2)
        left = self.beside(rows + [reply("left", "x-s1")])
        right = self.nested(rows + [reply("right", "x-s1"), reply("right-2", "right")])

        _, subs = self.export()

        # Five responses were made: two shared, one on one side, two on the other
        self.reconciles(subs, conversations=2, per_conversation=280, true_total=200)
        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(by_file[str(left)]["overlapping_source_files"],
                         [{"source_file": str(right), "shared_messages": 3}])
        self.assertEqual(by_file[str(right)]["overlapping_source_files"],
                         [{"source_file": str(left), "shared_messages": 3}])
        self.assertEqual([m["id"] for m in by_file[str(left)]["messages"]], ["x-u", "x-s0", "x-s1", "left"])
        self.assertEqual([m["id"] for m in by_file[str(right)]["messages"]],
                         ["x-u", "x-s0", "x-s1", "right", "right-2"])
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), ["overlapping_source_files"])

    def test_a_copy_two_others_carry_on_from_is_recorded_once_and_names_the_other(self):
        self.parent()
        rows = self.rows(replies=2)
        short = self.beside(rows[:3])  # left out: both other files have all of it
        left = self.nested(rows + [reply("left", "x-s1")])
        right = self.nested(rows + [reply("right", "x-s1")], under="later")

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=240, true_total=160)
        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(sorted(by_file), sorted([str(left), str(right)]))
        self.assertEqual(overlaps(by_file[str(right)]), {str(left): 3})
        self.assertEqual(overlaps(by_file[str(left)]), {str(right): 3})
        # both have as many messages, so the first read records it
        self.assertEqual(by_file[str(left)]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 2, "order_matched": True,
                           "also_in": [str(right)]}])
        self.assertEqual(by_file[str(left)]["duplicate_source_files"], [str(short)])
        self.assertEqual(copy_fields(by_file[str(right)]), ["overlapping_source_files"])

    def test_a_copy_three_others_carry_on_from_names_the_others_in_the_order_they_are_read(self):
        rows = self.rows(replies=2)
        short = self.beside(rows[:3])
        first, second, third = (self.nested(rows + [reply(name, "x-s1")], under=name) for name in ("a", "b", "c"))

        _, subs = self.export()

        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(by_file[str(first)]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 2, "order_matched": True,
                           "also_in": [str(second), str(third)]}])
        self.assertEqual(by_file[str(third)]["overlapping_source_files"],
                         [{"source_file": str(first), "shared_messages": 3}, {"source_file": str(second), "shared_messages": 3}])

    def test_a_copy_two_others_have_is_recorded_on_the_one_with_more_messages(self):
        rows = self.rows(replies=2)
        short = self.beside(rows[:3])
        left = self.nested(rows + [reply("left", "x-s1")])
        right = self.nested(rows + [reply("right", "x-s1"), reply("right-2", "right")], under="later")

        _, subs = self.export()

        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(by_file[str(right)]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 2, "order_matched": True,
                           "also_in": [str(left)]}])
        self.assertEqual(copy_fields(by_file[str(left)]), ["overlapping_source_files"])

    def test_transcripts_that_share_only_later_messages_name_each_other(self):
        self.parent()
        rows = self.rows(replies=2)
        one = self.beside(rows)
        two = self.nested([reply("other", None)] + rows[2:])

        _, subs = self.export()

        # x-s0 and x-s1 are in both; nothing is shared from the start
        self.reconciles(subs, conversations=2, per_conversation=200, true_total=120)
        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(overlaps(by_file[str(one)]), {str(two): 2})
        self.assertEqual(overlaps(by_file[str(two)]), {str(one): 2})

    def test_transcripts_that_share_no_message_carry_no_copy_fields(self):
        self.parent()
        rows = self.rows(replies=2)
        self.beside(rows)
        self.nested([reply("other", None), reply("other-2", "other")])

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=160, true_total=160)
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), [])

    def test_the_same_entries_in_another_order_are_exported_once_in_the_order_of_their_parent_ids(self):
        rows = self.rows(replies=3)
        mixed = [rows[0], rows[1], rows[4], rows[2], rows[3]]
        for in_order, nested_rows, beside_rows in (("nested", rows, mixed), ("beside", mixed, rows)):
            with self.subTest(in_order=in_order):
                nested, beside = self.nested(nested_rows), self.beside(beside_rows)
                kept, other = (nested, beside) if in_order == "nested" else (beside, nested)

                _, subs = self.export()

                self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
                self.assertEqual(subs[0]["source_file"], str(kept))
                self.assertEqual([m["id"] for m in subs[0]["messages"]], ["x-u", "x-s0", "x-s1", "x-s2"])
                self.assertEqual(subs[0]["duplicate_source_files"], [str(other)])
                self.assertEqual(subs[0]["contained_source_files"],
                                 [{"source_file": str(other), "shared_messages": 4, "order_matched": False}])

    def test_parent_ids_are_followed_through_entries_that_are_not_exported(self):
        # x-s0 answers a note that is not exported; the note answers the prompt
        rows = self.rows(replies=1)
        mixed = [rows[2], rows[1], rows[0]]
        beside, nested = self.beside(rows), self.nested(mixed)

        _, subs = self.export()

        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["source_file"], str(beside))
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(nested), "shared_messages": 2, "order_matched": False}])

    def test_without_an_order_to_follow_the_copy_under_the_session_is_kept(self):
        def unlinked(rows):
            return [dict(row, parentUuid=None) for row in rows]

        rows = unlinked(self.rows(replies=3))
        mixed = [rows[0], rows[1], rows[4], rows[2], rows[3]]
        for under_session, nested_rows, beside_rows in (("in order", rows, mixed), ("mixed", mixed, rows)):
            with self.subTest(under_session=under_session):
                nested, beside = self.nested(nested_rows), self.beside(beside_rows)

                _, subs = self.export()

                self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
                self.assertEqual(subs[0]["source_file"], str(nested))
                self.assertEqual(subs[0]["duplicate_source_files"], [str(beside)])
                self.assertEqual(subs[0]["contained_source_files"],
                                 [{"source_file": str(beside), "shared_messages": 4, "order_matched": False}])

    def test_neither_order_following_the_parent_ids_falls_back_to_the_copy_under_the_session(self):
        rows = self.rows(replies=3)
        beside = self.beside([rows[0], rows[1], rows[4], rows[2], rows[3]])
        nested = self.nested([rows[0], rows[1], rows[3], rows[2], rows[4]])

        _, subs = self.export()

        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["source_file"], str(nested))
        self.assertEqual(subs[0]["duplicate_source_files"], [str(beside)])

    def test_two_copies_under_the_session_with_no_order_to_follow_keep_the_first_read(self):
        rows = [dict(row, parentUuid=None) for row in self.rows(replies=3)]
        first = self.nested(rows)
        second = self.nested([rows[0], rows[1], rows[4], rows[2], rows[3]], under="later")

        _, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
        self.assertEqual(subs[0]["source_file"], str(first))
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(second), "shared_messages": 4, "order_matched": False}])

    def test_an_order_found_under_the_session_is_kept_from_the_first_file_that_has_it(self):
        rows = [dict(row, parentUuid=None) for row in self.rows(replies=3)]
        mixed = [rows[0], rows[1], rows[4], rows[2], rows[3]]
        beside = self.beside(rows)  # the same order is under the session too, and this file is read first
        nested = self.nested(rows)
        other_order = self.nested(mixed, under="aaa")  # read before the other file under the session

        _, subs = self.export()

        self.assertEqual([c["source_file"] for c in subs], [str(beside)])
        self.assertEqual(subs[0]["duplicate_source_files"], [str(other_order), str(nested)])
        self.assertEqual([r["order_matched"] for r in subs[0]["contained_source_files"]], [False, True])

    def test_an_order_found_only_under_another_session_does_not_decide_which_file_is_kept(self):
        rows = [dict(row, parentUuid=None) for row in self.rows(replies=3)]
        self.beside(rows)
        nested = self.nested([rows[0], rows[1], rows[4], rows[2], rows[3]])
        elsewhere = self.nested(rows, "sess-2")  # the order of the file beside the sessions, for another session

        _, subs = self.export()

        self.assertEqual(sorted(c["source_file"] for c in subs), sorted([str(nested), str(elsewhere)]))

    def test_a_parent_id_that_is_not_text_does_not_stop_the_comparison(self):
        self.parent()
        rows = self.rows(replies=3)
        rows[3]["parentUuid"] = ["not", "text"]
        nested = self.nested(rows)
        self.beside([rows[0], rows[1], rows[4], rows[2], rows[3]])

        mains, subs = self.export()

        self.assertEqual(len(mains), 1)
        self.assertEqual([c["source_file"] for c in subs], [str(nested)])

    def test_the_same_entries_a_different_number_of_times_are_not_one_transcript(self):
        rows = self.rows(replies=2)
        once_twice = self.beside([rows[0], rows[1], rows[2], rows[3], rows[3]])
        twice_once = self.nested([rows[0], rows[1], rows[2], rows[2], rows[3]])

        _, subs = self.export()

        # each has a message twice that the other has once: neither has all of the other
        self.reconciles(subs, conversations=2, per_conversation=240, true_total=80)
        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(overlaps(by_file[str(once_twice)]), {str(twice_once): 3})
        self.assertEqual(overlaps(by_file[str(twice_once)]), {str(once_twice): 3})
        for conversation in subs:
            self.assertNotIn("duplicate_source_files", conversation)

    def test_a_copy_with_a_line_twice_has_all_of_the_copy_with_it_once(self):
        rows = self.rows(replies=2)
        once = self.beside(rows)
        twice = self.nested([rows[0], rows[1], rows[2], rows[2], rows[3]])

        _, subs = self.export()

        # the response written twice is exported twice, with one key
        self.reconciles(subs, conversations=1, per_conversation=120, true_total=80)
        self.assertEqual(subs[0]["source_file"], str(twice))
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(once), "shared_messages": 3, "order_matched": True}])
        keys = [m["dedupe_key"] for m in subs[0]["messages"]]
        self.assertEqual(keys[1], keys[2])
        self.assertEqual(len(set(keys)), 3)

    def test_one_transcript_under_two_sessions_is_exported_for_each_and_each_names_the_other(self):
        self.parent("sess-1")
        self.parent("sess-2")
        rows = self.rows(replies=3)
        one = self.nested(rows, "sess-1")
        two = self.nested(rows, "sess-2")

        mains, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=240, true_total=120)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(by_session["sess-1"]["overlapping_source_files"], [{"source_file": str(two), "shared_messages": 4}])
        self.assertEqual(by_session["sess-2"]["overlapping_source_files"], [{"source_file": str(one), "shared_messages": 4}])
        for conversation in subs:
            self.assertNotIn("duplicate_source_files", conversation)
        self.assertEqual({m["session_id"]: m["subagent_sessions"] for m in mains}, {"sess-1": ["x"], "sess-2": ["x"]})

    def test_one_transcript_under_three_sessions_names_both_others(self):
        rows = self.rows(replies=1)
        files = {session: str(self.nested(rows, session)) for session in ("sess-1", "sess-2", "sess-3")}

        _, subs = self.export()

        self.reconciles(subs, conversations=3, per_conversation=120, true_total=40)
        for conversation in subs:
            others = sorted(s for s in files if s != conversation["parent_session"])
            self.assertEqual(conversation["overlapping_source_files"],
                             [{"source_file": files[s], "shared_messages": 2} for s in others])

    def test_a_shorter_copy_under_another_session_is_exported_and_both_say_what_they_share(self):
        one = self.nested(self.rows(replies=2), "sess-1")
        two = self.nested(self.rows(replies=3), "sess-2")

        _, subs = self.export()

        # sess-1 has nothing sess-2 does not have, and still keeps its sub-agent
        self.reconciles(subs, conversations=2, per_conversation=200, true_total=120)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(by_session["sess-1"]["overlapping_source_files"], [{"source_file": str(two), "shared_messages": 3}])
        self.assertEqual(by_session["sess-2"]["overlapping_source_files"], [{"source_file": str(one), "shared_messages": 3}])
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), ["overlapping_source_files"])

    def test_copies_that_part_ways_under_two_sessions_say_what_they_share(self):
        rows = self.rows(replies=2)
        one = self.nested(rows + [reply("left", "x-s1")], "sess-1")
        two = self.nested(rows + [reply("right", "x-s1")], "sess-2")

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=240, true_total=160)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(overlaps(by_session["sess-1"]), {str(two): 3})
        self.assertEqual(overlaps(by_session["sess-2"]), {str(one): 3})

    def test_different_transcripts_under_two_sessions_do_not_name_each_other(self):
        self.nested(self.rows(replies=2), "sess-1")
        self.nested(transcript("y", dict(COPY_USAGE), "sess-2", "x", replies=3), "sess-2")

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=200, true_total=200)
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), [])

    def test_copies_without_entry_ids_are_still_never_merged(self):
        rows = self.rows(replies=3)
        for row in rows:
            row.pop("uuid", None)
            row.pop("parentUuid", None)
        self.beside(rows[:3])
        self.nested(rows)
        self.nested(rows, "sess-2")

        _, subs = self.export()

        # nothing says these are the same messages, so each is counted: 1 + 3 + 3 responses
        self.reconciles(subs, conversations=3, per_conversation=280, true_total=280)
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), [])
            self.assertEqual([m["dedupe_key"] for m in conversation["messages"]],
                             [f"{conversation['source_file']}#{n}" for n in range(len(conversation["messages"]))])

    def test_copies_that_cannot_be_compared_are_all_exported_without_copy_fields(self):
        rows = self.rows(replies=3)
        self.beside(rows[:3])
        self.nested(rows)
        self.nested(rows, "sess-2")
        for error in (RecursionError("too deep"), TypeError("not comparable"), ValueError("not comparable")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(extract_claude_code, "_messages_digest", side_effect=error):
                    _, subs = self.export()
                self.assertEqual(len(subs), 3)
                for conversation in subs:
                    self.assertEqual(copy_fields(conversation), [])
                self.assertEqual(len({m["dedupe_key"] for c in subs for m in c["messages"]}), 2 + 4 + 4)

    def test_the_session_itself_is_exported_the_same_whatever_copies_its_sub_agent_has(self):
        self.parent()
        alone = json.dumps(self.export()[0][0], ensure_ascii=False)
        rows = self.rows(replies=3)
        self.beside(rows[:3])
        self.nested([rows[0], rows[1], rows[4], rows[2], rows[3]])
        self.nested(rows + [reply("more", "x-s2")], under="later")

        mains, subs = self.export()

        self.assertEqual(mains[0].pop("subagent_sessions"), ["x"])
        self.assertEqual(json.dumps(mains[0], ensure_ascii=False), alone)
        self.assertNotIn("dedupe_key", alone)
        # every response the files hold is in what was exported
        self.assertEqual({m["id"] for c in subs for m in c["messages"]}, {"x-u", "x-s0", "x-s1", "x-s2", "more"})

    def test_an_exact_copy_is_only_listed_and_the_first_file_read_is_kept(self):
        rows = self.rows(replies=3)
        beside, nested = self.beside(rows), self.nested(rows)

        _, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=120, true_total=120)
        self.assertEqual(subs[0]["source_file"], str(beside))
        self.assertEqual(subs[0]["duplicate_source_files"], [str(nested)])
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(nested), "shared_messages": 4, "order_matched": True}])
        # nothing else is added to what a transcript found once has
        nested.unlink()
        self.assertEqual(set(subs[0]) - set(self.export()[1][0]), {"duplicate_source_files", "contained_source_files"})

    def test_a_longer_copy_records_every_file_it_has_all_of(self):
        rows = self.rows(replies=3)
        in_order = self.beside(rows)
        mixed = self.nested([rows[0], rows[1], rows[4], rows[2], rows[3]])
        longer = self.nested(rows + [reply("more", "x-s2")], under="later")

        _, subs = self.export()

        self.reconciles(subs, conversations=1, per_conversation=160, true_total=160)
        self.assertEqual(subs[0]["source_file"], str(longer))
        self.assertEqual(subs[0]["duplicate_source_files"], [str(in_order), str(mixed)])
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(in_order), "shared_messages": 4, "order_matched": True},
                          {"source_file": str(mixed), "shared_messages": 4, "order_matched": False}])

    def test_parent_ids_that_go_round_in_a_circle_do_not_stop_the_export(self):
        rows = self.rows(replies=1)
        rows[0]["parentUuid"] = "note-2"
        rows[1]["parentUuid"] = "note-2"
        rows.insert(2, {"type": "attachment", "uuid": "note-2", "parentUuid": "x-a", "timestamp": TS,
                        "attachment": {"type": "date"}})
        self.nested(rows)

        _, subs = self.export()

        self.assertEqual([[m["id"] for m in c["messages"]] for c in subs], [["x-u", "x-s0"]])

    def test_one_transcript_under_two_sessions_as_two_agents_is_exported_for_each_and_each_names_the_other(self):
        rows = self.rows(replies=2)
        one = self.nested(rows, "sess-1")
        two = self.nested(rows, "sess-2", agent="y")

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=160, true_total=80)
        self.assertEqual([c["agent"] for c in subs], ["x", "y"])
        self.assertEqual(subs[0]["overlapping_source_files"], [{"source_file": str(two), "shared_messages": 3}])
        self.assertEqual(subs[1]["overlapping_source_files"], [{"source_file": str(one), "shared_messages": 3}])
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), ["overlapping_source_files"])

    def test_one_transcript_as_two_agents_of_one_session_is_exported_for_each(self):
        rows = self.rows(replies=2)
        one = self.nested(rows)
        two = self.nested(rows[:3], agent="y")

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=120, true_total=80)
        self.assertEqual(overlaps(subs[0]), {str(two): 2})
        self.assertEqual(overlaps(subs[1]), {str(one): 2})

    def test_one_transcript_under_two_sessions_in_two_orders_is_still_named_on_each(self):
        rows = self.rows(replies=3)
        one = self.nested(rows, "sess-1")
        two = self.nested([rows[0], rows[1], rows[4], rows[2], rows[3]], "sess-2")

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=240, true_total=120)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(by_session["sess-1"]["overlapping_source_files"], [{"source_file": str(two), "shared_messages": 4}])
        self.assertEqual(by_session["sess-2"]["overlapping_source_files"], [{"source_file": str(one), "shared_messages": 4}])

    def test_a_transcript_whose_session_is_not_known_names_the_copy_under_a_session(self):
        rows = [{k: v for k, v in row.items() if k != "sessionId"} for row in self.rows(replies=2)]
        beside, nested = self.beside(rows), self.nested(rows)

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=160, true_total=80)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(by_session[None]["overlapping_source_files"], [{"source_file": str(nested), "shared_messages": 3}])
        self.assertEqual(by_session["sess-1"]["overlapping_source_files"], [{"source_file": str(beside), "shared_messages": 3}])

    def test_a_shorter_copy_whose_session_is_not_known_is_exported_and_names_the_longer_one(self):
        rows = self.rows(replies=3)
        beside = self.beside([{k: v for k, v in row.items() if k != "sessionId"} for row in rows[:4]])
        nested = self.nested(rows)

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=200, true_total=120)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(overlaps(by_session[None]), {str(nested): 3})
        self.assertEqual(overlaps(by_session["sess-1"]), {str(beside): 3})

    def test_a_shorter_copy_that_names_another_session_is_exported_for_that_session(self):
        self.parent("sess-1")
        self.parent("sess-2")
        rows = self.rows(replies=3, session_id="sess-2")
        beside = self.beside(rows[:4])
        nested = self.nested(rows, "sess-1")

        mains, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=200, true_total=120)
        by_session = {c["parent_session"]: c for c in subs}
        self.assertEqual(overlaps(by_session["sess-2"]), {str(nested): 3})
        self.assertEqual(overlaps(by_session["sess-1"]), {str(beside): 3})
        self.assertEqual({m["session_id"]: m["subagent_sessions"] for m in mains}, {"sess-1": ["x"], "sess-2": ["x"]})

    def test_copies_with_empty_entry_ids_are_never_merged(self):
        rows = [dict(row, uuid="", parentUuid=None) for row in self.rows(replies=2)]
        self.beside(rows)
        self.nested(rows)

        _, subs = self.export()

        self.reconciles(subs, conversations=2, per_conversation=160, true_total=160)
        for conversation in subs:
            self.assertEqual(copy_fields(conversation), [])

    def test_a_parent_id_that_names_nothing_in_the_file_does_not_count_against_an_order(self):
        rows = self.rows(replies=3)
        rows[0]["parentUuid"] = "an-entry-of-the-session"
        beside = self.beside(rows)
        nested = self.nested([rows[0], rows[1], rows[4], rows[2], rows[3]])

        _, subs = self.export()

        self.assertEqual([c["source_file"] for c in subs], [str(beside)])
        self.assertEqual(subs[0]["duplicate_source_files"], [str(nested)])

    def test_an_entry_written_twice_is_in_order_once_it_has_been_seen(self):
        rows = self.rows(replies=3)
        prompt, note, first, second, third = rows
        beside = self.beside([prompt, note, first, second, first, third])
        nested = self.nested([first, prompt, note, second, first, third])

        _, subs = self.export()

        self.assertEqual([c["source_file"] for c in subs], [str(beside)])
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(nested), "shared_messages": 5, "order_matched": False}])

    def test_the_kept_copy_keeps_its_own_meta_when_both_copies_have_one(self):
        rows = self.rows(replies=2)
        self.beside(rows[:3]).with_name("agent-x.meta.json").write_text(
            json.dumps({"agentType": "from-the-short-copy", "description": "only the short copy says this"}))
        self.nested(rows).with_name("agent-x.meta.json").write_text(json.dumps({"agentType": "from-the-full-copy"}))

        _, subs = self.export()

        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["agent_type"], "from-the-full-copy")
        self.assertEqual(subs[0]["agent_description"], "only the short copy says this")

    def test_a_copy_is_compared_with_every_file_of_the_project(self):
        rows = self.rows(replies=2)
        other = self.beside([reply("other", None)])
        first = self.nested(rows)
        second = self.nested(rows, under="later")

        _, subs = self.export()

        self.assertEqual(sorted(c["source_file"] for c in subs), sorted([str(other), str(first)]))


def thread(count, prefix="x", after=None, first=0, session_id="sess-1"):
    """A prompt (when nothing comes before) and `count` responses, each with output tokens no sum of others gives"""
    rows = [] if after else [{"type": "user", "uuid": f"{prefix}-u", "parentUuid": None, "timestamp": TS, "cwd": "/work",
                              "sessionId": session_id, "agentId": "x", "isSidechain": True,
                              "message": {"role": "user", "content": f"{prefix} prompt"}}]
    parent = after or f"{prefix}-u"
    for n in range(first, first + count):
        rows.append(reply(f"{prefix}-{n}", parent, {"input_tokens": 1, "output_tokens": 2 ** n}, session_id))
        parent = f"{prefix}-{n}"
    return rows


def shuffled(rows, rng):
    """The same rows in an order that is not theirs"""
    mixed = list(rows)
    while mixed == rows:
        rng.shuffle(mixed)
    return mixed


COPY_KINDS = ("full", "prefix", "reordered", "reordered-prefix", "diverging", "reordered-diverging")
COPY_PLACES = ("same", "same", "same", "other-session", "other-agent")


def generated_copies(seed):
    """2 to 5 copies of one transcript: [(kind, rows, session, agent)], the first always under sess-1 as agent x"""
    rng = random.Random(seed)
    base = thread(rng.randint(3, 6))
    copies, bit = [], 10
    for number in range(2 + seed % 4):
        kind = rng.choice(COPY_KINDS)
        rows = base
        if "prefix" in kind:
            rows = base[:rng.randint(2, len(base) - 1)]
        if "diverging" in kind:
            stem = base[:rng.randint(2, len(base))]
            own = rng.randint(1, 2)
            rows = stem + thread(own, f"fork{number}", after=stem[-1]["uuid"], first=bit)
            bit += own
        if "reordered" in kind:
            rows = shuffled(rows, rng)
        place = rng.choice(COPY_PLACES) if number else "same"
        copies.append((f"{kind} {place}", rows, "sess-2" if place == "other-session" else "sess-1",
                       "y" if place == "other-agent" else "x"))
    return copies


def count_of_each(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def shared_count(one, other):
    """How many messages two lists of entry ids have in common, an id both have twice counting twice"""
    one, other = count_of_each(one), count_of_each(other)
    return sum(min(times, other.get(uuid, 0)) for uuid, times in one.items())


def met_in_order(part, whole):
    position = 0
    for uuid in part:
        if uuid not in whole[position:]:
            return False
        position = whole.index(uuid, position) + 1
    return True


class SubagentCopyGroupingTests(unittest.TestCase):
    """Any number of files that hold some of one transcript: one outcome, whatever the files are called"""

    setUp = ExtractClaudeCodeSubagentTests.setUp
    parent = ExtractClaudeCodeSubagentTests.parent
    export = ExtractClaudeCodeSubagentTests.export
    nested = SubagentCopyTests.nested
    beside = SubagentCopyTests.beside

    def place(self, copies, names):
        """Write each copy under its session as its agent, in the directory named for it; returns {path: copy}"""
        shutil.rmtree(self.project)
        self.project.mkdir()
        self.parent("sess-1")
        self.parent("sess-2")
        return {str(self.nested(rows, session, name, agent)): (rows, session, agent)
                for (_, rows, session, agent), name in zip(copies, names)}

    def outcome(self, written):
        """Check everything the export says against the files written; returns what must not depend on their names"""
        mains, subs = self.export()
        ids = {path: [row["uuid"] for row in rows] for path, (rows, _, _) in written.items()}
        owner = {path: (session, agent) for path, (_, session, agent) in written.items()}
        exported = {c["source_file"]: c for c in subs}
        self.assertEqual(len(exported), len(subs))
        self.assertLessEqual(set(exported), set(written))

        # the four sums
        true_total = output_written(*(rows for rows, _, _ in written.values()))
        self.assertEqual(output_by_dedupe_key(subs), true_total)
        self.assertEqual(output_by_message(subs), true_total)
        self.assertGreaterEqual(output_per_conversation(subs), true_total)
        # nothing a file holds is missing from the export
        self.assertEqual({m["id"] for c in subs for m in c["messages"]}, {uuid for path in ids for uuid in ids[path]})
        for path, conversation in exported.items():
            self.assertEqual([m["id"] for m in conversation["messages"]], ids[path])
            self.assertEqual((conversation["parent_session"], conversation["agent"]), owner[path])
            self.assertEqual(all("dedupe_key" in m for m in conversation["messages"]), True)

        # a file is exported or recorded once, on a conversation of its session and agent that has all of it
        recorded = {}
        for path, conversation in exported.items():
            records = conversation.get("contained_source_files", [])
            self.assertEqual(conversation.get("duplicate_source_files", []), [r["source_file"] for r in records])
            for record in records:
                left_out = record["source_file"]
                self.assertNotIn(left_out, recorded)
                self.assertNotIn(left_out, exported)
                recorded[left_out] = (path, record)
                self.assertEqual(owner[left_out], owner[path])
                self.assertEqual(record["shared_messages"], len(ids[left_out]))
                self.assertEqual(shared_count(ids[left_out], ids[path]), len(ids[left_out]))
                self.assertEqual(record["order_matched"], met_in_order(ids[left_out], ids[path]))
                have_it_all = [other for other in exported
                               if shared_count(ids[left_out], ids[other]) == len(ids[left_out])]
                self.assertEqual(sorted([path] + record.get("also_in", [])), sorted(have_it_all))
                self.assertEqual("also_in" in record, len(have_it_all) > 1)
                # recorded on the one with the most messages
                self.assertEqual(len(ids[path]), max(len(ids[other]) for other in have_it_all if owner[other] == owner[path]))
        self.assertEqual(sorted(list(exported) + list(recorded)), sorted(written))
        for path in exported:
            # no exported conversation has all of another of its own session and agent
            for other in exported:
                if other != path and owner[other] == owner[path]:
                    self.assertLess(shared_count(ids[path], ids[other]), len(ids[path]))
            # and each names every exported conversation it shares a message with
            sharing = {other: shared_count(ids[path], ids[other]) for other in exported if other != path}
            self.assertEqual(overlaps(exported[path]), {other: n for other, n in sharing.items() if n})
            self.assertEqual(len(exported[path].get("overlapping_source_files", [])), len(overlaps(exported[path])))
        # a session and agent that has a file has a conversation
        self.assertEqual({owner[path] for path in exported}, set(owner.values()))
        self.assertEqual({m["session_id"]: m["subagent_sessions"] for m in mains if "subagent_sessions" in m},
                         {session: sorted({a for s, a in owner.values() if s == session})
                          for session in {s for s, _ in owner.values()}})

        def content(path):
            return (owner[path], tuple(sorted(ids[path])))

        return {"exported": sorted(content(path) for path in exported),
                "left out": sorted((content(left_out), record["shared_messages"],
                                    sorted(content(p) for p in [path] + record.get("also_in", [])))
                                   for left_out, (path, record) in recorded.items()),
                "overlapping": sorted((content(path), content(other), n)
                                      for path in exported for other, n in overlaps(exported[path]).items()),
                "sums": (output_by_dedupe_key(subs), output_by_message(subs), true_total)}

    def test_generated_copies_reconcile_whatever_their_directories_are_called(self):
        names = ["", "a", "b", "c", "d"]
        kinds = set()
        for seed in range(48):
            copies = generated_copies(seed)
            kinds.update(word for kind, _, _, _ in copies for word in kind.split())
            orders = list(itertools.permutations(names[:len(copies)])) if len(copies) <= 4 else [names, names[::-1]]
            with self.subTest(seed=seed, copies=[kind for kind, _, _, _ in copies]):
                outcomes = [self.outcome(self.place(copies, order)) for order in orders]
                for order, outcome in zip(orders, outcomes):
                    self.assertEqual(outcome, outcomes[0], order)
        # what was generated did mix every kind of copy
        self.assertEqual(kinds, {"full", "prefix", "reordered", "reordered-prefix", "diverging", "reordered-diverging",
                                 "same", "other-session", "other-agent"})

    def test_three_copies_reconcile_whichever_directories_they_are_in(self):
        rows = thread(5)
        mixed = [rows[3], rows[1], rows[2], rows[0]]
        copies = [("full", rows, "sess-1", "x"), ("reordered-prefix", mixed, "sess-1", "x"),
                  ("prefix", rows[:4], "sess-1", "x")]
        for order in itertools.permutations(["", "a", "b"]):
            with self.subTest(order=order):
                written = self.place(copies, order)
                outcome = self.outcome(written)
                full = next(path for path, (copy, _, _) in written.items() if copy is rows)
                _, subs = self.export()
                # 1 + 2 + 4 + 8 + 16 output tokens were spent, and that is what the one conversation says
                self.assertEqual([c["source_file"] for c in subs], [full])
                self.assertEqual(output_per_conversation(subs), 31)
                self.assertEqual(outcome["sums"], (31, 31, 31))
                self.assertEqual(sorted(r["order_matched"] for r in subs[0]["contained_source_files"]), [False, True])

    def test_a_full_copy_beside_the_sessions_and_two_partial_copies_under_the_session_are_one_conversation(self):
        self.parent()
        rows = thread(5)
        full = self.beside(rows)
        mixed = self.nested([rows[3], rows[1], rows[2], rows[0]], under="a")
        short = self.nested(rows[:4], under="b")

        _, subs = self.export()

        self.assertEqual([c["source_file"] for c in subs], [str(full)])
        self.assertEqual(output_per_conversation(subs), 31)
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(mixed), "shared_messages": 4, "order_matched": False},
                          {"source_file": str(short), "shared_messages": 4, "order_matched": True}])

    def test_a_copy_in_another_order_has_all_of_a_shorter_copy(self):
        rows = thread(5)
        short = self.beside(rows[:4])
        mixed = self.nested([rows[5], rows[1], rows[2], rows[3], rows[4], rows[0]])

        _, subs = self.export()

        self.assertEqual([c["source_file"] for c in subs], [str(mixed)])
        self.assertEqual(output_per_conversation(subs), 31)
        self.assertEqual(subs[0]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 4, "order_matched": False}])

    def test_a_copy_in_another_order_and_a_copy_that_goes_another_way_say_what_they_share(self):
        rows = thread(4)
        mixed = self.beside([rows[4], rows[1], rows[2], rows[3], rows[0]])
        forked = self.nested(rows[:3] + thread(1, "fork", after="x-1", first=8))

        _, subs = self.export()

        self.assertEqual(len(subs), 2)
        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(overlaps(by_file[str(mixed)]), {str(forked): 3})
        self.assertEqual(overlaps(by_file[str(forked)]), {str(mixed): 3})
        self.assertEqual(output_per_conversation(subs), 15 + 3 + 256)
        self.assertEqual(output_by_dedupe_key(subs), 15 + 256)
        self.assertEqual(output_by_message(subs), 15 + 256)

    def test_two_conversations_hold_what_their_lengths_and_shared_count_say(self):
        rows = thread(4)
        self.beside(rows[:3] + thread(2, "left", after="x-1", first=8))
        self.nested(rows + thread(1, "right", after="x-3", first=12))

        _, subs = self.export()

        (shared,) = set(overlaps(subs[0]).values()) | set(overlaps(subs[1]).values())
        different = len({m["dedupe_key"] for c in subs for m in c["messages"]})
        self.assertEqual(len(subs[0]["messages"]) + len(subs[1]["messages"]) - shared, different)
        self.assertEqual(different, 8)

    def test_three_conversations_that_share_with_one_another_are_counted_by_key_not_by_shared_counts(self):
        rows = thread(3)
        for number, name in enumerate(("", "a", "b")):
            self.nested(rows + thread(1, f"fork{number}", after="x-2", first=8 + number), under=name)

        _, subs = self.export()

        self.assertEqual(len(subs), 3)
        pairs = sum(n for c in subs for n in overlaps(c).values()) // 2
        different = len({m["dedupe_key"] for c in subs for m in c["messages"]})
        self.assertEqual(different, 4 + 3)
        # the four messages all three hold are taken off three times, where twice would be right
        self.assertEqual(sum(len(c["messages"]) for c in subs) - pairs, different - 4)
        self.assertEqual(output_per_conversation(subs), 3 * 7 + 256 + 512 + 1024)
        self.assertEqual(output_by_dedupe_key(subs), 7 + 256 + 512 + 1024)
        self.assertEqual(output_by_message(subs), 7 + 256 + 512 + 1024)

    def test_the_order_a_directory_lists_its_files_in_changes_nothing(self):
        self.parent()
        rows = thread(5)
        self.beside(rows[:3] + thread(1, "fork", after="x-1", first=9))
        self.nested([rows[3], rows[1], rows[2], rows[0]], under="a")
        self.nested(rows[:4], under="b")
        self.nested(rows, under="c")
        self.nested(rows, "sess-2")
        self.nested(rows[:2], agent="y")
        listed = Path.glob

        def as_listed(mains, subs):
            return [json.dumps(c, sort_keys=True) for c in mains], [json.dumps(c, sort_keys=True) for c in subs]

        expected = as_listed(*self.export())
        for name, order in (("backwards", lambda found: found[::-1]),
                            ("shuffled", lambda found: random.Random(5).sample(found, len(found)))):
            with self.subTest(order=name):
                with mock.patch.object(Path, "glob", lambda path, pattern: iter(order(sorted(listed(path, pattern))))):
                    self.assertEqual(as_listed(*self.export()), expected)

    def test_every_sub_agent_message_has_a_key_and_a_message_of_the_session_has_none(self):
        self.parent()
        rows = thread(3)
        self.nested(rows)
        self.nested(rows[:3], "sess-2")

        mains, subs = self.export()

        self.assertEqual([m for c in mains for m in c["messages"] if "dedupe_key" in m], [])
        keys = [[m["dedupe_key"] for m in c["messages"]] for c in subs]
        self.assertEqual(keys[1], keys[0][:3])
        self.assertEqual(len(set(keys[0])), 4)
        # the key is the digest of the message as it is exported without the key
        for conversation in subs:
            for message in conversation["messages"]:
                rest = {k: v for k, v in message.items() if k != "dedupe_key"}
                self.assertEqual(message["dedupe_key"],
                                 hashlib.sha256(json.dumps([rest], sort_keys=True).encode()).hexdigest())

    def test_the_same_entry_id_with_other_content_is_another_message(self):
        rows = thread(3)
        changed = json.loads(json.dumps(rows[:3]))
        changed[2]["message"]["content"][0]["text"] = "said differently"
        one = self.beside(changed)
        two = self.nested(rows)

        _, subs = self.export()

        # x-1 was written two ways: both are exported and both count
        self.assertEqual(len(subs), 2)
        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(overlaps(by_file[str(one)]), {str(two): 2})
        self.assertEqual(output_by_dedupe_key(subs), 1 + 2 + 4 + 2)
        self.assertEqual(output_by_message(subs), 1 + 2 + 4 + 2)

    def test_a_file_left_out_gives_its_meta_and_the_first_read_is_asked_first(self):
        rows = thread(3)
        self.nested(rows, under="z")
        for name, kind in (("a", "from-a"), ("b", "from-b")):
            self.nested(rows[:2], under=name).with_name("agent-x.meta.json").write_text(json.dumps({"agentType": kind}))

        _, subs = self.export()

        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["agent_type"], "from-a")

    def test_a_copy_under_another_session_is_named_beside_the_file_that_records_a_left_out_copy(self):
        rows = thread(3)
        short = self.beside(rows[:3])
        full = self.nested(rows)
        elsewhere = self.nested(rows, "sess-2")

        _, subs = self.export()

        by_file = {c["source_file"]: c for c in subs}
        self.assertEqual(sorted(by_file), sorted([str(full), str(elsewhere)]))
        self.assertEqual(by_file[str(full)]["contained_source_files"],
                         [{"source_file": str(short), "shared_messages": 3, "order_matched": True,
                           "also_in": [str(elsewhere)]}])
        self.assertNotIn("contained_source_files", by_file[str(elsewhere)])
        self.assertEqual(output_by_dedupe_key(subs), 7)


ADDED_FIELDS = ("structured", "source_assistant_id", "subagent_ref", "stop_reason")
ID_LIMIT = 200            # characters of JSON, escaped, of an id in source_assistant_id and subagent_ref.agent
TOTAL_LIMIT = 1000        # characters of JSON, escaped, of each total in subagent_ref
GROWTH_LIMIT = 20_000     # bytes the added fields may add to one message
UNSERIALISABLE = {"omitted": True, "kind": "unserialisable"}
STRING_LIMIT = extract_claude_code.STRUCTURED_STRING_LIMIT
LIST_LIMIT = extract_claude_code.STRUCTURED_LIST_LIMIT
LINKING_SECONDS = 8       # bound on one export of 60,000 entries; it takes under 1 s, and took over 30 s


def call(uuid="s1", tool_id="toolu_1", parent="u1", name="Bash", **message):
    """An assistant entry that calls one tool"""
    row = assistant(uuid, parent, [{"type": "tool_use", "id": tool_id, "name": name, "input": {"command": "true"}}])
    row["message"].update(message)
    return row


def result(uuid="u2", tool_id="toolu_1", parent="s1", content="done", **extra):
    """The user-type entry that carries a tool's result; `extra` are top-level fields such as toolUseResult"""
    block = {"type": "tool_result", "tool_use_id": tool_id, "content": content}
    if "is_error" in extra:
        block["is_error"] = extra.pop("is_error")
    return user(uuid, parent, [block], **extra)


def marker(value):
    """What the export writes in place of a value it does not embed"""
    kind = {str: "string", list: "list", dict: "object"}[type(value)]
    raw = value.encode() if kind == "string" else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"omitted": True, "kind": kind, "length": len(value), "sha256": hashlib.sha256(raw).hexdigest()}


def key_marker(key):
    """The key the export writes in place of one it does not embed"""
    return "omitted-key-" + hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def shell_result(rng):
    return {"stdout": "built " * rng.choice([1, 3, 900]), "stderr": "", "interrupted": False,
            "returnCode": rng.randint(0, 2)}


def edit_result(rng):
    return {"filePath": "/work/app.py", "oldString": "a = 1", "newString": "a = 2",
            "structuredPatch": [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                                 "lines": ["-a = 1", "+a = 2"] * rng.choice([1, 80])}]}


def read_result(rng):
    lines = rng.choice([2, 400])
    return {"type": "text", "file": {"filePath": "/work/app.py", "content": "import os\n" * lines,
                                     "numLines": lines, "startLine": 1, "totalLines": lines}}


def search_result(rng):
    count = rng.choice([0, 3, 250])
    return {"filenames": [f"/work/src/module_{n}.py" for n in range(count)], "numFiles": count, "durationMs": 12}


def launch_result(rng):
    usage = {"input_tokens": rng.randint(1, 900), "output_tokens": rng.randint(1, 90)}
    return {"status": "completed", "agentId": "g", "totalTokens": sum(usage.values()), "totalDurationMs": 4200,
            "usage": usage, "content": [{"type": "text", "text": "report " * rng.choice([2, 700])}]}


def error_result(rng):
    return "Error: " + "command not found " * rng.choice([1, 300])


RESULTS = (shell_result, edit_result, read_result, search_result, launch_result, error_result)
CONTENTS = ("ok", [{"type": "text", "text": "ok"}],
            [{"type": "text", "text": "shot"},
             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}])


def generated_session(seed, new_fields):
    """A synthetic session of tool calls; with `new_fields`, some entries carry what the export used to drop"""
    rng = random.Random(seed)
    rows = [user("g-u0", None, f"task {seed}", sessionId="sess-1")]
    for n in range(rng.randint(1, 6)):
        tool_id = f"toolu_{seed}_{n}"
        made = call(f"g-s{n}", tool_id, rows[-1]["uuid"], name=rng.choice(["Bash", "Edit", "Read", "Glob", "Task"]),
                    id=f"msg_{seed}_{n}")
        back = result(f"g-r{n}", tool_id, f"g-s{n}", content=rng.choice(CONTENTS))
        if rng.random() < 0.5:
            back["message"]["content"][0]["is_error"] = rng.random() < 0.5
        # The same draws are made either way, so the two forms of one seed differ only in these fields
        stop_reason = rng.choice(["tool_use", "end_turn", None]) if rng.random() < 0.8 else "absent"
        tool_use_result = rng.choice(RESULTS)(rng) if rng.random() < 0.8 else None
        named = rng.random() < 0.6
        if new_fields:
            if stop_reason != "absent":
                made["message"]["stop_reason"] = stop_reason
            if tool_use_result is not None:
                back["toolUseResult"] = tool_use_result
            if named:
                back["sourceToolAssistantUUID"] = f"g-s{n}"
        rows += [made, back]
    return rows


def without_added(conversations, also=()):
    return [{**c, "messages": [{k: v for k, v in m.items() if k not in ADDED_FIELDS + tuple(also)}
                               for m in c["messages"]]} for c in conversations]


class ExtractClaudeCodeToolResultTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.project = self.root / "projects" / "-work"
        self.project.mkdir(parents=True)

    def export(self, rows, **agents):
        """Export a session of `rows`, with sub-agent transcripts by agent id; returns all conversations"""
        write_rows(self.project / "sess-1.jsonl", rows)
        for agent_id, agent_rows in agents.items():
            write_rows(self.project / "sess-1" / "subagents" / f"agent-{agent_id}.jsonl", agent_rows)
        return extract_claude_project_conversations(self.root)

    def tools(self, *rows):
        """The tool messages of a session that starts with a prompt and goes on with `rows`"""
        messages = self.export([user("u1", None, "go"), *rows])[0]["messages"]
        return [m for m in messages if m["role"] == "tool"]

    def structured(self, tool_use_result):
        return self.tools(call(), result(toolUseResult=tool_use_result))[0]["structured"]

    def test_finding_the_calling_entry_costs_time_in_proportion_to_the_messages(self):
        # 60,000 entries that all make the call toolu_1: seconds, not minutes, whether or not a result
        # has to be linked. Comparing each entry with every one before it took over 30 s here.
        calls = [call(f"s{n}", parent=f"s{n - 1}" if n else "u1") for n in range(60000)]
        for needs_lookup, fields in ((True, {"toolUseResult": {"stdout": "ok"}}), (False, {})):
            with self.subTest(needs_lookup=needs_lookup):
                started = time.perf_counter()
                tool = self.tools(*calls, result(parent="s59999", **fields))[0]
                self.assertLess(time.perf_counter() - started, LINKING_SECONDS)
                self.assertNotIn("source_assistant_id", tool)

    def test_an_entry_is_compared_with_at_most_two_others_that_made_the_same_call(self):
        class Entry:
            compared = 0

            def __eq__(self, other):
                Entry.compared += 1
                return self is other

        messages = [{"id": Entry(), "tool_uses": [{"id": "toolu_1"}]} for _ in range(2000)]
        tool = {"tool_call_id": "toolu_1"}
        extract_claude_code._link_calling_entries([tool], messages)
        self.assertLessEqual(Entry.compared, 2 * len(messages))
        self.assertEqual(tool, {"tool_call_id": "toolu_1"})

    def test_the_messages_are_not_read_when_no_result_needs_its_calling_entry(self):
        class Unread:
            def __iter__(self):
                raise AssertionError("the messages were read")

        extract_claude_code._link_calling_entries([], Unread())
        # ... and they are read as soon as one does
        with self.assertRaises(AssertionError):
            extract_claude_code._link_calling_entries([{"tool_call_id": "toolu_1"}], Unread())

    def test_a_tool_message_carries_the_structured_result_and_the_entry_that_made_the_call(self):
        shell = {"stdout": "", "stderr": "make: *** [build] Error 2", "interrupted": False, "returnCode": 2,
                 "timing": {"wallMs": 1250.5, "killed": None}, "signals": ["SIGPIPE", 13]}

        tool, = self.tools(call(), result(is_error=True, toolUseResult=shell, sourceToolAssistantUUID="elsewhere"))

        self.assertEqual(tool["tool_call_id"], "toolu_1")
        self.assertIs(tool["is_error"], True)
        self.assertEqual(tool["structured"], shell)
        # What the entry says is used as it is, not replaced by a search
        self.assertEqual(tool["source_assistant_id"], "elsewhere")

    def test_without_the_new_fields_a_tool_message_is_what_it_was(self):
        tool, = self.tools(call(), result())

        self.assertEqual(list(tool), ["role", "tool_call_id", "content", "is_error", "timestamp", "id", "parent_id"])
        self.assertIs(tool["is_error"], False)

    def test_the_calling_entry_is_found_by_its_tool_use_when_the_result_does_not_name_it(self):
        tool, = self.tools(call(), result(toolUseResult={"stdout": "ok"}))
        self.assertEqual(tool["source_assistant_id"], "s1")

        tool, = self.tools(call(), result(toolUseResult=0))
        self.assertEqual((tool["structured"], tool["source_assistant_id"]), (0, "s1"))

        # The call can be written after its result
        tool, = self.tools(result(toolUseResult={"stdout": "ok"}), call())
        self.assertEqual(tool["source_assistant_id"], "s1")

    def test_the_calling_entry_is_never_guessed(self):
        found = {"stdout": "ok"}
        cases = {
            "no entry made this call": [call(tool_id="toolu_other"), result(toolUseResult=found)],
            "two entries made it": [call(), call(uuid="s9"), result(toolUseResult=found)],
            "the entry has no id": [call(uuid=None), result(toolUseResult=found)],
            "the name is not an id": [call(), result(sourceToolAssistantUUID=["s1"])],
            "the name is empty": [call(), result(sourceToolAssistantUUID="")],
            "the call id is not an id": [call(tool_id=["toolu_1"]), result(tool_id=["toolu_1"], toolUseResult=found)],
            "the entry id is empty": [call(uuid=""), result(toolUseResult=found)],
            "the entry id is not an id": [call(uuid=7), result(toolUseResult=found)],
        }
        for case, rows in cases.items():
            with self.subTest(case):
                tool, = self.tools(*rows)
                self.assertNotIn("source_assistant_id", tool)

    def test_one_entry_written_twice_is_still_one_calling_entry(self):
        tool, = self.tools(call(), call(), result(toolUseResult={"stdout": "ok"}))
        self.assertEqual(tool["source_assistant_id"], "s1")

    def test_an_entry_with_several_results_gives_its_structured_result_once(self):
        both = user("u2", "s1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "one"},
                                 {"type": "tool_result", "tool_use_id": "toolu_2", "content": "two"}],
                    toolUseResult={"agentId": "g", "totalTokens": 7}, sourceToolAssistantUUID="s1")

        first, second = self.tools(call(), both)

        self.assertEqual(first["structured"], {"agentId": "g", "totalTokens": 7})
        self.assertIn("subagent_ref", first)
        self.assertNotIn("structured", second)
        self.assertNotIn("subagent_ref", second)
        self.assertEqual([first["source_assistant_id"], second["source_assistant_id"]], ["s1", "s1"])

        del both["sourceToolAssistantUUID"]
        both["message"]["content"][1]["tool_use_id"] = "toolu_1"
        first, second = self.tools(call(), both)
        self.assertEqual([first["source_assistant_id"], second["source_assistant_id"]], ["s1", "s1"])

    def test_a_null_result_or_one_on_an_entry_without_a_tool_result_adds_nothing(self):
        tool, = self.tools(call(), result(toolUseResult=None))
        self.assertEqual(without_added([{"messages": [tool]}])[0]["messages"], [tool])

        prompt = self.export([user("u1", None, "go", toolUseResult={"stdout": "ok"}, sourceToolAssistantUUID="s1")])
        self.assertEqual(list(prompt[0]["messages"][0]), ["role", "content", "timestamp", "id", "parent_id"])

    def test_a_long_string_is_described_not_embedded(self):
        kept, long = "k" * STRING_LIMIT, "é" * (STRING_LIMIT + 1)

        structured = self.structured({"stdout": kept, "stderr": long, "nested": {"deeper": [long]}})

        self.assertEqual(structured, {"stdout": kept, "stderr": marker(long), "nested": {"deeper": [marker(long)]}})
        self.assertEqual(structured["stderr"]["length"], STRING_LIMIT + 1)
        self.assertEqual(structured["stderr"]["sha256"], hashlib.sha256(long.encode("utf-8")).hexdigest())

    def test_a_long_list_is_described_not_embedded(self):
        kept, long = list(range(LIST_LIMIT)), [{"b": 1, "a": [n]} for n in range(LIST_LIMIT + 1)]

        structured = self.structured({"filenames": kept, "matches": long})

        self.assertEqual(structured, {"filenames": kept, "matches": marker(long)})
        canonical = "[" + ",".join('{"a":[%d],"b":1}' % n for n in range(LIST_LIMIT + 1)) + "]"
        self.assertEqual(structured["matches"]["sha256"], hashlib.sha256(canonical.encode()).hexdigest())

    def test_a_result_that_is_a_string_follows_the_same_limit(self):
        self.assertEqual(self.structured("Error: no such file"), "Error: no such file")
        self.assertEqual(self.structured("E" * (STRING_LIMIT + 1)), marker("E" * (STRING_LIMIT + 1)))
        self.assertEqual(self.structured(["a", "b"]), ["a", "b"])
        self.assertEqual(self.structured(0), 0)
        self.assertIs(self.structured(False), False)
        # The limit is in characters, and the size is that of the text written with \u escapes
        self.assertEqual(self.structured("\u00e9" * STRING_LIMIT), "\u00e9" * STRING_LIMIT)
        self.assertEqual(self.structured("\U0001f600" * STRING_LIMIT), marker("\U0001f600" * STRING_LIMIT))

    def test_two_exports_of_one_input_agree(self):
        rows = [user("u1", None, "go"), call(), result(toolUseResult=read_result(random.Random(1)) | {"n": [1] * 500})]
        self.assertEqual(json.dumps(self.export(rows)), json.dumps(self.export(rows)))

    def test_an_image_and_a_whole_file_add_less_than_1_kb_each(self):
        def size(*extra, **fields):
            conversation, = self.export([user("u1", None, "go"), call(), result(content="x", **fields)])
            text = json.dumps(conversation, ensure_ascii=False)
            for payload in extra:
                self.assertNotIn(payload, text)
            return len(text.encode("utf-8"))

        plain = size()
        picture = "iVBORw0KGgo" + "QUJD" * 100_000
        source_code = "def handler(event):\n    return event\n" * 140_000
        self.assertGreater(len(source_code), 5_000_000)
        cases = {
            "image": ({"type": "image", "file": {"base64": picture, "type": "image/png", "originalSize": 300_000,
                                                "dimensions": {"originalWidth": 800, "originalHeight": 600}}},
                      "QUJDQUJD"),
            # An image small enough to pass as a short string is still an image
            "small image": ({"type": "image", "file": {"base64": "iVBORw0KGgoQUJD", "type": "image/png"}}, "iVBOR"),
            "image block": ({"content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                      "data": "iVBORw0KGgoQUJD"}}]}, "iVBOR"),
            "file read": ({"type": "text", "file": {"filePath": "/work/handler.py", "content": source_code,
                                                    "numLines": 280_000, "startLine": 1, "totalLines": 280_000}},
                          "def handler"),
        }
        for case, (tool_use_result, payload) in cases.items():
            with self.subTest(case):
                added = size(payload, toolUseResult=tool_use_result) - plain
                self.assertGreater(added, 0)
                self.assertLess(added, 1024)

        image = self.structured(cases["small image"][0])
        self.assertEqual(image, {"type": "image", "file": {"base64": marker("iVBORw0KGgoQUJD"), "type": "image/png"}})
        # 'data' is only image data beside a base64 type
        self.assertEqual(self.structured({"data": "rows: 3", "type": "table"}), {"data": "rows: 3", "type": "table"})

    def test_many_small_values_cannot_add_up_to_a_large_field(self):
        wide = {f"field_{n}": "v" * STRING_LIMIT for n in range(200)}
        grid = [["cell " * 300] * LIST_LIMIT] * LIST_LIMIT

        self.assertEqual(self.structured(wide), marker(wide))
        self.assertEqual(self.structured(grid), marker(grid))
        small = {f"field_{n}": "v" * 100 for n in range(50)}
        self.assertEqual(self.structured(small), small)

    def test_what_json_cannot_hold_is_said_to_be_left_out(self):
        self.assertEqual(self.structured({"rate": float("nan"), "limit": float("inf"), "low": float("-inf"),
                                          "cut": "half a pair \ud83d", "fine": 1.5}),
                         {"rate": UNSERIALISABLE, "limit": UNSERIALISABLE, "low": UNSERIALISABLE,
                          "cut": UNSERIALISABLE, "fine": 1.5})
        self.assertEqual(self.structured("x" * STRING_LIMIT + "\udc80"), UNSERIALISABLE)
        self.assertEqual(self.structured(float("nan")), UNSERIALISABLE)
        self.assertEqual(self.structured({"bad key \ud800": 1, "fine": 2}), {key_marker("bad key \ud800"): 1, "fine": 2})
        self.assertEqual(self.structured([float("nan")] * (LIST_LIMIT + 1))["kind"], "list")

    def test_a_result_nested_too_deeply_is_said_to_be_left_out(self):
        limit = extract_claude_code.STRUCTURED_DEPTH_LIMIT
        self.assertEqual(self.structured(json.loads(deep(limit))), json.loads(deep(limit)))
        self.assertEqual(self.structured(json.loads(deep(limit + 1))), UNSERIALISABLE)
        self.assertEqual(self.structured({"a": json.loads(deep(900))}), UNSERIALISABLE)

    def test_the_stop_reason_of_a_response_is_kept(self):
        messages = self.export([user("u1", None, "go"), call(stop_reason="tool_use"), result(),
                                call(uuid="s2", stop_reason=None), call(uuid="s3"),
                                call(uuid="s4", stop_reason=float("inf"))])[0]["messages"]

        replies = [m for m in messages if m["role"] == "assistant"]
        self.assertEqual(replies[0]["stop_reason"], "tool_use")
        self.assertNotIn("stop_reason", replies[1])
        self.assertNotIn("stop_reason", replies[2])
        self.assertEqual(replies[3]["stop_reason"], UNSERIALISABLE)

    def test_a_stop_reason_that_is_not_null_is_kept_whatever_it_is(self):
        messages = self.export([user("u1", None, "go"), call(stop_reason=""), call(uuid="s2", stop_reason=0),
                                call(uuid="s3", stop_reason=False), call(uuid="s4", stop_reason="r" * (STRING_LIMIT + 1)),
                                call(uuid="s5", stop_reason="a\ud800b")])[0]["messages"]

        self.assertEqual([m["stop_reason"] for m in messages[1:4]], ["", 0, False])
        self.assertIs(messages[3]["stop_reason"], False)
        self.assertEqual(messages[4]["stop_reason"], marker("r" * (STRING_LIMIT + 1)))
        self.assertEqual(messages[5]["stop_reason"], UNSERIALISABLE)

    def test_a_result_of_exactly_the_size_limit_is_kept(self):
        limit = extract_claude_code.STRUCTURED_SIZE_LIMIT
        at, over = ["x" * 1996] * 8, ["x" * 1996] * 7 + ["x" * 1997]
        self.assertEqual((limit, len(json.dumps(at)), len(json.dumps(over))), (16000, 16000, 16001))
        self.assertEqual(self.structured(at), at)
        self.assertEqual(self.structured(over), marker(over))
        # The size is that of the text written with \u escapes, not its characters or its UTF-8 bytes
        wide, wider = ["\u00e9" * 332] * 7 + ["\u00e9" * 332 + "x" * 32], ["\u00e9" * 332] * 7 + ["\u00e9" * 332 + "x" * 33]
        self.assertEqual(len(json.dumps(wide)), limit)
        self.assertEqual(self.structured(wide), wide)
        self.assertEqual(self.structured(wider), marker(wider))

    def test_only_the_data_of_a_base64_source_is_image_data(self):
        source = {"type": "base64", "media_type": "image/png", "note": "thumbnail", "data": "QUJD"}
        self.assertEqual(self.structured(source), {**source, "data": marker("QUJD")})

    def test_a_key_that_is_long_or_not_text_is_replaced_by_its_hash(self):
        kept, long, cut = "k" * STRING_LIMIT, "\u00e9" * (STRING_LIMIT + 1), "bad key \ud800"
        output = "E" * (STRING_LIMIT + 1)

        structured = self.structured({kept: 1, long: {"stdout": output}, cut: [2], "nested": [{long: 3}], "fine": 4})

        self.assertEqual(structured, {kept: 1, key_marker(long): {"stdout": marker(output)}, key_marker(cut): [2],
                                      "nested": [{key_marker(long): 3}], "fine": 4})
        # The form: 'omitted-key-' and the SHA-256 of the key's UTF-8 bytes, all 64 hex digits
        self.assertEqual(key_marker(long), "omitted-key-" + hashlib.sha256(long.encode("utf-8")).hexdigest())
        self.assertEqual(len(key_marker(long)), 76)
        # Two keys that are not text are still two keys
        self.assertEqual(self.structured({cut: 1, "\udcff": 2}), {key_marker(cut): 1, key_marker("\udcff"): 2})
        big = "k" * 5_000_000
        self.assertEqual(self.structured({big: 1}), {key_marker(big): 1})
        # Image data under a replaced key's object is still image data
        self.assertEqual(self.structured({long: {"base64": "QUJD"}}), {key_marker(long): {"base64": marker("QUJD")}})

    def test_a_replaced_key_never_takes_the_place_of_another_key(self):
        long, cut = "k" * (STRING_LIMIT + 1), "\udcff"
        for case, taken in {"after": {long: 1, key_marker(long): 2}, "before": {key_marker(long): 2, long: 1},
                            "nested": {"a": [{cut: 1, key_marker(cut): 2}]}}.items():
            with self.subTest(case):
                self.assertEqual(self.structured(taken), UNSERIALISABLE)
        # Alone, a key that only looks like a marker is an ordinary key
        self.assertEqual(self.structured({key_marker(long): 2, "fine": 1}), {key_marker(long): 2, "fine": 1})

    def test_a_value_that_cannot_be_described_costs_only_itself(self):
        # A string over the limit that holds half a surrogate pair has no UTF-8 bytes to hash
        cut = "x" * STRING_LIMIT + "\udc80"
        self.assertEqual(self.structured({"stdout": cut, "fine": 1}), {"stdout": UNSERIALISABLE, "fine": 1})
        self.assertEqual(self.structured({"type": "base64", "data": "\ud800"}), {"type": "base64", "data": UNSERIALISABLE})

        # A list over the limit, nested too deeply to be written as the JSON that would be hashed
        levels = depth_that_parses_but_cannot_be_written()
        if levels is None:
            self.skipTest("this interpreter refuses to read any nesting it cannot write")
        matches = "[%s%s]" % (deep(levels), ", 0" * LIST_LIMIT)
        line = json.dumps(result(toolUseResult="@@")).replace('"@@"', '{"matches": %s, "fine": 1}' % matches)
        (self.project / "sess-1.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in (user("u1", None, "go"), call())) + line + "\n")

        messages = extract_claude_project_conversations(self.root)[0]["messages"]

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[2]["structured"], {"matches": UNSERIALISABLE, "fine": 1})

    def test_an_id_the_export_adds_is_short_text_or_absent(self):
        self.assertEqual((extract_claude_code.ID_LIMIT, extract_claude_code.SUBAGENT_TOTAL_LIMIT), (ID_LIMIT, TOTAL_LIMIT))
        found = {"stdout": "ok"}
        longest = "\u00e9" * (ID_LIMIT // 6) + "ss"  # ID_LIMIT characters once escaped
        tool, = self.tools(call(), result(sourceToolAssistantUUID=longest))
        self.assertEqual(tool["source_assistant_id"], longest)

        not_ids = {"over the limit": "s" * (ID_LIMIT + 1), "5 MB": "s" * 5_000_000, "half a pair": "a\ud800b",
                   "not UTF-8": "\udcff", "a number": 7, "a list": ["s1"], "an object": {"id": "s1"}, "empty": ""}
        for case, named in not_ids.items():
            # With a structured result too: the entry named its caller, so no other is looked for
            for extra in ({}, {"toolUseResult": found}):
                with self.subTest(case, **extra):
                    tool, = self.tools(call(), result(sourceToolAssistantUUID=named, **extra))
                    self.assertNotIn("source_assistant_id", tool)
        # null names no one, like a missing field
        tool, = self.tools(call(), result(sourceToolAssistantUUID=None, toolUseResult=found))
        self.assertEqual(tool["source_assistant_id"], "s1")

        # The entry found by its tool_use is written under the same limit
        tool, = self.tools(call(uuid="s" * ID_LIMIT), result(toolUseResult=found))
        self.assertEqual(tool["source_assistant_id"], "s" * ID_LIMIT)
        tool, = self.tools(call(uuid="s" * (ID_LIMIT + 1)), result(toolUseResult=found))
        self.assertNotIn("source_assistant_id", tool)

    def test_a_launch_reference_names_an_agent_only_by_a_short_text_id(self):
        tool, = self.tools(call(), result(toolUseResult={"agentId": "\u00e9" * (ID_LIMIT // 6), "totalTokens": 5}))
        self.assertEqual(tool["subagent_ref"], {"agent": "\u00e9" * (ID_LIMIT // 6), "total_tokens": 5})

        over, big = "g" * (ID_LIMIT + 1), "g" * 5_000_000
        cases = {"over the limit": (over, over), "5 MB": (big, marker(big)), "half a pair": ("a\ud800b", UNSERIALISABLE),
                 "not UTF-8": ("\udcff", UNSERIALISABLE), "a number": (7, 7), "a list": (["g"], ["g"]),
                 "an object": ({"id": "g"}, {"id": "g"}), "null": (None, None), "empty": ("", "")}
        for case, (agent_id, kept) in cases.items():
            with self.subTest(case):
                tool, = self.tools(call(), result(toolUseResult={"agentId": agent_id, "totalTokens": 5}))
                # No reference at all, and the id is still described under 'structured'
                self.assertNotIn("subagent_ref", tool)
                self.assertEqual(tool["structured"], {"agentId": kept, "totalTokens": 5})

    def test_the_totals_of_a_launch_reference_are_small_or_described(self):
        ref = lambda **totals: self.tools(call(), result(toolUseResult={"agentId": "g", **totals}))[0]["subagent_ref"]
        fits, large = {"note": "n" * (TOTAL_LIMIT - 12)}, {"note": "n" * (TOTAL_LIMIT - 11)}
        self.assertEqual((len(json.dumps(fits)), len(json.dumps(large))), (TOTAL_LIMIT, TOTAL_LIMIT + 1))

        self.assertEqual(ref(usage=fits, totalTokens=fits), {"agent": "g", "total_tokens": fits, "usage": fits})
        self.assertEqual(ref(usage=large, totalTokens=large),
                         {"agent": "g", "total_tokens": marker(large), "usage": marker(large)})

    def test_the_added_fields_of_one_message_stay_under_20_kb_at_their_largest(self):
        wide = "\U0001f600"  # twelve characters of JSON to the character, once escaped
        launch = {"agentId": wide * (ID_LIMIT // 12), "usage": {"note": "n" * (TOTAL_LIMIT - 12)},
                  "totalTokens": "t" * (TOTAL_LIMIT - 2), "pad": []}
        size = lambda value: len(json.dumps(value))  # escaped, the longest form a line is written in
        while size(launch) + STRING_LIMIT + 4 <= extract_claude_code.STRUCTURED_SIZE_LIMIT:
            launch["pad"].append("p" * STRING_LIMIT)
        launch["pad"].append("p" * (extract_claude_code.STRUCTURED_SIZE_LIMIT - size(launch) - 4))
        self.assertEqual(size(launch), extract_claude_code.STRUCTURED_SIZE_LIMIT)

        bare, = self.tools(call(), result())
        tool, = self.tools(call(), result(toolUseResult=launch, sourceToolAssistantUUID=wide * (ID_LIMIT // 12)))

        self.assertEqual(tool["structured"], launch)
        self.assertEqual(tool["subagent_ref"], {"agent": wide * (ID_LIMIT // 12), "total_tokens": launch["totalTokens"],
                                                "usage": launch["usage"]})
        self.assertEqual(tool["source_assistant_id"], wide * (ID_LIMIT // 12))
        self.assertLess(size(tool) - size(bare), GROWTH_LIMIT)
        self.assertGreater(size(tool) - size(bare), GROWTH_LIMIT - 2000)

    def test_a_sub_agent_launch_can_be_reconciled_with_the_sub_agent_conversation(self):
        sub_usage = {"input_tokens": 100, "output_tokens": 40, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 9}
        reported = {"input_tokens": 300, "output_tokens": 120, "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 27}
        launch = {"status": "completed", "agentId": "helper", "totalTokens": 447, "totalDurationMs": 5100,
                  "totalToolUseCount": 0, "usage": reported, "content": [{"type": "text", "text": "All three pass."}]}
        parent_rows = transcript("main", PARENT_USAGE) + [
            call(parent="main-s0", name="Task", id="msg-launch", usage=PARENT_USAGE),
            result(toolUseResult=launch, sourceToolAssistantUUID="s1")]

        conversations = self.export(parent_rows, helper=transcript("helper", sub_usage, "sess-1", "helper", replies=3))

        parent, sub = conversations
        tool = parent["messages"][-1]
        self.assertEqual({key: tool["structured"][key] for key in ("agentId", "totalTokens", "usage")},
                         {"agentId": "helper", "totalTokens": 447, "usage": reported})
        self.assertEqual(tool["subagent_ref"], {"agent": "helper", "total_tokens": 447, "usage": reported})
        # The same id names the launch, the session's list of sub-agents and the sub-agent conversation
        self.assertEqual(parent["subagent_sessions"], [tool["subagent_ref"]["agent"]])
        self.assertEqual(sub["agent"], tool["subagent_ref"]["agent"])
        # What the parent was told the sub-agent used against what the sub-agent's own responses add up to
        self.assertEqual(total_usage([sub]), tool["subagent_ref"]["usage"])
        self.assertEqual(sum(total_usage([sub]).values()), tool["subagent_ref"]["total_tokens"])
        # The parent's own usage is its two responses and nothing of the sub-agent's
        self.assertEqual(total_usage([parent]), {key: 2 * value for key, value in PARENT_USAGE.items()})
        self.assertNotIn("usage", tool)

    def test_a_launch_reference_holds_only_what_the_result_states(self):
        ref = lambda tool_use_result: self.tools(call(), result(toolUseResult=tool_use_result))[0].get("subagent_ref", "absent")

        self.assertEqual(ref({"agentId": "g", "totalTokens": 5}), {"agent": "g", "total_tokens": 5})
        self.assertEqual(ref({"agentId": "g", "usage": {"output_tokens": 5}}),
                         {"agent": "g", "usage": {"output_tokens": 5}})
        self.assertEqual(ref({"agentId": "g", "totalTokens": float("nan")}),
                         {"agent": "g", "total_tokens": UNSERIALISABLE})
        # A launch still running reports no totals; a result with totals and no agent names no one
        self.assertEqual("absent", ref({"agentId": "g", "status": "async_launched"}))
        self.assertEqual("absent", ref({"totalTokens": 5, "usage": {}}))
        self.assertEqual("absent", ref({"agentId": 7, "totalTokens": 5}))
        self.assertEqual("absent", ref({"agentId": "", "totalTokens": 5}))
        self.assertEqual("absent", ref("agentId totalTokens"))
        # A total of nothing is still a total that was stated
        self.assertEqual(ref({"agentId": "g", "totalTokens": 0}), {"agent": "g", "total_tokens": 0})
        self.assertEqual(ref({"agentId": "g", "usage": {}}), {"agent": "g", "usage": {}})
        self.assertEqual(ref({"agentId": "g", "totalTokens": 0, "usage": {}}), {"agent": "g", "total_tokens": 0, "usage": {}})

    def test_the_added_fields_are_the_only_difference_they_make(self):
        for seed in range(200):
            with self.subTest(seed=seed):
                rich, bare = generated_session(seed, True), generated_session(seed, False)
                before = self.export(bare, g=bare)
                after = self.export(rich, g=rich)

                self.assertEqual(without_added(before), before)
                self.assertEqual(without_added(after, also=["dedupe_key"]),
                                 without_added(before, also=["dedupe_key"]))
                self.assertLess(len(json.dumps(after)) - len(json.dumps(before)), 12 * 3000)

    def test_a_dedupe_key_covers_the_added_fields_and_copies_still_get_equal_keys(self):
        rich = [user("u1", None, "go", sessionId="sess-1"), call(stop_reason="tool_use"),
                result(toolUseResult={"stdout": "ok"}, sourceToolAssistantUUID="s1")]
        bare = [user("u1", None, "go", sessionId="sess-1"), call(), result()]

        def keys(rows):
            write_rows(self.project / "agent-g.jsonl", rows)  # a second copy, beside the session
            session, sub = self.export(rows, g=rows)
            self.assertEqual(len(sub["duplicate_source_files"]), 1)
            for message in sub["messages"]:
                rest = {k: v for k, v in message.items() if k != "dedupe_key"}
                self.assertEqual(message["dedupe_key"],
                                 hashlib.sha256(json.dumps([rest], sort_keys=True).encode()).hexdigest())
            return [message["dedupe_key"] for message in sub["messages"]]

        with_fields, without_fields = keys(rich), keys(bare)
        self.assertEqual(with_fields[0], without_fields[0])  # the prompt has none of the new fields
        self.assertNotEqual(with_fields[1], without_fields[1])
        self.assertNotEqual(with_fields[2], without_fields[2])


HOSTILE_FIELDS = ("toolUseResult", "sourceToolAssistantUUID", "stop_reason", "agentId", "totalTokens", "usage", "a key")
HOSTILE_KINDS = ("a lone surrogate", "5 MB of text", "a number", "a list", "an object", "null", "an empty string",
                 "not UTF-8 once unescaped")
HOSTILE_ROUNDS = 6


def hostile_value(kind, seed, largest):
    """A value of `kind` no entry should carry in a field; `largest` gives the 5 MB text, else a shorter long text"""
    rng = random.Random(seed)
    return {
        "a lone surrogate": "a\ud800b",
        "5 MB of text": "t" * (5_000_000 if largest else rng.choice([ID_LIMIT + 1, STRING_LIMIT + 1, 16001, 100_000])),
        "a number": rng.choice([0, 7, -1.5, 10 ** 30]),
        "a list": [f"item {n} " * 20 for n in range(rng.choice([0, 3, LIST_LIMIT, LIST_LIMIT + 1]))],
        "an object": {f"k{n}": "v" * 900 for n in range(rng.choice([0, 2, 15, 30]))},
        "null": None,
        "an empty string": "",
        "not UTF-8 once unescaped": rng.choice(["\udcff", "\udc80\udcfe tail", "\ud83d"]),
    }[kind]


def hostile_session(seed, field, value):
    """generated_session(seed, True) with `value` in `field` of its entries (of the first only, when it is large)"""
    rows = generated_session(seed, True)
    pairs = list(zip(rows[1::2], rows[2::2]))
    for made, back in pairs[:1] if isinstance(value, str) and len(value) > 1_000_000 else pairs:
        if field == "stop_reason":
            made["message"][field] = value
        elif field in ("toolUseResult", "sourceToolAssistantUUID"):
            back[field] = value
        elif field == "a key":
            key = value if isinstance(value, str) else json.dumps(value)
            back["toolUseResult"] = {"stdout": "ok", key: "value", "nested": [{key: {"base64": "QUJD"}}]}
        else:
            back["toolUseResult"] = {"status": "completed", "agentId": "g", "totalTokens": 9,
                                     "usage": {"output_tokens": 9}, field: value}
    return rows


def hostile_sessions():
    """(number, field, kind, rows with the value, the same rows without any new field), 336 of them"""
    combinations = list(itertools.product(HOSTILE_FIELDS, HOSTILE_KINDS))
    for number in range(HOSTILE_ROUNDS * len(combinations)):
        field, kind = combinations[number % len(combinations)]
        value = hostile_value(kind, number, largest=number < len(combinations))
        yield number, field, kind, hostile_session(number, field, value), generated_session(number, False)


class ExtractClaudeCodeHostileFieldTests(unittest.TestCase):
    """main() over sessions whose new fields carry what they should not: the export still completes"""

    run_main = ExtractClaudeCodeWholeExportTests.run_main
    written = ExtractClaudeCodeWholeExportTests.written

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "home"
        self.projects = self.home / ".claude" / "projects"
        self.out = Path(tmp.name) / "out"
        self.out.mkdir()

    def test_no_value_in_a_new_field_stops_the_export_or_adds_more_than_20_kb_to_a_message(self):
        expected = {}
        for number, field, kind, rich, bare in hostile_sessions():
            for project, rows in (("-rich", rich), ("-bare", bare)):
                write_rows(self.projects / project / f"s{number}.jsonl", rows)
                if number % 7 == 3:  # some as a sub-agent transcript too, which is written by another path
                    write_rows(self.projects / project / f"s{number}" / "subagents" / f"agent-a{number}.jsonl", rows)
            expected[f"s{number}"] = (field, kind)
        self.assertGreaterEqual(len(expected), 300)
        self.assertEqual({pair for pair in expected.values()}, set(itertools.product(HOSTILE_FIELDS, HOSTILE_KINDS)))

        self.run_main(encoding="utf-8")  # main() returns: the process would exit 0

        output, = (self.out / "extracted_data").glob("claude_code_conversations_*.jsonl")
        data = output.read_bytes()
        self.assertTrue(data.endswith(b"\n"))
        conversations = [json.loads(line) for line in data.decode("utf-8").split("\n")[:-1]]
        by_file = {c["source_file"]: c for c in conversations}
        rich = [c for c in conversations if "/-rich/" in c["source_file"]]
        sub_agents = sum(1 for number in range(len(expected)) if number % 7 == 3)
        self.assertEqual(len(rich), len(expected) + sub_agents)
        self.assertEqual(len(conversations), 2 * len(rich))

        size = lambda value: len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        added = Counter()
        for after in rich:
            before = by_file[after["source_file"].replace("/-rich/", "/-bare/")]
            with self.subTest(file=after["source_file"]):
                self.assertEqual(without_added([after], also=["dedupe_key"])[0]["messages"],
                                 without_added([before], also=["dedupe_key"])[0]["messages"])
                for message, was in zip(after["messages"], before["messages"]):
                    added.update(key for key in ADDED_FIELDS if key in message)
                    self.assertLessEqual(size(message) - size(was), GROWTH_LIMIT)
        # Every added field was written somewhere, so the limits above were measured on something
        self.assertEqual(set(added), set(ADDED_FIELDS))


# The output file is UTF-8 under any locale, and what the tool-result work adds is bounded as
# escaped JSON, the longest form a line can be written in

import subprocess
import sys

STOP_REASON_LIMIT = 200   # characters of escaped JSON a stop_reason may take before a marker replaces it
SCRIPT = Path(extract_claude_code.__file__).resolve()
ACCENT = "café \U0001f600"
META_TEXT = "réviseur"
BANNER = "Searching for Claude Code installations"
# One of these costs 2, 3 and 4 bytes as UTF-8 but 6, 6 and 12 as \u escapes
UNITS = {"two bytes": "é", "three bytes": "€", "four bytes": "\U0001f600",
         "mixed": "é\U0001f600p€"}
ESCAPED_IN_ASCII = {"a quote": '"', "a backslash": "\\", "a control character": "\x01"}
C_LOCALE = {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
# name -> (the whole environment besides HOME, whether stdout can show the first line main() prints;
# where it cannot, the line is printed escaped and the export goes on: ExtractClaudeCodeExportStopsTests)
ENVIRONMENTS = {
    "LANG=C.UTF-8": ({"LANG": "C.UTF-8"}, True),
    "LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0": (C_LOCALE, False),
    "LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 PYTHONIOENCODING=utf-8": ({**C_LOCALE, "PYTHONIOENCODING": "utf-8"}, True),
    "LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 PYTHONIOENCODING=ascii:backslashreplace":
        ({**C_LOCALE, "PYTHONIOENCODING": "ascii:backslashreplace"}, True),
    "PYTHONIOENCODING=ascii": ({"PYTHONIOENCODING": "ascii"}, False),
    "PYTHONIOENCODING=latin-1": ({"PYTHONIOENCODING": "latin-1"}, False),
}


def escaped(value):
    """Characters of a value's JSON with everything outside ASCII written as \\u escapes"""
    return len(json.dumps(value))


def utf8(value):
    """Bytes of a value's JSON written as UTF-8"""
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8", "surrogatepass"))


def sized(unit, size, measure=escaped):
    """Text of `unit` repeated, filled up with "p", whose JSON measures exactly `size`"""
    each = measure(unit) - 2
    return unit * ((size - 2) // each) + "p" * ((size - 2) % each)


def padded(value, unit, size, measure=escaped):
    """`value`, an object, with texts of `unit` under "pad" so that its JSON measures exactly `size`"""
    value = {**value, "pad": []}
    each = measure(unit) - 2
    while size - measure(value) > 8 + each:
        value["pad"].append(unit * min((size - measure(value) - 8) // each, 400 // len(unit)))
    value["pad"].append("p" * (size - measure(value) - (4 if value["pad"] else 2)))
    assert measure(value) == size and len(value["pad"]) <= LIST_LIMIT, (measure(value), size)
    return value


def widest_launch(unit, size, measure=escaped, agent=None):
    """A sub-agent launch result of `size` whose id and totals are each as long as `measure` lets them be"""
    return padded({"agentId": agent or sized(unit, ID_LIMIT + 2, measure),
                   "totalTokens": sized(unit, TOTAL_LIMIT, measure),
                   "usage": {"n": sized(unit, TOTAL_LIMIT - 7, measure)}}, unit, size, measure)


def worst_cases():
    """name -> entries whose added fields are as long as the export keeps, one character longer,
    and as long as it kept while it measured UTF-8 bytes and counted an id's characters"""
    cases = {}
    for name, unit in {**UNITS, "ASCII": "a"}.items():
        old_id = (unit * ID_LIMIT)[:ID_LIMIT]
        for which, launch, caller, stop in (
                ("at the limits", widest_launch(unit, 16000), sized(unit, ID_LIMIT + 2),
                 sized(unit, STOP_REASON_LIMIT)),
                ("one over", widest_launch(unit, 16001), sized(unit, ID_LIMIT + 3),
                 sized(unit, STOP_REASON_LIMIT + 1)),
                ("at the limits in UTF-8", widest_launch(unit, 16000, utf8, old_id), old_id,
                 padded({}, unit, 16000, utf8))):
            cases[f"{name}, {which}"] = [user("u1", None, "go", sessionId="sess-1"), call(stop_reason=stop),
                                         result(toolUseResult=launch, sourceToolAssistantUUID=caller)]
    return cases


def stripped(rows):
    """The same entries without anything the tool-result work reads"""
    rows = json.loads(json.dumps(rows))
    for row in rows:
        row.pop("toolUseResult", None)
        row.pop("sourceToolAssistantUUID", None)
        row["message"].pop("stop_reason", None)
    return rows


def half_a_pair(rows):
    """The same entries after a prompt that is not UTF-8, which only a sub-agent may hold: it is written escaped"""
    rows = json.loads(json.dumps(rows))
    rows[0]["message"]["content"] = "go \ud800"
    return rows


def lay_out_worst_cases(projects):
    """Each worst case as a session and, not UTF-8, as its sub-agent, under -rich; without the new fields under -bare"""
    for number, rows in enumerate(worst_cases().values()):
        for project, session in (("-rich", rows), ("-bare", stripped(rows))):
            write_rows(projects / project / f"s{number}.jsonl", session)
            write_rows(projects / project / f"s{number}" / "subagents" / f"agent-a{number}.jsonl",
                       half_a_pair(session))


def written_size(line):
    """How a message of this output line was measured when written: escaped, or as UTF-8"""
    return escaped if line.isascii() and b"\\ud800" in line else utf8


def encoding_sessions():
    """name -> a session with ACCENT in one exported place: its entries ('rows'), its sub-agent's ('sub'),
    whether the transcripts hold UTF-8 bytes and not \\u escapes ('raw'), and where the text is exported"""
    go = user("u1", None, "go", sessionId="sess-1")
    said = [user("u1", None, ACCENT, sessionId="sess-1"), assistant("s1", "u1", [{"type": "text", "text": ACCENT}])]
    plain = [go, assistant("s1", "u1", [{"type": "text", "text": "done"}])]
    tool = lambda found: next(m for m in found[0]["messages"] if m["role"] == "tool")
    first = lambda found: found[0]["messages"][0]["content"]
    in_sub = lambda found: next(c for c in found if c.get("subagent"))["messages"][0]["content"]
    return {
        "a field exported before, escaped in the transcript": {"rows": said, "where": first},
        "a field exported before, as UTF-8 in the transcript": {"rows": said, "raw": True, "where": first},
        "structured": {"rows": [go, call(), result(toolUseResult={"stdout": ACCENT})],
                       "where": lambda found: tool(found)["structured"]["stdout"]},
        "stop_reason": {"rows": [go, call(stop_reason=ACCENT)],
                        "where": lambda found: found[0]["messages"][1]["stop_reason"]},
        "source_assistant_id": {"rows": [go, call(), result(sourceToolAssistantUUID=ACCENT)],
                                "where": lambda found: tool(found)["source_assistant_id"]},
        "subagent_ref": {"rows": [go, call(), result(toolUseResult={"agentId": ACCENT, "totalTokens": 5})],
                         "where": lambda found: tool(found)["subagent_ref"]["agent"]},
        "a sub-agent, escaped in the transcript": {"rows": plain, "sub": said, "where": in_sub},
        "a sub-agent and its meta file, as UTF-8": {"rows": plain, "sub": said, "raw": True, "where": in_sub},
    }


def lay_out(root, session):
    """A synthetic home under `root` with the one session and its sub-agent; returns the home"""
    project = root / "home" / ".claude" / "projects" / "-work"
    as_bytes = lambda value: (json.dumps(value, ensure_ascii=not session.get("raw")) + "\n").encode("utf-8")
    files = {project / "sess-1.jsonl": b"".join(map(as_bytes, session["rows"]))}
    if session.get("sub"):
        agent = project / "sess-1" / "subagents" / "agent-a1.jsonl"
        files[agent] = b"".join(map(as_bytes, session["sub"]))
        files[agent.with_name("agent-a1.meta.json")] = as_bytes({"agentType": META_TEXT, "description": META_TEXT})
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root / "home"


def export_under(script, home, out, environment, python=sys.executable):
    """Run the export as a program in the new directory `out`, with nothing in its environment but
    `environment` and the home; returns (exit status, what it wrote to stderr, the output file's bytes or None)"""
    out.mkdir(parents=True)
    done = subprocess.run([python, "-s", "-B", str(script)], cwd=out, capture_output=True,
                          env={"HOME": str(home), **environment})
    files = list(out.glob("extracted_data/*.jsonl"))
    return done.returncode, done.stderr.decode("utf-8", "replace"), files[0].read_bytes() if files else None


class ExtractClaudeCodeOutputEncodingTests(unittest.TestCase):
    """The script itself where the locale or stdout is not UTF-8: the output file is UTF-8 all the same"""

    def test_the_output_file_is_complete_utf8_whatever_the_locale_and_stdout_are(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for number, (name, session) in enumerate(encoding_sessions().items()):
            home = lay_out(Path(tmp.name) / str(number), session)
            for run, (environment, (variables, shows_banner)) in enumerate(ENVIRONMENTS.items()):
                with self.subTest(session=name, environment=environment):
                    status, stderr, data = export_under(SCRIPT, home, home.parent / f"out{run}", variables)
                    self.assertEqual(status, 0, stderr)
                    self.assertTrue(data.endswith(b"\n"))
                    found = [json.loads(line) for line in data.decode("utf-8").split("\n")[:-1]]
                    self.assertEqual(len(found), 2 if session.get("sub") else 1)
                    self.assertEqual(session["where"](found), ACCENT)
                    self.assertIn(ACCENT.encode("utf-8"), data)  # as itself, not as an escape
                    if session.get("sub"):
                        self.assertIn(META_TEXT.encode("utf-8"), data)


class ExtractClaudeCodeEscapedBoundTests(unittest.TestCase):
    """The limits on the added fields count \\u escapes, so they hold for a line written escaped too"""

    setUp = ExtractClaudeCodeToolResultTests.setUp
    export = ExtractClaudeCodeToolResultTests.export
    tools = ExtractClaudeCodeToolResultTests.tools
    structured = ExtractClaudeCodeToolResultTests.structured

    def stop_reason(self, value):
        return self.export([user("u1", None, "go"), call(stop_reason=value)])[0]["messages"][1]["stop_reason"]

    def test_a_structured_result_is_kept_up_to_the_limit_of_its_escaped_json(self):
        limit = extract_claude_code.STRUCTURED_SIZE_LIMIT
        for name, unit in UNITS.items():
            with self.subTest(name):
                kept, over = padded({"stdout": "ok"}, unit, limit), padded({"stdout": "ok"}, unit, limit + 1)
                self.assertEqual(self.structured(kept), kept)
                self.assertLess(utf8(over), limit)  # as UTF-8 it is far under the limit
                self.assertEqual(self.structured(over), marker(over))

    def test_an_id_is_kept_while_its_escaped_json_is_within_the_limit(self):
        for name, unit in {**UNITS, **ESCAPED_IN_ASCII}.items():
            with self.subTest(name):
                longest, over = sized(unit, ID_LIMIT + 2), sized(unit, ID_LIMIT + 3)
                self.assertLess(len(over), ID_LIMIT)  # in characters it is under the limit
                tool, = self.tools(call(), result(toolUseResult={"agentId": longest, "totalTokens": 5},
                                                  sourceToolAssistantUUID=longest))
                self.assertEqual((tool["source_assistant_id"], tool["subagent_ref"]["agent"]), (longest, longest))
                tool, = self.tools(call(), result(toolUseResult={"agentId": over, "totalTokens": 5},
                                                  sourceToolAssistantUUID=over))
                self.assertNotIn("source_assistant_id", tool)
                self.assertNotIn("subagent_ref", tool)
                tool, = self.tools(call(uuid=longest), result(parent=longest, toolUseResult={"stdout": "ok"}))
                self.assertEqual(tool["source_assistant_id"], longest)
                tool, = self.tools(call(uuid=over), result(parent=over, toolUseResult={"stdout": "ok"}))
                self.assertNotIn("source_assistant_id", tool)

    def test_a_total_is_kept_while_its_escaped_json_is_within_the_limit(self):
        for name, unit in UNITS.items():
            with self.subTest(name):
                kept, over = sized(unit, TOTAL_LIMIT), sized(unit, TOTAL_LIMIT + 1)
                self.assertLess(utf8(over), TOTAL_LIMIT)
                tool, = self.tools(call(), result(toolUseResult={"agentId": "g", "totalTokens": kept,
                                                                 "usage": {"n": over}}))
                self.assertEqual(tool["subagent_ref"],
                                 {"agent": "g", "total_tokens": kept, "usage": marker({"n": over})})

    def test_a_stop_reason_is_kept_while_short_and_replaced_by_a_marker_when_not(self):
        self.assertEqual(extract_claude_code.STOP_REASON_LIMIT, STOP_REASON_LIMIT)
        for value in ("end_turn", "tool_use", "", 7, True, {"type": "refusal"}, ["a"], "s" * (STOP_REASON_LIMIT - 2)):
            with self.subTest(value=value):
                self.assertEqual(self.stop_reason(value), value)
        for value in ("s" * (STOP_REASON_LIMIT - 1), {"reasons": ["r" * 150, "s" * 150]}, ["r"] * 99):
            with self.subTest(value=value):
                self.assertEqual(self.stop_reason(value), marker(value))
        for name, unit in UNITS.items():
            with self.subTest(name):
                longest, over = sized(unit, STOP_REASON_LIMIT), sized(unit, STOP_REASON_LIMIT + 1)
                self.assertLess(utf8(over), STOP_REASON_LIMIT)
                self.assertEqual(self.stop_reason(longest), longest)
                self.assertEqual(self.stop_reason(over), marker(over))


class ExtractClaudeCodeEscapedGrowthTests(unittest.TestCase):
    """main() over the largest values each added field can hold, written as UTF-8 and written escaped"""

    run_main = ExtractClaudeCodeWholeExportTests.run_main
    written = ExtractClaudeCodeWholeExportTests.written
    setUp = ExtractClaudeCodeHostileFieldTests.setUp

    def test_no_message_grows_by_more_than_20_kb_as_utf8_or_as_escapes(self):
        lay_out_worst_cases(self.projects)
        self.run_main()

        output, = (self.out / "extracted_data").glob("claude_code_conversations_*.jsonl")
        lines = {json.loads(line)["source_file"]: line for line in output.read_bytes().splitlines()}
        self.assertEqual(len(lines), 4 * len(worst_cases()))
        largest = Counter()
        for source_file, line in lines.items():
            if "/-rich/" not in source_file:
                continue
            was_line = lines[source_file.replace("/-rich/", "/-bare/")]
            after, before = json.loads(line), json.loads(was_line)
            path = written_size(line)
            self.assertIs(written_size(was_line), path)
            self.assertIs(path, escaped if "/subagents/" in source_file else utf8)
            with self.subTest(file=source_file):
                self.assertEqual(without_added([after], also=["dedupe_key"])[0]["messages"],
                                 without_added([before], also=["dedupe_key"])[0]["messages"])
                for message, was in zip(after["messages"], before["messages"]):
                    for measure in (escaped, utf8):
                        self.assertLessEqual(measure(message) - measure(was), GROWTH_LIMIT)
                    largest[path.__name__] = max(largest[path.__name__], path(message) - path(was))
        # The bound was measured on something close to it, on both paths
        self.assertGreater(min(largest["escaped"], largest["utf8"]), 18_000)

    def test_an_id_over_the_limit_is_refused_without_being_written_out(self):
        # One entry can carry thousands of results that name the same oversized caller id
        for value in ("x" * (extract_claude_code.ID_LIMIT + 1), "\u00e9" * 1_000_000):
            with mock.patch.object(extract_claude_code.json, "dumps", side_effect=AssertionError("written out")):
                self.assertIsNone(extract_claude_code._id(value))
        self.assertEqual(extract_claude_code._id("x" * extract_claude_code.ID_LIMIT), "x" * extract_claude_code.ID_LIMIT)


import contextlib
import io
import re

LONE = "\ud800"  # half of a surrogate pair: JSON can say it, UTF-8 cannot hold it
# name -> (the whole environment besides HOME, what stdout is written in): none can show the banner's emoji
STRICT_STDOUT = {
    "LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0": (C_LOCALE, "ascii"),
    "PYTHONIOENCODING=ascii": ({"PYTHONIOENCODING": "ascii"}, "ascii"),
    "PYTHONIOENCODING=latin-1": ({"PYTHONIOENCODING": "latin-1"}, "latin-1"),
    "PYTHONIOENCODING=cp1252": ({"PYTHONIOENCODING": "cp1252"}, "cp1252"),
}
# the field of a transcript entry -> the field of the exported message that holds it as it is
EXPORTED_AS_IT_IS = {"uuid": "id", "parentUuid": "parent_id", "timestamp": "timestamp"}


class ExtractClaudeCodeExportStopsTests(unittest.TestCase):
    """Neither a terminal that cannot show a line nor a session that is not text stops the whole export"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.home = self.root / "home"
        self.project = self.home / ".claude" / "projects" / "-work"
        self.home.mkdir()
        self.runs = 0

    def session(self, session_id, **changed):
        rows = transcript(session_id, PARENT_USAGE, session_id)
        rows[0]["message"]["content"] = ACCENT
        rows[0].update(changed)
        return write_rows(self.project / f"{session_id}.jsonl", rows)

    def program(self, **environment):
        """The script run as a program with nothing in its environment but `environment` and the home;
        returns (exit status, what it printed as bytes, what it wrote to stderr, the output file's bytes or None)"""
        self.runs += 1
        out = self.root / f"out{self.runs}"
        out.mkdir()
        done = subprocess.run([sys.executable, "-s", "-B", str(SCRIPT)], cwd=out, capture_output=True,
                              env={"HOME": str(self.home), **environment})
        files = list(out.glob("extracted_data/*.jsonl"))
        return (done.returncode, done.stdout, done.stderr.decode("utf-8", "replace"),
                files[0].read_bytes() if files else None)

    def printed_to_ascii(self):
        """main() here, printing to a terminal that takes ASCII only; returns what it printed"""
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict", write_through=True)
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        cwd = os.getcwd()
        os.chdir(out)
        try:
            # Only the synthetic home is visible: no configured directory, no real one
            with mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=True), contextlib.redirect_stdout(stream):
                extract_claude_code.main()
        finally:
            os.chdir(cwd)
        return raw.getvalue().decode("ascii")

    def test_the_export_is_the_same_file_whatever_stdout_can_show(self):
        self.session("sess-1")
        self.session("sess-2")
        stamp = lambda printed: re.sub(r"conversations_\d{8}_\d{6}", "conversations_STAMP", printed)
        status, shown, stderr, expected = self.program(LANG="C.UTF-8")
        self.assertEqual((status, stderr), (0, ""))
        shown = shown.decode("utf-8")
        self.assertIn("\U0001f50d " + BANNER, shown)  # as itself where it can be shown
        self.assertEqual(expected.count(b"\n"), 2)
        for name, (variables, codec) in STRICT_STDOUT.items():
            with self.subTest(name):
                status, printed, stderr, data = self.program(**variables)
                self.assertEqual((status, stderr), (0, ""))
                self.assertEqual(data, expected)
                # Every line is printed, and only what this stdout cannot show is escaped
                self.assertEqual(stamp(printed.decode(codec)),
                                 stamp(shown).encode(codec, "backslashreplace").decode(codec))
                self.assertIn("\\U0001f50d " + BANNER, printed.decode(codec))

    def test_no_installation_is_said_on_an_ascii_terminal(self):
        self.assertIn("\\u274c No Claude Code installations found!\n", self.printed_to_ascii())

    def test_an_installation_with_no_conversations_is_said_on_an_ascii_terminal(self):
        self.project.mkdir(parents=True)
        printed = self.printed_to_ascii()
        self.assertIn("\\u2705 Found 1 installation(s):\n", printed)
        self.assertIn("   \\u26a0\\ufe0f  No conversations found\n", printed)
        self.assertTrue(printed.endswith("No conversations found!\n"))

    def test_every_line_of_a_complete_run_is_printed_on_an_ascii_terminal(self):
        self.session("sess-1")
        write_rows(self.project / "sess-1" / "subagents" / "agent-a.jsonl", transcript("a", SUB_USAGE, "sess-1", "a"))
        real_dumps = json.dumps

        def dumps(value, **kwargs):
            # Stands in for nesting too deep to serialise, which no fixture reaches on every interpreter
            if isinstance(value, dict) and value.get("agent") == "a":
                raise RecursionError("maximum recursion depth exceeded")
            return real_dumps(value, **kwargs)

        with mock.patch.object(extract_claude_code.json, "dumps", dumps):
            printed = self.printed_to_ascii()
        for line in ("\\U0001f50d " + BANNER + "...\n", "\\u2705 Found 1 installation(s):\n",
                     f"\\U0001f4c2 Processing: {self.home / '.claude'}\n", "   \\u2705 2 conversations\n",
                     "\\u26a0\\ufe0f  Not written, nested too deeply to serialise: ",
                     "\\u2705 Saved to: extracted_data", "   Sub-agent sessions not written: 1\n",
                     "   Format: JSONL (one conversation per line)\n"):
            self.assertIn(line, printed)
        self.assertEqual(len(list((self.root / "out" / "extracted_data").glob("*.jsonl"))), 1)

    def test_a_session_with_an_id_that_is_not_text_costs_no_other_conversation(self):
        self.session("sess-a")
        self.session("sess-z")
        status, _, stderr, expected = self.program(LANG="C.UTF-8")
        self.assertEqual((status, stderr), (0, ""))
        for field in EXPORTED_AS_IT_IS:
            self.session(f"sess-m-{field}", **{field: LONE})
        status, _, stderr, data = self.program(LANG="C.UTF-8")
        self.assertEqual((status, stderr), (0, ""))
        lines = {json.loads(line)["session_id"]: line for line in data.decode("ascii", "surrogateescape").splitlines(True)}
        self.assertEqual(sorted(lines), ["sess-a"] + sorted(f"sess-m-{field}" for field in EXPORTED_AS_IT_IS) + ["sess-z"])
        # The others are written as they are without it, byte for byte
        self.assertEqual(sorted(line.encode("ascii", "surrogateescape") for name, line in lines.items()
                                if name in ("sess-a", "sess-z")), sorted(expected.splitlines(True)))
        self.assertIn(ACCENT.encode("utf-8"), expected)
        for field, exported in EXPORTED_AS_IT_IS.items():
            with self.subTest(field):
                line = lines[f"sess-m-{field}"]
                conversation = json.loads(line)
                # Nothing is dropped or changed: the value reads back as it was in the transcript
                self.assertEqual(conversation["messages"][0][exported], LONE)
                self.assertEqual(conversation["messages"][0]["content"], ACCENT)
                # The whole line, and only this line, has everything outside ASCII as \u escapes
                self.assertTrue(line.isascii())
                self.assertEqual(line, json.dumps(conversation) + "\n")
                self.assertIn(json.dumps(LONE), line)
