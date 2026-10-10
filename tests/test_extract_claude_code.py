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
