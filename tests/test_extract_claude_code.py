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
