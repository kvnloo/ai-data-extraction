import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from corpus_to_skills import (
    GeneratedSkill,
    build_prompt,
    call_model,
    chat_completions_url,
    main,
    parse_json_response,
    sample_corpus,
    validate_skills,
    write_skills,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CorpusSamplingTests(unittest.TestCase):
    def test_samples_valid_records_within_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            records = [
                {"source": "codex", "messages": [{"role": "user", "content": str(i)}]}
                for i in range(5)
            ]
            corpus.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            sampled, selected_count, total_count = sample_corpus(
                [corpus], char_budget=240, max_conversations=3, seed=4
            )

            self.assertLessEqual(len(sampled), 240)
            self.assertGreater(selected_count, 0)
            self.assertEqual(total_count, 5)
            self.assertIn("messages", sampled)


class ModelClientTests(unittest.TestCase):
    def test_normalizes_versioned_and_unversioned_base_urls(self):
        self.assertEqual(
            chat_completions_url("http://localhost:8000/v1"),
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertEqual(
            chat_completions_url("http://localhost:8000"),
            "http://localhost:8000/v1/chat/completions",
        )

    @patch("urllib.request.OpenerDirector.open")
    def test_calls_chat_completions_without_output_token_limit(self, open_request):
        response = {"choices": [{"message": {"content": '{"skills": []}'}}]}
        open_request.return_value = FakeResponse(json.dumps(response).encode())

        content = call_model(
            base_url="http://127.0.0.1:8000/v1",
            provider="local",
            api_key=None,
            model="local-model",
            messages=build_prompt("example", 1),
            timeout=1,
        )

        request = open_request.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(content, '{"skills": []}')
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)



class RecordingTransport(urllib.request.BaseHandler):
    """Synthetic HTTP responses; the opener never opens a real connection."""

    handler_order = 100

    def __init__(self, *, status=200, location=None):
        self.status = status
        self.location = location
        self.requests = []

    def http_open(self, request):
        self.requests.append(request)
        headers = Message()
        if self.location:
            headers["Location"] = self.location
        payload = {"choices": [{"message": {"content": '{"skills": []}'}}]}
        response = urllib.response.addinfourl(
            io.BytesIO(json.dumps(payload).encode()),
            headers,
            request.full_url,
            self.status,
        )
        response.msg = "OK" if self.status == 200 else "Found"
        return response

    https_open = http_open


class ModelRoutingTests(unittest.TestCase):
    def call(self, **overrides):
        options = {
            "base_url": "https://api.openai.com/v1",
            "api_key": None,
            "model": "gpt-5",
            "messages": build_prompt("synthetic owner text", 1),
            "timeout": 1,
        }
        options.update(overrides)
        return call_model(**options)

    def test_default_private_route_denies_remote_and_loopback_gateways(self):
        with patch("urllib.request.Request") as request, patch(
            "urllib.request.build_opener"
        ) as build_opener:
            for base_url in (
                "https://api.openai.com/v1",
                "http://127.0.0.1:8000/v1",
                "http://[::1]:8000/v1",
            ):
                with self.subTest(base_url=base_url), self.assertRaisesRegex(
                    ValueError, "PRIV2: private corpus"
                ):
                    self.call(base_url=base_url)
            request.assert_not_called()
            build_opener.assert_not_called()

    def test_private_provider_labels_cannot_bypass_endpoint_checks(self):
        cases = [
            {"provider": "local", "base_url": "https://api.openai.com/v1"},
            {"provider": "local", "base_url": "http://localhost:8000/v1"},
            {"provider": "local", "base_url": "http://127.0.0.1.example/v1"},
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com.example/v1",
                "model": "claude-sonnet-4-5",
            },
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com:444/v1",
                "model": "claude-sonnet-4-5",
            },
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/relay",
                "model": "claude-sonnet-4-5",
            },
            {
                "provider": "anthropic",
                "base_url": "http://api.anthropic.com/v1",
                "model": "claude-sonnet-4-5",
            },
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/v1",
                "model": "gpt-5",
            },
            {
                "provider": "openai-compatible",
                "base_url": "https://api.anthropic.com/v1",
                "model": "claude-sonnet-4-5",
            },
        ]
        with patch("urllib.request.Request") as request, patch(
            "urllib.request.build_opener"
        ) as build_opener:
            for options in cases:
                with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, "PRIV2:"
                ):
                    self.call(**options)
            request.assert_not_called()
            build_opener.assert_not_called()

    def test_unknown_classifications_and_invalid_urls_fail_closed(self):
        cases = [
            {"corpus_tier": "filtered"},
            {"corpus_tier": "_pub"},
            {"corpus_tier": None},
            {"provider": "unknown"},
            {"base_url": "file:///tmp/model"},
            {"base_url": "https://"},
            {"base_url": "https://api.anthropic.com@relay.example/v1"},
            {"base_url": "https://api.anthropic.com:invalid/v1"},
            {"base_url": "https://api.anthropic.com/v1?route=other"},
            {"base_url": "https://api.anthropic.com/v1#other"},
            {"base_url": "https://api.anthropic.com\n/v1"},
            {
                "base_url": "http://public.example/v1",
                "corpus_tier": "public",
            },
        ]
        with patch("urllib.request.Request") as request, patch(
            "urllib.request.build_opener"
        ) as build_opener:
            for options in cases:
                with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, "PRIV2:"
                ):
                    self.call(**options)
            request.assert_not_called()
            build_opener.assert_not_called()

    @patch("urllib.request.OpenerDirector.open")
    def test_explicit_local_and_direct_claude_routes_accept_private_data(
        self, open_request
    ):
        routes = [
            ("local", "http://127.0.0.1:8000/v1", "local-model"),
            ("local", "http://[::1]:8000/v1", "local-model"),
            (
                "anthropic",
                "https://api.anthropic.com/v1/",
                "claude-sonnet-4-5",
            ),
        ]
        for provider, base_url, model in routes:
            with self.subTest(provider=provider, base_url=base_url):
                response = {"choices": [{"message": {"content": '{"skills": []}'}}]}
                open_request.return_value = FakeResponse(json.dumps(response).encode())
                self.assertEqual(
                    self.call(
                        base_url=base_url,
                        model=model,
                        provider=provider,
                        api_key="synthetic-test-key",
                    ),
                    '{"skills": []}',
                )
                request = open_request.call_args.args[0]
                self.assertEqual(request.full_url, chat_completions_url(base_url))
                payload = json.loads(request.data)
                self.assertEqual(payload["model"], model)
                self.assertEqual(
                    payload["messages"], build_prompt("synthetic owner text", 1)
                )
                self.assertEqual(
                    request.get_header("Authorization"), "Bearer synthetic-test-key"
                )
                if provider == "anthropic":
                    self.assertNotIn("temperature", payload)

    @patch("urllib.request.OpenerDirector.open")
    def test_explicit_public_tier_allows_compatible_remote_and_relay_routes(
        self, open_request
    ):
        for base_url in (
            "https://api.openai.com/v1",
            "http://127.0.0.1:8000/v1",
        ):
            with self.subTest(base_url=base_url):
                response = {"choices": [{"message": {"content": '{"skills": []}'}}]}
                open_request.return_value = FakeResponse(json.dumps(response).encode())
                self.assertEqual(
                    self.call(
                        base_url=base_url,
                        corpus_tier="public",
                        messages=build_prompt('{"aggregate_count":3}', 1),
                    ),
                    '{"skills": []}',
                )
                request = open_request.call_args.args[0]
                self.assertEqual(request.full_url, chat_completions_url(base_url))
                self.assertIn("aggregate_count", json.loads(request.data)["messages"][1]["content"])

    def test_cli_does_not_treat_filtered_or_pub_filenames_as_public(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {
                "source": "hermes",
                "messages": [{"role": "user", "content": "synthetic owner text"}],
            }
            for filename in ("filtered_data.jsonl", "corpus_pub.jsonl"):
                with self.subTest(filename=filename):
                    corpus = root / filename
                    corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
                    output = root / "skills"
                    with patch("urllib.request.OpenerDirector.open") as open_request, patch(
                        "corpus_to_skills.discover_jsonl"
                    ) as discover, patch("sys.stderr", new_callable=io.StringIO) as stderr:
                        result = main(
                            [
                                str(corpus),
                                "--model", "gpt-5",
                                "--base-url", "https://api.openai.com/v1",
                                "--output-dir", str(output),
                            ]
                        )
                    self.assertEqual(result, 1)
                    self.assertIn("PRIV2:", stderr.getvalue())
                    discover.assert_not_called()
                    open_request.assert_not_called()
                    self.assertFalse(output.exists())

    def test_cli_passes_explicit_provider_and_tier_to_the_model_gate(self):
        routes = [
            ("local", "private", "http://127.0.0.1:8000/v1", "local-model"),
            ("openai-compatible", "public", "https://api.openai.com/v1", "gpt-5"),
        ]
        for provider, tier, base_url, model in routes:
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus = root / "corpus.jsonl"
                corpus.write_text('{"aggregate_count":3}\n', encoding="utf-8")
                output = root / "skills"
                skill_result = {
                    "skills": [{
                        "name": "check-results",
                        "description": "Check aggregate results when comparing runs.",
                        "body": "Compare the recorded counts.",
                    }]
                }
                response = {
                    "choices": [{"message": {"content": json.dumps(skill_result)}}]
                }
                with patch("urllib.request.OpenerDirector.open") as open_request, patch(
                    "sys.stderr", new_callable=io.StringIO
                ), patch("sys.stdout", new_callable=io.StringIO):
                    open_request.return_value = FakeResponse(json.dumps(response).encode())
                    result = main(
                        [
                            str(corpus),
                            "--model", model,
                            "--base-url", base_url,
                            "--provider", provider,
                            "--corpus-tier", tier,
                            "--skills", "1",
                            "--output-dir", str(output),
                        ]
                    )
                self.assertEqual(result, 0)
                open_request.assert_called_once()
                self.assertTrue((output / "check-results" / "SKILL.md").is_file())

    def test_environment_proxies_cannot_redirect_private_requests(self):
        transport = RecordingTransport()
        real_build_opener = urllib.request.build_opener
        proxies = {
            "http_proxy": "http://proxy.example:3128",
            "https_proxy": "http://proxy.example:3128",
            "HTTP_PROXY": "http://proxy.example:3128",
            "HTTPS_PROXY": "http://proxy.example:3128",
            "no_proxy": "",
            "NO_PROXY": "",
        }

        def build_opener(*handlers):
            return real_build_opener(*handlers, transport)

        with patch.dict(os.environ, proxies), patch(
            "urllib.request.build_opener", side_effect=build_opener
        ), patch("urllib.request.urlopen", side_effect=AssertionError("unexpected transport")):
            self.assertEqual(
                self.call(
                    base_url="https://api.anthropic.com/v1",
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                ),
                '{"skills": []}',
            )
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(transport.requests[0].host, "api.anthropic.com")
        self.assertEqual(
            transport.requests[0].full_url,
            "https://api.anthropic.com/v1/chat/completions",
        )

    def test_model_redirects_do_not_forward_private_requests(self):
        transport = RecordingTransport(
            status=302, location="https://other.example/v1/chat/completions"
        )
        real_build_opener = urllib.request.build_opener

        def build_opener(*handlers):
            return real_build_opener(*handlers, transport)

        with patch("urllib.request.build_opener", side_effect=build_opener), patch(
            "urllib.request.urlopen", side_effect=AssertionError("unexpected transport")
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                self.call(
                    base_url="http://127.0.0.1:8000/v1",
                    provider="local",
                    model="local-model",
                )
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(transport.requests[0].host, "127.0.0.1:8000")

    @patch("urllib.request.OpenerDirector.open")
    def test_http_errors_do_not_echo_private_response_bodies(self, open_request):
        open_request.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8000/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b"synthetic owner text"),
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 400") as raised:
            self.call(
                base_url="http://127.0.0.1:8000/v1",
                provider="local",
                model="local-model",
            )
        self.assertNotIn("synthetic owner text", str(raised.exception))


class SkillOutputTests(unittest.TestCase):
    def test_parses_fenced_json_and_validates_skills(self):
        content = """```json
{"skills":[{"name":"review-python","description":"Review Python when asked for correctness checks.","body":"# Workflow\\n\\nInspect behavior first."}]}
```"""
        skills = validate_skills(parse_json_response(content), expected_count=1)
        self.assertEqual(skills[0].name, "review-python")

    def test_rejects_invalid_skill_name(self):
        payload = {
            "skills": [
                {"name": "Bad Name", "description": "Useful trigger", "body": "Act."}
            ]
        }
        with self.assertRaisesRegex(ValueError, "invalid Agent Skills name"):
            validate_skills(payload, expected_count=1)

    def test_writes_spec_compliant_skill_and_refuses_overwrite(self):
        skill = GeneratedSkill(
            "review-python",
            "Review Python when asked for correctness checks.",
            "# Workflow\n\nInspect behavior first.",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = write_skills([skill], output_dir=output, overwrite=False)
            text = paths[0].read_text(encoding="utf-8")
            self.assertIn("name: review-python", text)
            self.assertIn('description: "Review Python', text)
            self.assertTrue(text.endswith("\n"))
            with self.assertRaises(FileExistsError):
                write_skills([skill], output_dir=output, overwrite=False)


if __name__ == "__main__":
    unittest.main()

