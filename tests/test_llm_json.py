"""Verify JSON mode, bounded truncation retries and model fallback."""

import json
import unittest
from unittest.mock import patch

import httpx

from agent.llm_client import LLMClient, LLMJSONError


def completion(content, finish_reason="stop"):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
    })


class JSONChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        async_client = httpx.AsyncClient

        def make_client(**kwargs):
            return async_client(transport=httpx.MockTransport(self.respond), **kwargs)

        mock = patch("agent.llm_client.httpx.AsyncClient", side_effect=make_client)
        mock.start()
        self.addCleanup(mock.stop)

    def respond(self, request):
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0)

    async def test_truncation_retries_same_model_before_moving_to_next_model(self):
        self.responses = [
            httpx.Response(403),
            completion(None, "length"),
            completion('{"questions": [', "length"),
            completion('{"questions": []}'),
        ]
        client = LLMClient(
            api_key="test",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            models=["unavailable", "glm-5.3", "backup"],
        )

        self.assertEqual(await client.chat_json("提取问题"), {"questions": []})
        self.assertEqual(
            [(request["model"], request["max_tokens"]) for request in self.requests],
            [("unavailable", 4096), ("glm-5.3", 4096), ("glm-5.3", 8192),
             ("glm-5.3", 16384)],
        )
        self.assertEqual(self.requests[1]["response_format"], {"type": "json_object"})
        self.assertEqual(self.requests[1]["reasoning_effort"], "low")

    async def test_malformed_finished_response_moves_to_next_model(self):
        self.responses = [completion("[]"), completion('```json\n{"ok": true}\n```')]
        client = LLMClient(api_key="test", base_url="https://example.test/v1", models=["a", "b"])

        self.assertEqual(await client.chat_json("给出结果"), {"ok": True})
        self.assertEqual([request["model"] for request in self.requests], ["a", "b"])
        self.assertEqual([request["max_tokens"] for request in self.requests], [4096, 4096])

    async def test_unsupported_json_mode_retries_with_prompt_only(self):
        self.responses = [
            httpx.Response(400, json={"error": {"message": "unsupported response_format"}}),
            completion('{"ok": true}'),
        ]
        client = LLMClient(api_key="test", base_url="https://example.test/v1", models=["a"])

        self.assertEqual(await client.chat_json("给出结果"), {"ok": True})
        self.assertIn("response_format", self.requests[0])
        self.assertNotIn("response_format", self.requests[1])

    async def test_truncated_response_is_rejected_at_token_ceiling(self):
        self.responses = [completion('{"partial": true}', "length")]
        client = LLMClient(api_key="test", base_url="https://example.test/v1", models=["a"])

        with self.assertRaises(LLMJSONError) as error:
            await client.chat_json("给出结果", max_tokens=16384)
        self.assertTrue(error.exception.truncated)
        self.assertEqual(len(self.requests), 1)

    async def test_regular_chat_does_not_request_json_mode(self):
        self.responses = [completion("普通回复")]
        client = LLMClient(api_key="test", base_url="https://example.test/v1", models=["a"])

        self.assertEqual(await client.chat("你好"), "普通回复")
        self.assertNotIn("response_format", self.requests[0])

    async def test_auto_mode_uses_new_order_and_stops_at_first_success(self):
        from config import DEFAULT_LLM_MODELS

        expected = [
            'qwen3.8-flash', 'qwen3.8-max-0902', 'deepseek-v4.1-flash',
            'deepseek-v4.1-flash-1017', 'kimi-k3.1-pro', 'qwen4-max',
        ]
        self.assertEqual(DEFAULT_LLM_MODELS, expected)
        for failures in (0, 3, 4, 5):
            with self.subTest(failures=failures):
                self.requests = []
                self.responses = [httpx.Response(403) for _ in range(failures)] + [
                    completion('{"ok":true}')
                ]
                with patch('agent.llm_client.settings.llm_models', DEFAULT_LLM_MODELS):
                    client = LLMClient(api_key='test', base_url='https://example.test/v1')
                    self.assertEqual(await client.chat_json('回退顺序验收'), {'ok': True})
                self.assertEqual([r['model'] for r in self.requests], expected[:failures + 1])
                self.assertEqual(client.model, expected[failures])
