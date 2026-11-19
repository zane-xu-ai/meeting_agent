"""Private meeting facts must never be invented when retrieval is empty."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import server


class KnowledgeAuditTests(unittest.TestCase):
    def test_standard_questions_cover_eighty_cases(self):
        cases = json.loads(
            (Path(__file__).parent / "fixtures/meeting_queries.json").read_text()
        )
        self.assertEqual(len(cases), 80)
        self.assertEqual(len({item["question"] for item in cases}), 80)
        self.assertTrue(all(item["start"] < item["stop"] for item in cases))

    def test_no_evidence_avoids_model_generation_for_both_endpoints(self):
        with (
            patch.object(server.settings, "workspace_token", ""),
            patch.object(server.settings, "enable_vector_store", False),
            patch.object(
                server, "_retrieve_relevant_history", new=AsyncMock(return_value=[])
            ),
            patch("agent.llm_client.LLMClient") as model,
            TestClient(
                server.app, base_url="http://localhost", client=("127.0.0.1", 1234)
            ) as client,
        ):
            result = client.post(
                "/api/global-chat", json={"message": "上周有哪些人参会？"}
            )
            self.assertEqual(result.json()["status"], "no_match")
            stream = client.post(
                "/api/global-chat/stream", json={"message": "去年今天的主题？"}
            )
            self.assertIn("no_match", stream.text)
            model.assert_not_called()
