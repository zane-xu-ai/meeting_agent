"""Knowledge entry points keep authorization and the feature rollback switch."""

import unittest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
import server


class KnowledgeAPITests(unittest.TestCase):
    def test_external_clients_cannot_read_query_data(self):
        with (
            patch.object(server.settings, "workspace_token", "test-secret"),
            TestClient(server.app) as client,
        ):
            self.assertEqual(client.get("/api/knowledge/sessions").status_code, 401)
            self.assertEqual(
                client.post(
                    "/api/knowledge/query", json={"question": "上周会议"}
                ).status_code,
                401,
            )

    def test_switch_disables_knowledge_without_model_calls(self):
        with (
            patch.object(server.settings, "workspace_token", ""),
            patch.object(server.settings, "knowledge_enabled", False),
            patch("agent.llm_client.LLMClient") as model,
            TestClient(
                server.app, base_url="http://localhost", client=("127.0.0.1", 1234)
            ) as client,
        ):
            self.assertEqual(client.get("/api/knowledge/coverage").status_code, 503)
            self.assertEqual(
                client.post("/api/global-chat", json={"message": "上周会议"}).json()[
                    "status"
                ],
                "no_match",
            )
            model.assert_not_called()

    @unittest.skipUnless(
        os.environ.get("TEST_DATABASE_URL"), "requires disposable PostgreSQL"
    )
    def test_live_coverage_includes_rebuild_and_projection_watermarks(self):
        import os
        from pathlib import Path
        from persistence.repository import Repository

        repo = Repository(
            os.environ["TEST_DATABASE_URL"], Path("/tmp/knowledge-api-fixture")
        )
        self.addCleanup(repo.engine.dispose)
        with (
            patch("knowledge.api.repository", return_value=repo),
            patch.object(server.settings, "workspace_token", ""),
            TestClient(
                server.app, base_url="http://localhost", client=("127.0.0.1", 1234)
            ) as client,
        ):
            result = client.get("/api/knowledge/coverage")
            self.assertEqual(result.status_code, 200)
            data = result.json()
            self.assertLessEqual(data["embedded_chunks"], data["current_chunks"])
            self.assertGreaterEqual(data["completed_builds"], 0)
            self.assertGreaterEqual(data["failed_builds"], 0)
            metrics = client.get("/api/knowledge/metrics")
            self.assertEqual(metrics.status_code, 200)
            self.assertIn("meeting_knowledge_revision", metrics.text)
