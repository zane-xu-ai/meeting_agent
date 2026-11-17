"""Integration coverage using PostgreSQL and a disposable RabbitMQ vhost."""

import os
import uuid
import unittest
import wave
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import aio_pika
import httpx
from sqlalchemy import select, update

import server
from agent.asr_client import ASRClient
from models.schemas import Sentence, Transcript
from persistence import broker, dispatch
from persistence.queue_schema import outbox
from persistence.repository import Repository
from utils.media_network import parse_media_url


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL") and os.environ.get("TEST_AMQP_URL"),
                     "requires PostgreSQL and a disposable RabbitMQ vhost")
class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = Repository(os.environ["TEST_DATABASE_URL"], self.root / "media")
        self.addCleanup(self.repo.engine.dispose)

    async def test_expired_lease_fences_old_worker_and_cancel_needs_no_broker(self):
        task = uuid.uuid4().hex
        self.repo.create({"task_id": task, "status": "pending", "source_type": "url_audio"},
                         "https://example.com/sample.wav", {}, queued=True)
        with self.repo.engine.connect() as connection:
            delivery = connection.scalar(select(outbox.c.id).where(outbox.c.job_id == task))
        first = dispatch.claim(self.repo, delivery)
        self.assertIsNone(dispatch.claim(self.repo, delivery))
        with self.repo.engine.begin() as connection:
            connection.execute(update(outbox).where(outbox.c.id == delivery).values(
                lease_until=dispatch.utcnow() - timedelta(seconds=1),
            ))
        replacement = dispatch.claim(self.repo, delivery)
        self.assertNotEqual(first["lease_token"], replacement["lease_token"])
        self.assertFalse(dispatch.complete(self.repo, first, {}))
        self.assertTrue(dispatch.cancel(self.repo, task))
        self.assertTrue(dispatch.complete(self.repo, replacement, {}))
        self.assertEqual(self.repo.task(task)["status"], "cancelled")

    async def test_full_stage_flow_through_broker_and_duplicate_delivery(self):
        task = uuid.uuid4().hex
        sample = self.root / "voice.wav"
        with wave.open(str(sample), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0" * 6400)
        self.repo.create({"task_id": task, "status": "pending", "source_type": "local_audio",
                          "filename": "voice.wav", "progress": 0}, str(sample), {}, queued=True)
        sample.unlink()
        transcript = Transcript(sentences=[Sentence(text="合成会议发言。", begin_time=10, end_time=100)])
        with (
            patch.object(server.settings, "database_url", os.environ["TEST_DATABASE_URL"]),
            patch.object(server.settings, "media_storage_dir", self.root / "media"),
            patch.object(server.settings, "output_dir", self.root / "outputs"),
            patch.object(server.settings, "dashscope_api_key", "fixture"),
            patch.object(server.settings, "workspace_token", ""),
            patch.object(ASRClient, "transcribe", new=AsyncMock(return_value=transcript)),
            patch("agent.graph.agent_loop", new=AsyncMock(return_value={
                "rec_type": "meeting", "output_text": "# 合成纪要", "analysis_data": {"summary": "测试"},
            })),
        ):
            async with await aio_pika.connect_robust(os.environ["TEST_AMQP_URL"]) as connection:
                exchange, queues = await broker.topology(connection)
                for index, (_, route) in enumerate(dispatch.STEPS):
                    await broker.publish_pending(self.repo, exchange)
                    if index == 0:
                        async with await aio_pika.connect_robust(os.environ["TEST_AMQP_URL"]) as interrupted:
                            _, other_queues = await broker.topology(interrupted)
                            await other_queues[route].get(timeout=5)
                    message = await queues[route].get(timeout=5)
                    if index == 0:
                        self.assertTrue(message.redelivered)
                    body = message.body
                    await broker.handle(self.repo, message, route)
                    await exchange.publish(aio_pika.Message(body), routing_key=route, mandatory=True)
                    duplicate = await queues[route].get(timeout=5)
                    await broker.handle(self.repo, duplicate, route)
            self.assertEqual(self.repo.task(task)["status"], "completed")
            self.assertEqual(self.repo.content(task, "result"), "# 合成纪要")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234)),
                base_url="http://localhost",
            ) as client:
                response = await client.get(f"/api/stream/{task}")
                self.assertEqual(response.status_code, 200)
                self.assertIn("id: ", response.text)
                self.assertIn("合成纪要", response.text)

    async def test_cloud_retry_is_persisted_with_backoff(self):
        task = uuid.uuid4().hex
        self.repo.create({"task_id": task, "status": "pending", "source_type": "url_audio"},
                         "https://example.com/sample.wav", {}, queued=True)
        with self.repo.engine.connect() as connection:
            delivery = connection.scalar(select(outbox.c.id).where(outbox.c.job_id == task))
        work = dispatch.claim(self.repo, delivery)
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(429, request=request)
        dispatch.failed(self.repo, work, httpx.HTTPStatusError("busy", request=request, response=response))
        self.assertEqual(self.repo.task(task)["status"], "retrying")
        with self.repo.engine.connect() as connection:
            followup = connection.execute(select(outbox).where(
                outbox.c.job_id == task, outbox.c.attempt == 2,
            )).mappings().one()
        self.assertGreater(followup["available_at"], dispatch.utcnow())
        self.assertTrue(dispatch.cancel(self.repo, task))

    async def test_mixed_batch_idempotency_and_partial_rejection(self):
        key = uuid.uuid4().hex
        files = [("files", (f"test{i}.wav", f"tiny{i}".encode())) for i in range(3)]
        urls = "https://example.com/1.wav\nhttp://127.0.0.1/private\nhttps://example.com/3.wav"

        def validate(url):
            from utils.media_network import UnsafeMediaURL
            parsed = parse_media_url(url)
            if parsed.hostname == "127.0.0.1":
                raise UnsafeMediaURL("private")
            return parsed

        with (
            patch.object(server, "repository", return_value=self.repo),
            patch.object(server, "UPLOAD_DIR", self.root / "uploads"),
            patch.object(server.settings, "amqp_url", os.environ["TEST_AMQP_URL"]),
            patch.object(server.settings, "workspace_token", ""),
            patch.object(server.quota_manager, "whitelist", {"127.0.0.1"}),
            patch.object(server, "validate_media_url", side_effect=validate),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app, client=("127.0.0.1", 1234)),
                base_url="http://localhost",
            ) as client:
                first = await client.post("/api/batches", files=files, data={"urls": urls}, headers={"Idempotency-Key": key})
                self.assertEqual(first.status_code, 202)
                result = first.json()
                self.assertEqual(len(result["accepted"]), 5)
                self.assertEqual(len(result["rejected"]), 1)
                replay = await client.post("/api/batches", files=files, data={"urls": urls}, headers={"Idempotency-Key": key})
                self.assertEqual(replay.json(), result)
                self.assertEqual(len((await client.get("/api/batches/" + result["batch_id"])).json()["items"]), 5)
                for item in result["accepted"]:
                    self.assertEqual((await client.post(f"/api/status/{item['task_id']}/cancel")).status_code, 200)
