"""GPU admission and alert rules on a disposable scheduler database."""

import os
import unittest
import uuid
from datetime import timedelta

from sqlalchemy import update

from gpu_ops.scheduler import Memory, Scheduler, now
from gpu_ops import schema


@unittest.skipUnless(os.environ.get("TEST_GPU_DATABASE_URL"), "requires a disposable GPU scheduler database")
class GpuTests(unittest.TestCase):
    def setUp(self):
        self.identifier = f"GPU-test-{uuid.uuid4()}"
        self.profile = f"profile-{uuid.uuid4()}"
        self.pids = (111, 222)
        self.free = 3000
        self.scheduler = Scheduler(
            os.environ["TEST_GPU_DATABASE_URL"], self.identifier,
            sensor=lambda _: Memory(self.identifier, 8192, self.free, self.pids),
        )
        self.addCleanup(self.scheduler.engine.dispose)
        self.scheduler.initialize()
        self.scheduler.configure(self.profile, 3000, 1024)

    def test_wait_alert_and_recovery(self):
        job = uuid.uuid4()
        self.addCleanup(self.scheduler.cancel, job)
        self.assertEqual(self.scheduler.acquire(job, "rag", self.profile), (None, "external_memory"))
        with self.scheduler.engine.begin() as connection:
            connection.execute(update(schema.requests).where(schema.requests.c.id == job)
                               .values(created_at=now() - timedelta(minutes=6)))
        self.assertEqual(self.scheduler.observe(), ("warning", "external_wait"))
        self.assertIsNone(self.scheduler.observe())
        self.free = 7000
        lease, reason = self.scheduler.acquire(job, "rag", self.profile)
        self.assertIsNone(reason)
        self.assertTrue(self.scheduler.heartbeat(lease))
        self.assertTrue(self.scheduler.release(lease))
        self.assertEqual(self.scheduler.observe(), ("ok", None))

    def test_one_slot_and_fencing(self):
        self.free = 7000
        first, second = uuid.uuid4(), uuid.uuid4()
        self.addCleanup(self.scheduler.cancel, second)
        lease, _ = self.scheduler.acquire(first, "meeting", self.profile)
        self.assertIsNotNone(lease)
        self.assertEqual(self.scheduler.acquire(second, "rag", self.profile), (None, "managed_busy"))
        self.assertTrue(self.scheduler.release(lease))
        replacement, _ = self.scheduler.acquire(second, "rag", self.profile)
        self.assertIsNotNone(replacement)
        self.assertFalse(self.scheduler.release(lease))
        self.assertTrue(self.scheduler.heartbeat(replacement))
        self.scheduler.release(replacement)


class GpuPolicyTests(unittest.TestCase):
    def test_external_occupant_blocks_exclusive_profile_even_with_free_memory(self):
        sample = Memory("GPU-test", 8192, 7000, (999,))
        self.assertFalse(Scheduler._can_run({"memory_mib": 1024, "margin_mib": 1024, "exclusive": True}, sample))
        self.assertTrue(Scheduler._can_run({"memory_mib": 1024, "margin_mib": 1024, "exclusive": False}, sample))


@unittest.skipUnless(
    all(os.environ.get(name) for name in ("TEST_GPU_DATABASE_URL", "TEST_DATABASE_URL", "TEST_AMQP_URL",
                                        "TEST_LOCAL_ASR_PYTHON", "TEST_LOCAL_ASR_MODEL", "TEST_LOCAL_ASR_SAMPLE")),
    "requires CUDA, the synthetic fixture, PostgreSQL and RabbitMQ",
)
class LiveGpuTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_whisper_runs_behind_rabbit_and_releases_card(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import AsyncMock, patch

        import aio_pika
        from config import settings
        from persistence import broker, dispatch
        from persistence.repository import Repository

        gpu_dsn = os.environ["TEST_GPU_DATABASE_URL"]
        task_db = os.environ["TEST_DATABASE_URL"]
        scheduler = Scheduler(gpu_dsn)
        self.addCleanup(scheduler.engine.dispose)
        scheduler.initialize()
        scheduler.configure("whisper-small", 3072, 1024)
        if scheduler.snapshot()["free_mib"] < 4096:
            self.skipTest("the live GPU is below the approved test budget")
        scheduler.configure("blocked-verification", 8192, 1024)
        waiting = uuid.uuid4()
        self.addCleanup(scheduler.cancel, waiting)
        lease, reason = scheduler.acquire(waiting, "meeting-agent-verification", "blocked-verification")
        self.assertIsNone(lease)
        self.assertIn(reason, ("external_memory", "capacity"))
        scheduler.cancel(waiting)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = Repository(task_db, root / "archive")
            self.addCleanup(repo.engine.dispose)
            task = uuid.uuid4().hex
            repo.create({"task_id": task, "status": "pending", "source_type": "local_audio",
                         "filename": "synthetic.wav"}, os.environ["TEST_LOCAL_ASR_SAMPLE"],
                        {"asr_model": "local-whisper"}, queued=True)
            with (
                patch.object(settings, "database_url", task_db),
                patch.object(settings, "gpu_database_url", gpu_dsn),
                patch.object(settings, "amqp_url", os.environ["TEST_AMQP_URL"]),
                patch.object(settings, "local_asr_python", Path(os.environ["TEST_LOCAL_ASR_PYTHON"])),
                patch.object(settings, "local_asr_model", Path(os.environ["TEST_LOCAL_ASR_MODEL"])),
                patch.object(settings, "media_storage_dir", root / "archive"),
                patch.object(settings, "output_dir", root / "outputs"),
                patch("agent.asr_client.ASRClient.transcribe", new=AsyncMock(side_effect=AssertionError("no cloud ASR"))),
                patch("agent.graph.agent_loop", new=AsyncMock(return_value={
                    "rec_type": "general", "output_text": "# 合成音频验收",
                })),
            ):
                async with await aio_pika.connect_robust(os.environ["TEST_AMQP_URL"]) as connection:
                    exchange, queues = await broker.topology(connection)
                    for route in ("io", "cpu", "gpu", "cpu", "cloud", "cpu"):
                        await broker.publish_pending(repo, exchange)
                        await broker.handle(repo, await queues[route].get(timeout=5), route)
                result = repo.task(task)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["asr_model_used"], "local-whisper")
                self.assertTrue(repo.content(task, "asr"))
                self.assertIsNone(scheduler.snapshot()["lease"])
