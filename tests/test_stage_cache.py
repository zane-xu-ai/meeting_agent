"""Exercise cache policy snapshots, concurrent deduplication and invalidation."""

import asyncio
import os
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from persistence.repository import Repository
from persistence.stage_cache import invalidate, list_cache, memoize, set_policy


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "requires disposable PostgreSQL")
class StageCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_computations_coalesce_and_policy_is_snapshotted(self):
        with TemporaryDirectory() as root:
            repo = Repository(os.environ["TEST_DATABASE_URL"], Path(root))
            self.addCleanup(repo.engine.dispose)
            key = uuid.uuid4().hex

            def new_job():
                identifier = uuid.uuid4().hex
                repo.create({"task_id": identifier, "source_type": "legacy", "status": "pending"}, None, {})
                return identifier

            calls = 0

            async def compute():
                nonlocal calls
                calls += 1
                await asyncio.sleep(.05)
                return {"text": f"attempt {calls}"}

            set_policy(repo, "asr", True, True, 600)
            self.addCleanup(set_policy, repo, "asr", True, True, 90 * 86400)
            first, second = new_job(), new_job()
            actual = await asyncio.gather(
                memoize(repo, first, "asr", key, compute),
                memoize(repo, second, "asr", key, compute),
            )
            self.assertEqual(actual[0], actual[1])
            self.assertEqual(calls, 1)

            set_policy(repo, "asr", False, False, 600)
            bypass = new_job()
            self.assertEqual(await memoize(repo, bypass, "asr", key, compute), {"text": "attempt 2"})
            self.assertEqual(await memoize(repo, first, "asr", key, compute), {"text": "attempt 1"})

            entry = list_cache(repo, "asr")["entries"][0]
            self.assertGreaterEqual(entry["hit_count"], 2)
            self.assertEqual(invalidate(repo, identifier=entry["id"]), 1)
            self.assertEqual(await memoize(repo, first, "asr", key, compute), {"text": "attempt 3"})
            self.assertEqual(repo.task(first)["status"], "pending")
