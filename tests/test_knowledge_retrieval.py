"""Hybrid retrieval respects authorized event sets and source revisions."""

import os
import uuid
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock
from sqlalchemy import select, update
from persistence.repository import Repository, PERSONAL
from knowledge.store import build
from knowledge import schema as k
from knowledge.retrieval import index, retrieve, cosine
from knowledge_support import sample


class CosineTests(unittest.TestCase):
    def test_zero_and_dimension_mismatch(self):
        self.assertEqual(cosine([0, 0], [1, 2]), 0)
        self.assertEqual(cosine([1], [1, 2]), 0)
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1)


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL"), "requires disposable PostgreSQL"
)
class RetrievalTests(unittest.TestCase):
    def test_filtering_delete_and_stale_projection(self):
        with TemporaryDirectory() as root:
            repo = Repository(os.environ["TEST_DATABASE_URL"], Path(root) / "media")
            self.addCleanup(repo.engine.dispose)
            session = build(repo, sample(repo, root))
            embedder = Mock(
                model="fixture-encoder",
                embed_batch=AsyncMock(
                    side_effect=lambda texts: [[1, 0] for _ in texts]
                ),
                embed_text=AsyncMock(return_value=[1, 0]),
            )
            index(repo, embedder)
            with repo.engine.connect() as c:
                hits, coverage = retrieve(
                    c, PERSONAL, [session], "预算", embedder=embedder
                )
                self.assertTrue(hits)
                self.assertTrue(coverage["semantic"])
                empty, _ = retrieve(
                    c, uuid.uuid4(), [session], "预算", embedder=embedder
                )
                self.assertFalse(empty)
            with repo.engine.begin() as c:
                c.execute(
                    update(k.sessions)
                    .where(k.sessions.c.id == session)
                    .values(revision=99)
                )
            with repo.engine.connect() as c:
                stale, _ = retrieve(c, PERSONAL, [session], "预算", embedder=embedder)
                self.assertFalse(stale)
