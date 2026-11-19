"""Facts are idempotent and unconfirmed metadata never becomes a confirmed roster."""

import os
import unittest
import uuid
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from sqlalchemy import select, func
from persistence.repository import Repository, PERSONAL
from knowledge import schema as k
from knowledge.store import build, confirm, owned, backfill, EXTRACTOR
from knowledge_support import sample


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL"), "requires disposable PostgreSQL"
)
class KnowledgeStoreTests(unittest.TestCase):
    def test_build_and_correction_are_scoped_and_keep_provenance(self):
        with TemporaryDirectory() as root:
            repo = Repository(os.environ["TEST_DATABASE_URL"], Path(root) / "media")
            self.addCleanup(repo.engine.dispose)
            identifier = sample(repo, root)
            session = build(repo, identifier)
            self.assertEqual(build(repo, identifier), session)
            with repo.engine.connect() as c:
                row = owned(c, PERSONAL, session)
                self.assertEqual(row["time_status"], "candidate")
                self.assertEqual(row["event_date"], date(2026, 11, 10))
                self.assertFalse(
                    c.scalar(
                        select(func.count())
                        .select_from(k.participants)
                        .where(
                            k.participants.c.session_id == session,
                            k.participants.c.status == "confirmed",
                        )
                    )
                )
                with self.assertRaises(LookupError):
                    owned(c, uuid.uuid4(), session)
            confirm(
                repo,
                PERSONAL,
                session,
                event_date=date(2026, 11, 10),
                names=["张三", "李四", "张三"],
            )
            with repo.engine.connect() as c:
                self.assertEqual(
                    owned(c, PERSONAL, session)["time_status"], "confirmed"
                )
                self.assertEqual(
                    c.scalar(
                        select(func.count())
                        .select_from(k.participants)
                        .where(
                            k.participants.c.session_id == session,
                            k.participants.c.active.is_(True),
                        )
                    ),
                    2,
                )
                self.assertGreater(
                    c.scalar(
                        select(func.count())
                        .select_from(k.evidence)
                        .where(k.evidence.c.session_id == session)
                    ),
                    2,
                )

    def test_failed_build_does_not_block_next_and_cooldown_is_durable(self):
        from unittest.mock import patch
        from sqlalchemy import update
        from persistence.schema import analyses

        with TemporaryDirectory() as root:
            repo = Repository(os.environ["TEST_DATABASE_URL"], Path(root) / "media")
            self.addCleanup(repo.engine.dispose)
            bad = sample(repo, root)
            good = sample(repo, root)
            original = build

            def failing(repository, identifier):
                if identifier == bad:
                    raise ValueError("do not expose content")
                return original(repository, identifier)

            with patch("knowledge.store.build", side_effect=failing):
                self.assertGreaterEqual(backfill(repo), 1)
            with repo.engine.connect() as c:
                state = (
                    c.execute(
                        select(k.builds).where(
                            k.builds.c.analysis_id == bad,
                            k.builds.c.extractor_version == EXTRACTOR,
                        )
                    )
                    .mappings()
                    .one()
                )
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["error_code"], "ValueError")
                self.assertEqual(
                    c.scalar(
                        select(k.builds.c.status).where(k.builds.c.analysis_id == good)
                    ),
                    "completed",
                )
            with patch("knowledge.store.build") as mock:
                backfill(repo)
                self.assertNotIn(bad, [call.args[1] for call in mock.call_args_list])

    def test_real_topics_schema_null_fields_and_reanalysis_preserve_confirmations(self):
        from sqlalchemy import update
        from persistence.schema import analyses
        from datetime import datetime

        with TemporaryDirectory() as root:
            repo = Repository(os.environ["TEST_DATABASE_URL"], Path(root) / "media")
            self.addCleanup(repo.engine.dispose)
            identifier = sample(repo, root)
            with repo.engine.begin() as c:
                c.execute(
                    update(analyses)
                    .where(analyses.c.id == identifier)
                    .values(
                        result_json={
                            "participants": None,
                            "topics": [{"title": "真实议题", "summary": None}],
                            "keywords": None,
                        }
                    )
                )
            session = build(repo, identifier)
            start = datetime.fromisoformat("2026-11-10T23:50:00+08:00")
            confirm(repo, PERSONAL, session, event_start_at=start, names=["人工确认者"])
            # Force projection rebuild just as a new extractor would do.
            with repo.engine.begin() as c:
                c.execute(k.builds.delete().where(k.builds.c.analysis_id == identifier))
            build(repo, identifier)
            with repo.engine.connect() as c:
                event = owned(c, PERSONAL, session)
                self.assertEqual(event["event_start_at"], start)
                self.assertEqual(event["precision"], "second")
                rows = (
                    c.execute(
                        select(k.participants).where(
                            k.participants.c.session_id == session,
                            k.participants.c.active.is_(True),
                        )
                    )
                    .mappings()
                    .all()
                )
                self.assertIn("人工确认者", [r["display_name"] for r in rows])
                self.assertEqual(
                    c.scalar(
                        select(k.topics.c.title).where(
                            k.topics.c.session_id == session,
                            k.topics.c.active.is_(True),
                        )
                    ),
                    "真实议题",
                )
