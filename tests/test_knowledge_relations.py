"""Constrained plans reject SQL and graph edges retain scoped evidence."""

import os
import unittest
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from pydantic import ValidationError
from sqlalchemy import select
from knowledge.aggregate import AggregatePlan, execute
from knowledge.relations import RelationInput, add, traverse, alias, remove_alias
from knowledge import schema as k
from knowledge.store import build, confirm
from persistence.repository import Repository, PERSONAL
from knowledge_support import sample


class AggregateGuardTests(unittest.TestCase):
    def test_untrusted_fields_and_dimensions_rejected(self):
        for value in [
            {"sql": "SELECT pg_sleep(1000)"},
            {"workspace_id": str(uuid.uuid4())},
            {"group_by": "pg_catalog.pg_authid"},
            {"top_n": 1000},
            {"metric": "DROP TABLE people"},
        ]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AggregatePlan(start=date(2026, 1, 1), stop=date(2026, 2, 1), **value)

    def test_invalid_intervals_rejected(self):
        with self.assertRaises(ValidationError):
            AggregatePlan(start=date(2026, 2, 1), stop=date(2026, 1, 1))


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL"), "requires disposable PostgreSQL"
)
class RelationTests(unittest.TestCase):
    def test_graph_scope_two_hops_and_alias_undo(self):
        with TemporaryDirectory() as root:
            repo = Repository(os.environ["TEST_DATABASE_URL"], Path(root) / "archive")
            self.addCleanup(repo.engine.dispose)
            event = build(repo, sample(repo, root))
            confirm(
                repo, PERSONAL, event, event_date=date(2039, 1, 2), names=["关系验收甲"]
            )
            with repo.engine.connect() as c:
                proof = c.scalar(
                    select(k.evidence.c.id).where(k.evidence.c.session_id == event)
                )
                person = c.scalar(
                    select(k.participants.c.person_id).where(
                        k.participants.c.session_id == event,
                        k.participants.c.active.is_(True),
                    )
                )
            for source, target, kind, source_kind, target_kind in [
                ("项目甲", "决定甲", "decides", "project", "decision"),
                ("决定甲", "行动甲", "follows_up", "decision", "action"),
            ]:
                add(
                    repo,
                    PERSONAL,
                    RelationInput(
                        session_id=event,
                        evidence_id=proof,
                        source_name=source,
                        target_name=target,
                        source_kind=source_kind,
                        target_kind=target_kind,
                        kind=kind,
                    ),
                )
            with repo.engine.connect() as c:
                self.assertEqual(
                    len(traverse(c, PERSONAL, ["项目甲"], [event], hops=2)["edges"]), 2
                )
                self.assertFalse(
                    traverse(c, uuid.uuid4(), ["项目甲"], [event])["edges"]
                )
                self.assertFalse(traverse(c, PERSONAL, ["项目甲"], [])["edges"])
                rows = execute(
                    c,
                    PERSONAL,
                    AggregatePlan(start=date(2039, 1, 2), stop=date(2039, 1, 3)),
                )
                self.assertTrue(any(row["id"] == person for row in rows))
            alias(repo, PERSONAL, person, "甲别名")
            remove_alias(repo, PERSONAL, person, "甲别名")
            with repo.engine.connect() as c:
                self.assertIsNone(
                    c.scalar(
                        select(k.aliases.c.id).where(k.aliases.c.person_id == person)
                    )
                )
