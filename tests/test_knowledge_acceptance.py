"""Compare SQL results with independent in-memory truth for all eighty questions."""

import json
import os
import uuid
import unittest
from datetime import date, datetime
from pathlib import Path
from sqlalchemy import insert
from persistence.repository import Repository
from persistence.schema import workspaces
from knowledge import schema as k
from knowledge.query import query
from knowledge.store import add_evidence


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL"), "requires disposable PostgreSQL"
)
class KnowledgeAcceptanceTests(unittest.TestCase):
    def test_eighty_questions_against_isolated_truth(self):
        repo = Repository(
            os.environ["TEST_DATABASE_URL"], Path("/tmp/unused-knowledge-fixture")
        )
        self.addCleanup(repo.engine.dispose)
        workspace = uuid.uuid4()
        dates = [
            date(2025, 11, 19),
            date(2026, 10, 15),
            date(2026, 11, 9),
            date(2026, 11, 10),
            date(2026, 11, 15),
            date(2026, 11, 16),
            date(2026, 11, 18),
            date(2026, 11, 19),
            date(2026, 11, 22),
            date(2026, 11, 23),
        ]
        ids = {day: uuid.uuid4() for day in dates}
        person = uuid.uuid4()
        with repo.engine.begin() as c:
            c.execute(
                insert(workspaces).values(id=workspace, name="synthetic-query-oracle")
            )
            c.execute(
                insert(k.people).values(
                    id=person, workspace_id=workspace, name="张三", active=True
                )
            )
            for day in dates:
                event = ids[day]
                c.execute(
                    insert(k.sessions).values(
                        id=event,
                        workspace_id=workspace,
                        identity_key=str(event),
                        title="合成会议",
                        category="meeting",
                        event_date=day,
                        timezone="Asia/Shanghai",
                        precision="day",
                        time_status="confirmed",
                        revision=1,
                        deleted=False,
                    )
                )
                evidence = add_evidence(
                    c,
                    workspace,
                    event,
                    "张三发言，讨论预算。",
                    kind="user_confirmation",
                )
                c.execute(
                    insert(k.participants),
                    [
                        {
                            "id": uuid.uuid4(),
                            "session_id": event,
                            "person_id": person,
                            "display_name": "张三",
                            "status": status,
                            "evidence_id": evidence,
                            "active": True,
                        }
                        for status in ("confirmed", "observed_speaker")
                    ],
                )
                c.execute(
                    insert(k.topics).values(
                        id=uuid.uuid4(),
                        session_id=event,
                        title="预算",
                        summary="讨论预算",
                        evidence_id=evidence,
                        active=True,
                    )
                )
                c.execute(
                    insert(k.chunks).values(
                        id=uuid.uuid4(),
                        workspace_id=workspace,
                        session_id=event,
                        evidence_id=evidence,
                        text="讨论预算",
                        source_revision=1,
                    )
                )
        cases = json.loads(
            (Path(__file__).parent / "fixtures/meeting_queries.json").read_text()
        )
        for case in cases:
            with self.subTest(question=case["question"]):
                expected = [
                    day
                    for day in dates
                    if date.fromisoformat(case["start"])
                    <= day
                    < date.fromisoformat(case["stop"])
                ]
                result = query(
                    repo,
                    workspace,
                    case["question"],
                    uuid.uuid4().hex,
                    anchor=datetime.fromisoformat(case["anchor"]),
                )
                self.assertEqual(result["coverage"]["scanned"], len(expected))
                self.assertEqual(
                    set(result["matched_ids"]), {str(ids[day]) for day in expected}
                )
                self.assertEqual(result["plan"]["intent"], case["intent"])
                if case["intent"] == "attendance_count":
                    self.assertEqual(result["claims"][0]["count"], len(expected))
                if case["intent"] == "count":
                    self.assertIn(f"共 {len(expected)} 场", result["reply"])
                if case["intent"] == "attendees":
                    self.assertTrue(
                        all(item["names"] == ["张三"] for item in result["claims"])
                    )

    def test_instant_ranges_preserve_precision_and_cross_midnight_overlap(self):
        repo = Repository(
            os.environ["TEST_DATABASE_URL"], Path("/tmp/unused-knowledge-fixture")
        )
        self.addCleanup(repo.engine.dispose)
        workspace = uuid.uuid4()
        rows = [
            ("2026-11-11T23:50:00+08:00", "2026-11-12T01:00:00+08:00"),
            ("2026-11-12T12:00:00+08:00", "2026-11-12T13:00:00+08:00"),
            ("2026-11-19T12:00:00+08:00", "2026-11-19T13:00:00+08:00"),
            (None, None),
        ]
        ids = []
        with repo.engine.begin() as c:
            c.execute(insert(workspaces).values(id=workspace, name="precision-oracle"))
            for start, end in rows:
                event = uuid.uuid4()
                ids.append(str(event))
                value = datetime.fromisoformat(start) if start else None
                c.execute(
                    insert(k.sessions).values(
                        id=event,
                        workspace_id=workspace,
                        identity_key=str(event),
                        title="边界会议",
                        category="meeting",
                        event_date=value.date() if value else date(2026, 11, 12),
                        event_start_at=value,
                        event_end_at=datetime.fromisoformat(end) if end else None,
                        timezone="Asia/Shanghai",
                        precision="second" if start else "day",
                        time_status="confirmed",
                        revision=1,
                        deleted=False,
                    )
                )
        anchor = datetime.fromisoformat("2026-11-19T12:00:00+08:00")
        result = query(repo, workspace, "最近七天几场会议", anchor=anchor)
        self.assertEqual(result["matched_ids"], [ids[1]])
        self.assertEqual(result["coverage"]["unknown_dates"], 1)
        self.assertEqual(result["status"], "partial")
        overlap = query(
            repo, workspace, "2026年11月12日期间举行过的会议", anchor=anchor
        )
        self.assertEqual(set(overlap["matched_ids"]), set(ids[:2]))
        self.assertEqual(overlap["status"], "partial")
