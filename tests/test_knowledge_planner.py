"""Freeze dates and compare against an independent eighty-question corpus."""
import json
import unittest
from pathlib import Path
from datetime import datetime
from knowledge.planner import plan,period,Clarification

class PlannerTests(unittest.TestCase):
    def test_eighty_frozen_questions(self):
        cases=json.loads((Path(__file__).parent/'fixtures/meeting_queries.json').read_text())
        for case in cases:
            with self.subTest(question=case['question']):
                p=plan(case['question'],datetime.fromisoformat(case['anchor']),case['timezone'])
                self.assertEqual((p.intent,str(p.start),str(p.stop)),(case['intent'],case['start'],case['stop']))
    def test_leap_day_and_ambiguous_date_require_clarification(self):
        for question,anchor in [('去年今天的主题',datetime.fromisoformat('2024-02-29T12:00:00+08:00')),('11月19日有哪些会议',datetime.fromisoformat('2026-11-19T12:00:00+08:00'))]:
            with self.assertRaises(Clarification):plan(question,anchor)
    def test_local_timezone_controls_the_day(self):
        anchor=datetime.fromisoformat('2026-11-19T01:00:00+00:00')
        p=plan('今天会议',anchor,'America/Los_Angeles')
        self.assertEqual(str(p.start),'2026-11-18')
