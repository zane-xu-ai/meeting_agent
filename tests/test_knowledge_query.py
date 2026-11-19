"""Baseline SQL answers, evidence and frozen temporal sets."""
import os
import uuid
import unittest
from datetime import datetime,date
from pathlib import Path
from tempfile import TemporaryDirectory
from sqlalchemy import select,insert
from persistence.schema import workspaces
from persistence.repository import Repository,PERSONAL
from knowledge.store import build,confirm
from knowledge.query import query
from knowledge_support import sample

@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL'),'requires disposable PostgreSQL')
class KnowledgeQueryTests(unittest.TestCase):
    def test_attendance_topics_counts_and_cross_workspace(self):
        with TemporaryDirectory() as root:
            repo=Repository(os.environ['TEST_DATABASE_URL'],Path(root)/'archive');self.addCleanup(repo.engine.dispose)
            analysis=sample(repo,root,title=uuid.uuid4().hex)
            event=build(repo,analysis)
            confirm(repo,PERSONAL,event,event_date=date(2037,11,10),names=['测试甲','测试乙'])
            anchor=datetime.fromisoformat('2037-11-19T12:00:00+08:00')
            a=query(repo,PERSONAL,'2037年11月10日有哪些人参加？',uuid.uuid4().hex,anchor=anchor)
            self.assertIn('测试甲',a['reply']);self.assertTrue(a['evidence'])
            b=query(repo,PERSONAL,'2037年11月10日讨论了什么主题？',uuid.uuid4().hex,anchor=anchor)
            self.assertIn('预算',b['reply'])
            other=uuid.uuid4()
            with repo.engine.begin() as c:c.execute(insert(workspaces).values(id=other,name='isolated'))
            private=query(repo,other,'2037年11月10日有哪些人参加？',anchor=anchor)
            self.assertEqual(private['status'],'no_match');self.assertFalse(private['evidence'])
            follow=query(repo,PERSONAL,'其中测试甲参加几次？',a['query_id'],anchor=anchor)
            self.assertEqual(follow['status'],'clarify')
