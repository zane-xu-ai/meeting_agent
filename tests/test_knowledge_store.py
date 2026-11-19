"""Facts are idempotent and unconfirmed metadata never becomes a confirmed roster."""
import os
import unittest
import uuid
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from sqlalchemy import select,func
from persistence.repository import Repository,PERSONAL
from knowledge import schema as k
from knowledge.store import build,confirm,owned
from knowledge_support import sample

@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL'),'requires disposable PostgreSQL')
class KnowledgeStoreTests(unittest.TestCase):
    def test_build_and_correction_are_scoped_and_keep_provenance(self):
        with TemporaryDirectory() as root:
            repo=Repository(os.environ['TEST_DATABASE_URL'],Path(root)/'media')
            self.addCleanup(repo.engine.dispose)
            identifier=sample(repo,root)
            session=build(repo,identifier)
            self.assertEqual(build(repo,identifier),session)
            with repo.engine.connect() as c:
                row=owned(c,PERSONAL,session)
                self.assertEqual(row['time_status'],'candidate')
                self.assertEqual(row['event_date'],date(2026,11,10))
                self.assertFalse(c.scalar(select(func.count()).select_from(k.participants).where(k.participants.c.session_id==session,k.participants.c.status=='confirmed')))
                with self.assertRaises(LookupError):owned(c,uuid.uuid4(),session)
            confirm(repo,PERSONAL,session,event_date=date(2026,11,10),names=['张三','李四','张三'])
            with repo.engine.connect() as c:
                self.assertEqual(owned(c,PERSONAL,session)['time_status'],'confirmed')
                self.assertEqual(c.scalar(select(func.count()).select_from(k.participants).where(k.participants.c.session_id==session,k.participants.c.active.is_(True))),2)
                self.assertGreater(c.scalar(select(func.count()).select_from(k.evidence).where(k.evidence.c.session_id==session)),2)
