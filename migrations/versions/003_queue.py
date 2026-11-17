"""Add durable dispatch and checkpoint state."""

from alembic import op
from persistence import queue_schema as s

revision = "003_queue"
down_revision = "002_cache"


def upgrade():
    for table in (s.outbox, s.checkpoints, s.batch_receipts):
        table.create(op.get_bind())


def downgrade():
    for table in (s.batch_receipts, s.checkpoints, s.outbox):
        table.drop(op.get_bind())
