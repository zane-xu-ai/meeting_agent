"""Add versioned knowledge events, evidence, queries and relation projections."""
from alembic import op
from knowledge.schema import TABLES
revision = '004_knowledge'
down_revision = '003_queue'

def upgrade():
    for table in TABLES:
        table.create(op.get_bind())

def downgrade():
    for table in reversed(TABLES):
        table.drop(op.get_bind())
