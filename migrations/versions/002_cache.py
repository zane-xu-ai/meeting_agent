"""Add cache policy and content-addressed result tables."""

from alembic import op
from persistence.cache_schema import cache_entries, cache_policies

revision = "002_cache"
down_revision = "001_history"


def upgrade():
    cache_policies.create(op.get_bind())
    cache_entries.create(op.get_bind())


def downgrade():
    cache_entries.drop(op.get_bind())
    cache_policies.drop(op.get_bind())
