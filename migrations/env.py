"""Run Meeting Agent schema migrations with a dedicated DSN."""

import os

from alembic import context
from sqlalchemy import create_engine, text

from config import settings
from persistence.schema import metadata
from persistence import cache_schema, queue_schema  # noqa: F401
from knowledge import schema as knowledge_schema  # noqa: F401


dsn = os.environ.get("MIGRATION_DATABASE_URL") or settings.database_url
if not dsn:
    raise RuntimeError("请配置 MIGRATION_DATABASE_URL 或 DATABASE_URL")

options = {"target_metadata": metadata, "version_table_schema": "meeting", "include_schemas": True}
if context.is_offline_mode():
    context.configure(url=dsn, literal_binds=True, **options)
    with context.begin_transaction():
        context.execute("CREATE SCHEMA IF NOT EXISTS meeting")
        context.run_migrations()
else:
    engine = create_engine(dsn)
    with engine.connect() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS meeting"))
        connection.commit()
        context.configure(connection=connection, **options)
        with context.begin_transaction():
            context.run_migrations()
