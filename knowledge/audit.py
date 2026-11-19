"""Report existing analysis coverage without exposing document contents."""

import json

from sqlalchemy import func, select

from persistence.repository import repository
from persistence.schema import analyses, media_sources, analysis_jobs


def audit(repo):
    with repo.engine.connect() as connection:
        return {
            'sources': connection.scalar(select(func.count()).select_from(media_sources)),
            'analyses': connection.scalar(select(func.count()).select_from(analyses)),
            'meeting_jobs': connection.scalar(select(func.count()).select_from(analysis_jobs).where(analysis_jobs.c.category == 'meeting')),
            'sources_without_explicit_creation_time': connection.scalar(select(func.count()).select_from(media_sources).where(media_sources.c.source_created_at.is_(None))),
            'structured_analyses': connection.scalar(select(func.count()).select_from(analyses).where(analyses.c.result_json.is_not(None))),
        }


if __name__ == '__main__':
    repo = repository()
    if repo is None:
        raise SystemExit('DATABASE_URL is required')
    print(json.dumps(audit(repo), ensure_ascii=False))
