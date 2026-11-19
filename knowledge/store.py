"""Fact construction and correction using existing analyses and source evidence."""

from datetime import date
import hashlib
import re
import uuid
from sqlalchemy import select, insert, update, func
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from loguru import logger
from sqlalchemy.dialects.postgresql import insert as upsert
from persistence import schema as base
from knowledge import schema as k

EXTRACTOR = "facts-v2"


def bump(connection, workspace):
    stmt = upsert(k.state).values(workspace_id=workspace, revision=1)
    return connection.scalar(
        stmt.on_conflict_do_update(
            index_elements=["workspace_id"], set_={"revision": k.state.c.revision + 1}
        ).returning(k.state.c.revision)
    )


def revision(connection, workspace):
    return (
        connection.scalar(
            select(k.state.c.revision).where(k.state.c.workspace_id == workspace)
        )
        or 0
    )


def owned(connection, workspace, session, *, lock=False):
    query = select(k.sessions).where(
        k.sessions.c.id == session,
        k.sessions.c.workspace_id == workspace,
        k.sessions.c.deleted.is_(False),
    )
    if lock:
        query = query.with_for_update()
    row = connection.execute(query).mappings().one_or_none()
    if row is None:
        raise LookupError("会议不存在")
    return row


def add_evidence(
    c,
    workspace,
    session,
    quote,
    *,
    analysis=None,
    segment=None,
    kind="transcript",
    rev=1,
):
    identifier = uuid.uuid4()
    c.execute(
        insert(k.evidence).values(
            id=identifier,
            workspace_id=workspace,
            session_id=session,
            analysis_id=analysis,
            segment_id=segment["id"] if segment else None,
            kind=kind,
            quote=quote,
            start_ms=segment["start_ms"] if segment else None,
            end_ms=segment["end_ms"] if segment else None,
            source_revision=rev,
            text_hash=hashlib.sha256(quote.encode()).hexdigest(),
        )
    )
    return identifier


def build(repo, analysis_id):
    """Reconcile an analysis without invoking ASR or a model; unknown identities stay unknown."""
    with repo.engine.begin() as c:
        # Serializes concurrent builders and corrections across processes.
        c.execute(
            select(
                func.pg_advisory_xact_lock(func.hashtextextended(str(analysis_id), 0))
            )
        )
        done = (
            c.execute(
                select(k.builds).where(
                    k.builds.c.analysis_id == analysis_id,
                    k.builds.c.extractor_version == EXTRACTOR,
                )
            )
            .mappings()
            .one_or_none()
        )
        if done and done["status"] == "completed":
            return done["session_id"]
        row = (
            c.execute(
                select(
                    base.analyses,
                    base.analysis_jobs.c.workspace_id,
                    base.analysis_jobs.c.media_version_id,
                    base.media_sources.c.original_filename,
                )
                .join(
                    base.analysis_jobs,
                    base.analyses.c.job_id == base.analysis_jobs.c.id,
                )
                .join(
                    base.media_sources,
                    base.analysis_jobs.c.source_id == base.media_sources.c.id,
                )
                .where(base.analyses.c.id == analysis_id)
            )
            .mappings()
            .one()
        )
        workspace = row["workspace_id"]
        checksum = (
            c.scalar(
                select(base.media_versions.c.content_sha256).where(
                    base.media_versions.c.id == row["media_version_id"]
                )
            )
            if row["media_version_id"]
            else None
        )
        identity = checksum or "analysis:" + str(analysis_id)
        stmt = upsert(k.sessions).values(
            id=uuid.uuid4(),
            workspace_id=workspace,
            identity_key=identity,
            title=row["title"] or row["original_filename"] or "未命名会议",
            category=row["category"],
            timezone="Asia/Shanghai",
            precision="unknown",
            time_status="unknown",
            revision=1,
            deleted=False,
        )
        session = c.scalar(
            stmt.on_conflict_do_update(
                index_elements=["workspace_id", "identity_key"],
                set_={"updated_at": func.now()},
            ).returning(k.sessions.c.id)
        )
        event = (
            c.execute(
                select(k.sessions).where(k.sessions.c.id == session).with_for_update()
            )
            .mappings()
            .one()
        )
        if event["deleted"]:
            c.execute(
                upsert(k.builds)
                .values(
                    id=uuid.uuid4(),
                    analysis_id=analysis_id,
                    session_id=session,
                    status="completed",
                    extractor_version=EXTRACTOR,
                )
                .on_conflict_do_update(
                    index_elements=["analysis_id", "extractor_version"],
                    set_={
                        "status": "completed",
                        "session_id": session,
                        "updated_at": func.now(),
                    },
                )
            )
            return session
        exists = c.scalar(
            select(k.recordings.c.id).where(k.recordings.c.analysis_id == analysis_id)
        )
        if not exists:
            c.execute(
                update(k.recordings)
                .where(k.recordings.c.session_id == session)
                .values(current=False)
            )
            c.execute(
                insert(k.recordings).values(
                    id=uuid.uuid4(),
                    session_id=session,
                    analysis_id=analysis_id,
                    media_version_id=row["media_version_id"],
                    current=True,
                )
            )
        # Preserve confirmed identity/date corrections while refreshing derived text.
        confirmed = c.scalar(
            select(func.count())
            .select_from(k.participants)
            .where(
                k.participants.c.session_id == session,
                k.participants.c.active.is_(True),
                k.participants.c.status == "confirmed",
            )
        )
        c.execute(
            update(k.participants)
            .where(
                k.participants.c.session_id == session,
                k.participants.c.status != "confirmed",
            )
            .values(active=False)
        )
        c.execute(
            update(k.topics)
            .where(k.topics.c.session_id == session)
            .values(active=False)
        )
        match = re.search(
            r"(20\d{2})[年._-](\d{1,2})[月._-](\d{1,2})", row["original_filename"] or ""
        )
        if match and event["time_status"] != "confirmed":
            try:
                value = date(*map(int, match.groups()))
            except ValueError:
                value = None
            if value:
                c.execute(
                    insert(k.time_candidates).values(
                        id=uuid.uuid4(),
                        session_id=session,
                        raw=match.group(),
                        source="filename",
                        value=value,
                        status="candidate",
                    )
                )
                c.execute(
                    update(k.sessions)
                    .where(k.sessions.c.id == session)
                    .values(event_date=value, precision="day", time_status="candidate")
                )
        data = row["result_json"] if isinstance(row["result_json"], dict) else {}
        segments = list(
            c.execute(
                select(base.transcript_segments)
                .where(base.transcript_segments.c.transcript_id == row["transcript_id"])
                .order_by(base.transcript_segments.c.ordinal)
            ).mappings()
        )
        for name in (data.get("participants") or []) if not confirmed else []:
            if not isinstance(name, str) or not name.strip():
                continue
            segment = next((part for part in segments if name in part["text"]), None)
            quote = segment["text"] if segment else "分析报告列出的参与人：" + name
            evidence = add_evidence(
                c,
                workspace,
                session,
                quote,
                analysis=analysis_id,
                segment=segment,
                kind="transcript" if segment else "analysis",
            )
            c.execute(
                insert(k.participants).values(
                    id=uuid.uuid4(),
                    session_id=session,
                    display_name=name,
                    status="reported",
                    evidence_id=evidence,
                    active=True,
                )
            )
        labels = {part["speaker"] for part in segments if part["speaker"]}
        for label in sorted(labels):
            part = next(part for part in segments if part["speaker"] == label)
            speaker = upsert(k.speakers).values(
                id=uuid.uuid4(),
                session_id=session,
                transcript_id=row["transcript_id"],
                label=label,
                confirmed=False,
            )
            speaker_id = c.scalar(
                speaker.on_conflict_do_update(
                    index_elements=["session_id", "transcript_id", "label"],
                    set_={"label": label},
                ).returning(k.speakers.c.id)
            )
            evidence = add_evidence(
                c, workspace, session, part["text"], analysis=analysis_id, segment=part
            )
            c.execute(
                insert(k.participants).values(
                    id=uuid.uuid4(),
                    session_id=session,
                    speaker_id=speaker_id,
                    display_name=label,
                    status="observed_speaker",
                    evidence_id=evidence,
                    active=True,
                )
            )
        # Topics are explicit report sections/keywords; never guess participants from mere mentions.
        titles = []
        for item in data.get("topics") or data.get("chapters") or []:
            if isinstance(item, dict) and isinstance(item.get("title"), str):
                title = item["title"]
                titles.append(
                    (
                        title,
                        (
                            item.get("summary")
                            if isinstance(item.get("summary"), str)
                            else title
                        ),
                    )
                )
        for item in data.get("keywords", []) or []:
            if isinstance(item, str):
                titles.append((item, item))
        summary = data.get("summary") or data.get("overall_summary")
        if not titles and isinstance(summary, str) and summary.strip():
            titles.append(("报告摘要", summary))
        for title, summary in titles:
            part = next((p for p in segments if summary and summary in p["text"]), None)
            evidence = add_evidence(
                c,
                workspace,
                session,
                part["text"] if part else summary,
                analysis=analysis_id,
                segment=part,
                kind="transcript" if part else "analysis",
            )
            c.execute(
                insert(k.topics).values(
                    id=uuid.uuid4(),
                    session_id=session,
                    title=title,
                    summary=summary,
                    evidence_id=evidence,
                    active=True,
                )
            )
        c.execute(k.chunks.delete().where(k.chunks.c.session_id == session))
        for part in segments:
            evidence = add_evidence(
                c, workspace, session, part["text"], analysis=analysis_id, segment=part
            )
            c.execute(
                insert(k.chunks).values(
                    id=uuid.uuid4(),
                    workspace_id=workspace,
                    session_id=session,
                    evidence_id=evidence,
                    text=part["text"],
                    source_revision=event["revision"],
                )
            )
        c.execute(
            upsert(k.builds)
            .values(
                id=uuid.uuid4(),
                analysis_id=analysis_id,
                session_id=session,
                status="completed",
                extractor_version=EXTRACTOR,
            )
            .on_conflict_do_update(
                index_elements=["analysis_id", "extractor_version"],
                set_={
                    "status": "completed",
                    "session_id": session,
                    "updated_at": func.now(),
                },
            )
        )
        bump(c, workspace)
        return session


def backfill(repo):
    """Retry failed records after a cooldown; a bad report cannot block later reports."""
    with repo.engine.connect() as c:
        ids = list(
            c.scalars(
                select(base.analyses.c.id)
                .outerjoin(
                    k.builds,
                    (k.builds.c.analysis_id == base.analyses.c.id)
                    & (k.builds.c.extractor_version == EXTRACTOR),
                )
                .where(
                    (k.builds.c.id.is_(None))
                    | (
                        (k.builds.c.status == "failed")
                        & (
                            k.builds.c.updated_at
                            < datetime.now(timezone.utc) - timedelta(minutes=5)
                        )
                    )
                )
                .order_by(base.analyses.c.created_at)
                .limit(100)
            )
        )
    completed = 0
    for identifier in ids:
        try:
            build(repo, identifier)
            completed += 1
        except Exception as error:
            with repo.engine.begin() as c:
                c.execute(
                    upsert(k.builds)
                    .values(
                        id=uuid.uuid4(),
                        analysis_id=identifier,
                        status="failed",
                        extractor_version=EXTRACTOR,
                        error_code=type(error).__name__,
                    )
                    .on_conflict_do_update(
                        index_elements=["analysis_id", "extractor_version"],
                        set_={
                            "status": "failed",
                            "error_code": type(error).__name__,
                            "updated_at": func.now(),
                        },
                    )
                )
            logger.warning(
                "知识回填失败 analysis={} error={}", identifier, type(error).__name__
            )
    return completed


def confirm(
    repo,
    workspace,
    session,
    *,
    event_date=None,
    names=None,
    event_start_at=None,
    event_end_at=None,
):
    if event_start_at is not None:
        if event_start_at.tzinfo is None or (
            event_end_at is not None and event_end_at.tzinfo is None
        ):
            raise ValueError("准确发生时间必须带时区")
        if event_end_at is not None and event_end_at <= event_start_at:
            raise ValueError("结束时间必须晚于开始时间")
    elif event_end_at is not None:
        raise ValueError("填写结束时间时必须提供开始时间")
    with repo.engine.begin() as c:
        row = owned(c, workspace, session, lock=True)
        rev = row["revision"] + 1
        values = {"revision": rev, "updated_at": func.now()}
        if event_date is not None:
            values.update(
                event_date=event_date, time_status="confirmed", precision="day"
            )
            values.update(event_start_at=None, event_end_at=None)
            c.execute(
                insert(k.time_candidates).values(
                    id=uuid.uuid4(),
                    session_id=session,
                    raw=str(event_date),
                    source="user",
                    value=event_date,
                    status="confirmed",
                )
            )
        if event_start_at is not None:
            local_date = event_start_at.astimezone(ZoneInfo(row["timezone"])).date()
            if event_date is not None and event_date != local_date:
                raise ValueError("日期与开始时间冲突")
            values.update(
                event_date=local_date,
                event_start_at=event_start_at,
                event_end_at=event_end_at,
                time_status="confirmed",
                precision="second",
            )
        if names is not None:
            c.execute(
                update(k.participants)
                .where(k.participants.c.session_id == session)
                .values(active=False)
            )
            for name in dict.fromkeys(names):
                person = c.scalars(
                    select(k.people.c.id).where(
                        k.people.c.workspace_id == workspace,
                        k.people.c.name == name,
                        k.people.c.active.is_(True),
                    )
                ).all()
                if len(person) > 1:
                    raise ValueError("同名人物需要指定身份")
                person_id = person[0] if person else uuid.uuid4()
                if not person:
                    c.execute(
                        insert(k.people).values(
                            id=person_id, workspace_id=workspace, name=name, active=True
                        )
                    )
                evidence = add_evidence(
                    c,
                    workspace,
                    session,
                    "用户确认参会：" + name,
                    kind="user_confirmation",
                    rev=rev,
                )
                c.execute(
                    insert(k.participants).values(
                        id=uuid.uuid4(),
                        session_id=session,
                        person_id=person_id,
                        display_name=name,
                        status="confirmed",
                        evidence_id=evidence,
                        active=True,
                    )
                )
        c.execute(update(k.sessions).where(k.sessions.c.id == session).values(**values))
        c.execute(
            update(k.chunks)
            .where(k.chunks.c.session_id == session)
            .values(source_revision=rev)
        )
        bump(c, workspace)


def schedule_rebuild(repo):
    """Invalidate projection completion markers; preserves facts and human corrections."""
    with repo.engine.begin() as c:
        result = c.execute(
            update(k.builds).values(
                status="failed",
                error_code="operator_rebuild",
                updated_at=datetime.now(timezone.utc) - timedelta(minutes=6),
            )
        )
        return result.rowcount
