"""Synthetic analysis fixture for real PostgreSQL acceptance."""

import uuid
from pathlib import Path
from persistence.repository import Repository
from models.schemas import Transcript, Sentence


def sample(repo, root, *, title="2026年11月10日 合成会议", category="meeting"):
    task = uuid.uuid4().hex
    source = Path(root) / f"{task}.wav"
    source.write_bytes(uuid.uuid4().bytes)
    report = Path(root) / f"{task}.md"
    report.write_text("# 合成报告")
    state = {
        "task_id": task,
        "status": "processing",
        "source_type": "local_audio",
        "filename": title + ".wav",
        "rec_type": category,
        "result_path": str(report),
    }
    repo.create(state, str(source), {})
    transcript = Transcript(
        sentences=[
            Sentence(
                text="张三：我们讨论预算和成本，决定下周复核。",
                speaker="Speaker-01",
                begin_time=0,
                end_time=1000,
            )
        ]
    )
    repo.record_result(
        task,
        {
            "source": str(source),
            "output_text": report.read_text(),
            "rec_type": category,
            "transcript_text": transcript.formatted_text,
            "cleaned_transcript": transcript,
            "analysis_data": {
                "participants": ["张三"],
                "chapters": [
                    {"title": "预算", "summary": "讨论预算和成本，决定下周复核。"}
                ],
            },
        },
        state,
    )
    from sqlalchemy import select
    from persistence.schema import analyses

    with repo.engine.connect() as c:
        identifier = c.scalar(select(analyses.c.id).where(analyses.c.job_id == task))
    return identifier
