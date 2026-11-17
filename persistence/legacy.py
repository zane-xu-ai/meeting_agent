"""Import unambiguous historical exports without treating old filenames as cache keys."""

import argparse
import re
from pathlib import Path

from persistence.repository import digest_text, repository


def legacy_records(root):
    for report in sorted((root / "summary").glob("**/*.md")):
        category = report.relative_to(root / "summary").parts[0]
        if category not in {"interview", "meeting", "video", "other"}:
            continue
        if category == "interview" and (report.name != "analyze.md" or not report.with_name("question.md").exists()):
            continue
        label = report.parent.name if category == "interview" else report.stem
        matches = [item for item in (root / "asr").glob("*.txt") if label.startswith(item.stem + "_")]
        yield report, category, label, matches[0] if len(matches) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = repository() if args.apply else None
    if args.apply and not repo:
        parser.error("DATABASE_URL is required for --apply")
    imported = unmatched = 0
    for report, category, label, transcript in legacy_records(args.root):
        item = report.read_text(encoding="utf-8")
        if not item.strip():
            continue
        if not transcript:
            unmatched += 1
        task_id = digest_text(str(report.relative_to(args.root)) + "\0" + digest_text(item))[:32]
        if repo:
            state = repo.task(task_id)
            if state and state["status"] == "completed":
                continue
            category = "general" if category == "other" else category
            state = {"task_id": task_id, "status": "processing", "source_type": "legacy",
                     "filename": label, "result_path": str(report), "rec_type": category}
            if not repo.task(task_id):
                repo.create(state, None, {"legacy": True})
            data = {}
            if category == "interview":
                names = re.findall(r"^### Q(\d+)\. (.+)$", report.with_name("question.md").read_text(), re.M)
                data["q_data"] = {"questions": [{"id": int(n), "question": name} for n, name in names]}
            repo.record_result(task_id, {
                "output_text": item, "rec_type": category, "analysis_data": data,
                "transcript_text": transcript.read_text(encoding="utf-8") if transcript else "",
            }, state)
        imported += 1
    print(f"{'已导入' if repo else '拟导入'} {imported} 份报告，其中 {unmatched} 份无唯一转写关联。")


if __name__ == "__main__":
    main()
