"""Validated plans and deterministic calendar boundaries. No arbitrary SQL."""

from datetime import date, datetime, time, timedelta, timezone as utc_timezone
from typing import Literal
from zoneinfo import ZoneInfo
import re
from pydantic import BaseModel, Field


class Clarification(ValueError):
    pass


class Plan(BaseModel):
    intent: Literal[
        "meetings",
        "attendees",
        "speakers",
        "topics",
        "count",
        "attendance_count",
        "search",
        "relations",
    ]
    start: date
    stop: date
    timezone: str = "Asia/Shanghai"
    category: Literal["meeting", "interview", "general", "video"] = "meeting"
    person: str | None = Field(default=None, max_length=100)
    terms: str | None = Field(default=None, max_length=200)
    complete: bool = True
    inherited_ids: list[str] | None = None
    start_at: datetime | None = None
    stop_at: datetime | None = None
    overlap: bool = False


def month_start(today, delta):
    value = today.year * 12 + today.month - 1 + delta
    return date(value // 12, value % 12 + 1, 1)


def period(question, anchor, timezone):
    try:
        zone = ZoneInfo(timezone)
    except Exception as error:
        raise Clarification("请选择有效时区") from error
    if anchor.tzinfo is None:
        raise Clarification("查询时刻需要带时区")
    today = anchor.astimezone(zone).date()
    if "最近七天" in question or "最近7天" in question:
        return (anchor.astimezone(utc_timezone.utc) - timedelta(days=7)).astimezone(
            zone
        ).date(), today + timedelta(days=1)
    monday = today - timedelta(days=today.weekday())
    if "上季度" in question:
        this_quarter = date(today.year, ((today.month - 1) // 3) * 3 + 1, 1)
        return month_start(this_quarter, -3), this_quarter
    if "去年今天" in question:
        try:
            target = today.replace(year=today.year - 1)
        except ValueError as error:
            raise Clarification(
                "去年没有 2 月 29 日，请指定 2 月 28 日或 3 月 1 日"
            ) from error
        return target, target + timedelta(days=1)
    for phrase, start, stop in [
        ("上周", monday - timedelta(days=7), monday),
        ("本周", monday, monday + timedelta(days=7)),
        ("上个月", month_start(today, -1), month_start(today, 0)),
        ("本月", month_start(today, 0), month_start(today, 1)),
        ("昨天", today - timedelta(days=1), today),
        ("今天", today, today + timedelta(days=1)),
    ]:
        if phrase in question:
            return start, stop
    explicit = re.findall(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})", question)
    if explicit:
        try:
            start = date(*map(int, explicit[0]))
            stop = (
                date(*map(int, explicit[1])) + timedelta(days=1)
                if len(explicit) > 1
                else start + timedelta(days=1)
            )
        except ValueError as error:
            raise Clarification("日期无效，请检查年月日") from error
        if stop <= start:
            raise Clarification("结束日期需要晚于开始日期")
        return start, stop
    raise Clarification("请指定会议发生日期，例如“上周”或“2025年11月19日”")


def plan(question, anchor, timezone="Asia/Shanghai", previous=None):
    if not question.strip() or len(question) > 2000:
        raise Clarification("请输入 1–2000 字的问题")
    inherited = (
        previous
        and "start" in previous
        and "stop" in previous
        and any(word in question for word in ("其中", "那几场", "这些会议"))
    )
    if inherited:
        start, stop = date.fromisoformat(previous["start"]), date.fromisoformat(
            previous["stop"]
        )
    else:
        start, stop = period(question, anchor, timezone)
    if (stop - start).days > 3660:
        raise Clarification("单次查询范围最多十年，请缩小范围")
    category = next(
        (
            kind
            for word, kind in [
                ("面试", "interview"),
                ("视频", "video"),
                ("通用录音", "general"),
            ]
            if word in question
        ),
        "meeting",
    )
    if inherited:
        category = previous.get("category", category)
    if any(word in question for word in ("发言人", "说话人")):
        intent = "speakers"
    elif any(
        word in question for word in ("参加几次", "参会几次", "参加最多", "参与最多")
    ):
        intent = "attendance_count"
    elif any(
        word in question
        for word in ("哪些人", "参与人员", "参会人", "谁参加", "人员名单")
    ):
        intent = "attendees"
    elif any(word in question for word in ("几场", "多少次", "多少场")):
        intent = "count"
    elif any(word in question for word in ("什么主题", "议题", "讨论了什么")):
        intent = "topics"
    elif "讨论" in question or "涉及" in question:
        intent = "search"
    elif any(word in question for word in ("决定", "决策", "跟进", "负责")):
        intent = "relations"
    else:
        intent = "meetings"
    terms = None
    match = re.search(r"(?:讨论|涉及)(.+?)(?:的有哪些|的会议|有哪些|[？?]|$)", question)
    if intent == "search" and match:
        terms = match.group(1).strip("了过的 ")
    if intent == "search" and not terms:
        raise Clarification("请给出要搜索的主题")
    if intent == "relations":
        entity = re.search(r"[“\"]([^”\"]+)[”\"]", question)
        terms = entity.group(1) if entity else None
    person = None
    match = re.search(
        r"(?:里|其中|中|，|,)([^\s，,？?]{1,20}?)(?:参加几次|参会几次)", question
    )
    if match:
        person = match.group(1)
    exact = {}
    if inherited:
        exact = {
            key: previous.get(key)
            for key in ("start_at", "stop_at", "overlap")
            if key in previous
        }
    elif "最近七天" in question or "最近7天" in question:
        stop_at = anchor.astimezone(utc_timezone.utc)
        exact = {"start_at": stop_at - timedelta(days=7), "stop_at": stop_at}
    elif "期间举行过" in question:
        zone = ZoneInfo(timezone)
        exact = {
            "start_at": datetime.combine(start, time.min, zone),
            "stop_at": datetime.combine(stop, time.min, zone),
            "overlap": True,
        }
    return Plan(
        intent=intent,
        start=start,
        stop=stop,
        timezone=timezone,
        category=category,
        person=person,
        terms=terms,
        inherited_ids=previous.get("matched_ids") if inherited else None,
        **exact,
    )
