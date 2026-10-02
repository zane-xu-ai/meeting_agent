"""会议助手 - 数据模型定义"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    """ASR 转写的单条句子"""

    text: str = Field(description="句子文本")
    speaker: str | None = Field(default=None, description="说话人标签，如 Speaker-01")
    begin_time: int = Field(default=0, description="起始时间(毫秒)")
    end_time: int = Field(default=0, description="结束时间(毫秒)")


class Transcript(BaseModel):
    """ASR 转写结果"""

    sentences: list[Sentence] = Field(default_factory=list)
    duration_ms: int = Field(default=0, description="音频总时长(毫秒)")

    @property
    def full_text(self) -> str:
        """拼接全部文本"""
        return "\n".join(
            f"[{s.speaker or 'unknown'}] {s.text}" for s in self.sentences
        )

    @property
    def formatted_text(self) -> str:
        """带时间戳的格式化文本"""
        parts = []
        for s in self.sentences:
            start = _ms_to_time(s.begin_time)
            end = _ms_to_time(s.end_time)
            speaker = s.speaker or "未知"
            parts.append(f"[{start} -> {end}] {speaker}: {s.text}")
        return "\n".join(parts)


class Topic(BaseModel):
    """章节/话题"""

    title: str = Field(description="章节标题")
    summary: str = Field(description="章节摘要")
    start_time: str | None = Field(default=None, description="起始时间")
    end_time: str | None = Field(default=None, description="结束时间")


class TodoItem(BaseModel):
    """待办事项"""

    content: str = Field(description="任务内容")
    owner: str | None = Field(default=None, description="负责人")
    deadline: str | None = Field(default=None, description="截止时间")


class MeetingResult(BaseModel):
    """LLM 分析后的会议纪要结构化结果"""

    summary: str = Field(description="全文摘要")
    topics: list[Topic] = Field(default_factory=list, description="章节速览")
    decisions: list[str] = Field(default_factory=list, description="关键决策")
    todos: list[TodoItem] = Field(default_factory=list, description="待办事项")
    keywords: list[str] = Field(default_factory=list, description="关键词")
    participants: list[str] = Field(default_factory=list, description="参会人")


def _ms_to_time(ms: int) -> str:
    """毫秒转 MM:SS 格式"""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"
