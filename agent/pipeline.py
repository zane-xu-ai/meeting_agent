"""会议助手 - 核心处理流水线编排"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from agent.asr_client import ASRClient
from agent.llm_client import LLMClient
from agent.prompts import (
    MEETING_ANALYSIS_PROMPT,
    MERGE_SUMMARIES_PROMPT,
    SEGMENT_SUMMARY_PROMPT,
)
from agent.text_processor import TextProcessor
from config import settings
from models.schemas import MeetingResult, Transcript
from utils.audio import preprocess_audio


class MeetingPipeline:
    """会议处理流水线: 音频 -> 转写 -> 预处理 -> LLM 分析 -> 结构化输出"""

    def __init__(self):
        self.asr = ASRClient()
        self.llm = LLMClient()
        self.text_processor = TextProcessor()

    async def run(self, audio_source: str | Path) -> MeetingResult:
        """
        执行完整的会议分析流水线。

        Args:
            audio_source: 音频文件路径 或 可访问的音频 URL

        Returns:
            MeetingResult 结构化会议纪要
        """
        audio_str = str(audio_source)
        is_url = audio_str.startswith(("http://", "https://"))
        label = "URL" if is_url else Path(audio_source).name
        logger.info(f"========== 会议分析开始: {label} ==========")

        if is_url:
            # URL 模式: 跳过预处理，直接转写
            logger.info("[1/4] 跳过预处理 (远程 URL)")
            asr_input = audio_str
            preprocessed = None
        else:
            # 本地文件模式: 预处理
            audio_path = Path(audio_source)
            logger.info("[1/4] 音频预处理...")
            preprocessed = preprocess_audio(
                audio_path, sample_rate=settings.asr_sample_rate)
            asr_input = preprocessed

        try:
            # Step 2: ASR 转写
            logger.info("[2/4] ASR 语音转写...")
            transcript = await self.asr.transcribe(asr_input)

            if not transcript.sentences:
                raise ValueError("ASR 转写结果为空，请检查音频文件是否包含有效语音")

            # Step 3: 文本预处理
            logger.info("[3/4] 文本预处理...")
            cleaned = self.text_processor.clean(transcript)

            # Step 4: LLM 分析
            logger.info("[4/4] LLM 智能分析...")
            result = await self._analyze(cleaned)

            logger.info(f"========== 会议分析完成 ==========")
            return result

        finally:
            # 清理预处理文件
            if preprocessed and preprocessed.exists():
                preprocessed.unlink()

    async def _analyze(self, transcript: Transcript) -> MeetingResult:
        """根据文本长度选择整体分析或分段分析"""
        full_text = transcript.formatted_text
        # 粗略估算 token 数 (中文约 1.5 字/token)
        estimated_tokens = len(full_text) // 2

        if estimated_tokens <= settings.max_tokens_per_chunk:
            logger.info(f"文本长度适中 (~{estimated_tokens} tokens)，执行整体分析")
            return await self._single_analysis(full_text)
        else:
            logger.info(
                f"文本较长 (~{estimated_tokens} tokens)，执行分段分析"
            )
            return await self._segmented_analysis(transcript)

    async def _single_analysis(self, text: str) -> MeetingResult:
        """整体分析: 一次性将全部文本交给 LLM"""
        prompt = MEETING_ANALYSIS_PROMPT.format(transcript=text)
        data = await self.llm.chat_json(prompt)
        return MeetingResult.model_validate(data)

    async def _segmented_analysis(self, transcript: Transcript) -> MeetingResult:
        """
        分段分析策略:
        1. 按 5 分钟窗口切片
        2. 每个切片分别调 LLM 生成局部摘要
        3. 合并所有局部摘要，再调一次 LLM 生成全局结果
        """
        # 切片
        chunks = self.text_processor.chunk_by_time(
            transcript, window_minutes=5)

        if len(chunks) <= 1:
            return await self._single_analysis(transcript.formatted_text)

        # 分段分析
        segment_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"分析第 {i + 1}/{len(chunks)} 段...")
            prompt = SEGMENT_SUMMARY_PROMPT.format(transcript=chunk)
            try:
                data = await self.llm.chat_json(prompt)
                segment_results.append(json.dumps(data, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"第 {i + 1} 段分析失败: {e}")
                continue

        if not segment_results:
            raise ValueError("所有分段分析均失败")

        # 合并
        logger.info(f"合并 {len(segment_results)} 段分析结果...")
        combined = "\n\n---\n\n".join(
            f"### 片段 {i + 1}\n{s}" for i, s in enumerate(segment_results)
        )
        merge_prompt = MERGE_SUMMARIES_PROMPT.format(summaries=combined)
        data = await self.llm.chat_json(merge_prompt)
        return MeetingResult.model_validate(data)


def format_markdown(result: MeetingResult, title: str = "会议纪要") -> str:
    """将 MeetingResult 格式化为 Markdown 输出"""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# {title} - {now}")
    lines.append("")

    # 参会人
    if result.participants:
        lines.append("## 参会人")
        lines.append("、".join(result.participants))
        lines.append("")

    # 全文摘要
    lines.append("## 全文摘要")
    lines.append(result.summary)
    lines.append("")

    # 章节速览
    if result.topics:
        lines.append("## 章节速览")
        for i, topic in enumerate(result.topics, 1):
            time_range = ""
            if topic.start_time or topic.end_time:
                time_range = f" ({topic.start_time or '?'} - {topic.end_time or '?'})"
            lines.append(f"### {i}. {topic.title}{time_range}")
            lines.append(topic.summary)
            lines.append("")

    # 关键决策
    if result.decisions:
        lines.append("## 关键决策")
        for i, d in enumerate(result.decisions, 1):
            lines.append(f"{i}. {d}")
        lines.append("")

    # 待办事项
    if result.todos:
        lines.append("## 待办事项")
        lines.append("| 任务 | 负责人 | 截止时间 |")
        lines.append("|------|--------|----------|")
        for todo in result.todos:
            owner = todo.owner or "-"
            deadline = todo.deadline or "-"
            lines.append(f"| {todo.content} | {owner} | {deadline} |")
        lines.append("")

    # 关键词
    if result.keywords:
        lines.append("## 关键词")
        lines.append("、".join(result.keywords))
        lines.append("")

    return "\n".join(lines)
