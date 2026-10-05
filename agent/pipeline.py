"""会议助手 - 核心处理流水线编排"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from agent.asr_client import ASRClient
from agent.llm_client import LLMClient
from agent.prompts import (
    GENERAL_ANALYSIS_PROMPT,
    GENERAL_MERGE_PROMPT,
    GENERAL_SEGMENT_PROMPT,
    INTERVIEW_ANALYSIS_PROMPT,
    INTERVIEW_QUESTIONS_PROMPT,
    MEETING_ANALYSIS_PROMPT,
    MERGE_SUMMARIES_PROMPT,
    SEGMENT_SUMMARY_PROMPT,
)
from agent.text_processor import TextProcessor
from config import settings
from models.schemas import MeetingResult, Transcript
from utils.audio import preprocess_audio


def _build_meta_block(
    event_time: str | None = None,
    timing_info: dict | None = None,
) -> list[str]:
    """
    构建文档头部的元数据信息块。

    Args:
        event_time: 事件时间(会议/面试时间)，来自音频文件修改时间
        timing_info: 耗时信息 {generation_time, asr_duration, llm_duration, total_duration}
    """
    lines = []
    gen_time = (timing_info or {}).get("generation_time", "")

    parts = []
    if event_time:
        parts.append(f"事件时间: {event_time}")
    if gen_time:
        parts.append(f"生成时间: {gen_time}")
    if parts:
        lines.append("> " + " | ".join(parts))

    if timing_info:
        asr_d = timing_info.get("asr_duration")
        llm_d = timing_info.get("llm_duration")
        total_d = timing_info.get("total_duration")
        timing_parts = []
        if asr_d is not None:
            timing_parts.append(f"ASR 耗时: {asr_d:.1f}s")
        if llm_d is not None:
            timing_parts.append(f"LLM 耗时: {llm_d:.1f}s")
        if total_d is not None:
            timing_parts.append(f"总耗时: {total_d:.1f}s")
        if timing_parts:
            lines.append("> " + " | ".join(timing_parts))

    if lines:
        lines.append("")  # 空行分隔
    return lines


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
            preprocessed, _ = preprocess_audio(
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
        分段分析策略 (滑动窗口):
        1. 按 5 分钟窗口切片，相邻片段重叠 3 句
        2. 每个切片分别调 LLM 生成局部摘要
        3. 合并所有局部摘要，再调一次 LLM 生成全局结果
        """
        # 切片 (带重叠)
        chunks = self.text_processor.chunk_by_time(
            transcript, window_minutes=5, overlap_sentences=3)

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


# ---------- 面试录音分析 ----------

async def analyze_interview(
    transcript_text: str,
    llm_model: str | None = None,
) -> dict:
    """
    面试录音分析: 调用 LLM 获取问题和评分的原始 JSON 数据。

    长文本采用分段提取 + 合并策略 (滑动窗口)。

    Args:
        transcript_text: 转写文本
        llm_model: 指定 LLM 模型，None 使用默认模型列表

    Returns:
        {"q_data": {...}, "a_data": {...}} 两份原始 JSON 数据
    """
    from agent.text_processor import TextProcessor

    llm = LLMClient(models=[llm_model] if llm_model else None)
    estimated_tokens = len(transcript_text) // 2

    if estimated_tokens <= settings.max_tokens_per_chunk:
        # 短文本: 直接整体分析
        return await _interview_single(llm, transcript_text)

    # 长文本: 分段提取问题 + 合并分析
    logger.info(
        f"面试文本较长 (~{estimated_tokens} tokens)，采用分段提取策略"
    )
    return await _interview_segmented(llm, transcript_text)


async def _interview_single(llm: LLMClient, transcript_text: str) -> dict:
    """面试整体分析 (短文本)"""
    logger.info("面试分析 [1/2]: 提取面试官问题...")
    q_prompt = INTERVIEW_QUESTIONS_PROMPT.format(transcript=transcript_text)
    q_data = await llm.chat_json(q_prompt)

    logger.info("面试分析 [2/2]: 逐题分析评分...")
    a_prompt = INTERVIEW_ANALYSIS_PROMPT.format(transcript=transcript_text)
    a_data = await llm.chat_json(a_prompt, max_tokens=16384)

    return {"q_data": q_data, "a_data": a_data}


async def _interview_segmented(llm: LLMClient, transcript_text: str) -> dict:
    """
    面试分段分析 (滑动窗口):
    1. 按句子边界切片，相邻片段重叠 3 句
    2. 每个片段分别提取问题
    3. 合并去重所有问题
    4. 对合并后的问题列表统一分析
    """
    from agent.text_processor import TextProcessor

    processor = TextProcessor()
    chunks = processor.chunk_by_sentences(
        transcript_text, chunk_size=3000, overlap_sentences=3)

    # Step 1: 每个片段提取问题
    all_questions = []
    seen_questions = set()
    for i, chunk in enumerate(chunks):
        logger.info(f"面试分段提取 [片段 {i + 1}/{len(chunks)}]...")
        q_prompt = INTERVIEW_QUESTIONS_PROMPT.format(transcript=chunk)
        try:
            q_data = await llm.chat_json(q_prompt)
            for q in q_data.get("questions", []):
                # 简单去重: 按问题文本前 20 字去重
                q_key = q.get("question", "")[:20]
                if q_key and q_key not in seen_questions:
                    seen_questions.add(q_key)
                    all_questions.append(q)
        except Exception as e:
            logger.warning(f"片段 {i + 1} 问题提取失败: {e}")

    if not all_questions:
        raise ValueError("所有分段均未提取到面试问题")

    # 重新编号
    for i, q in enumerate(all_questions, 1):
        q["id"] = i

    logger.info(f"合并去重后共 {len(all_questions)} 个问题")
    q_data = {"questions": all_questions}

    # Step 2: 用合并后的问题列表 + 全文进行分析
    # 将问题列表注入文本开头，引导 LLM 按问题顺序分析
    question_list = "\n".join(
        f"{i + 1}. {q['question']}" for i, q in enumerate(all_questions)
    )
    guided_transcript = (
        f"## 已提取的面试问题列表\n{question_list}\n\n"
        f"## 面试转写原文\n{transcript_text}"
    )

    logger.info("面试分析 [2/2]: 逐题分析评分...")
    a_prompt = INTERVIEW_ANALYSIS_PROMPT.format(transcript=guided_transcript)
    a_data = await llm.chat_json(a_prompt, max_tokens=16384)

    return {"q_data": q_data, "a_data": a_data}


def format_interview_questions(
    data: dict,
    event_time: str | None = None,
    timing_info: dict | None = None,
) -> str:
    """将面试问题 JSON 格式化为 Markdown"""
    lines = []
    title_suffix = event_time or datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# 面试问题列表 - {title_suffix}")
    lines.append("")
    lines.extend(_build_meta_block(event_time, timing_info))

    questions = data.get("questions", [])
    if not questions:
        lines.append("未识别到面试官提问。")
        return "\n".join(lines)

    lines.append(f"共提取 **{len(questions)}** 个问题：")
    lines.append("")

    for q in questions:
        qid = q.get("id", "?")
        question = q.get("question", "")
        context = q.get("context", "")
        time_range = q.get("time_range", "")
        tag = f" `{context}`" if context else ""
        time_tag = f" ({time_range})" if time_range else ""
        lines.append(f"### Q{qid}. {question}{tag}{time_tag}")
        lines.append("")

    return "\n".join(lines)


def format_interview_analysis(
    data: dict,
    event_time: str | None = None,
    timing_info: dict | None = None,
) -> str:
    """将面试分析 JSON 格式化为 Markdown"""
    lines = []
    title_suffix = event_time or datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# 面试分析报告 - {title_suffix}")
    lines.append("")
    lines.extend(_build_meta_block(event_time, timing_info))

    # 总结放在最上面
    overall_summary = data.get("overall_summary", "")
    overall_score = data.get("overall_score", 0)
    lines.append("## 面试总结")
    lines.append("")
    lines.append(f"**最终评分: {overall_score}/100**")
    lines.append("")
    lines.append(overall_summary)
    lines.append("")

    # 逐题分析
    analyses = data.get("question_analyses", [])
    if not analyses:
        return "\n".join(lines)

    lines.append("---")
    lines.append("")
    lines.append("## 逐题分析")
    lines.append("")

    for a in analyses:
        qid = a.get("id", "?")
        question = a.get("question", "")
        score = a.get("score", 0)
        candidate_answer = a.get("candidate_answer", "")
        recommended = a.get("recommended_answer", "")
        improvement = a.get("improvement", {})

        lines.append(f"### Q{qid}. {question}")
        lines.append("")
        lines.append(f"**得分: {score}/100**")
        lines.append("")

        lines.append("#### 面试者回答")
        lines.append(candidate_answer)
        lines.append("")

        lines.append("#### 推荐答案")
        lines.append(recommended)
        lines.append("")

        lines.append("#### 改进建议")
        if improvement:
            for dim in ["accuracy", "completeness", "expression", "suggestions"]:
                label_map = {
                    "accuracy": "准确性",
                    "completeness": "全面性",
                    "expression": "表达",
                    "suggestions": "改进建议",
                }
                val = improvement.get(dim, "")
                if val:
                    lines.append(f"- **{label_map[dim]}**: {val}")
        lines.append("")

    return "\n".join(lines)


# ---------- 通用录音分析 ----------

async def analyze_general(
    transcript_text: str,
    llm_model: str | None = None,
) -> dict:
    """
    通用录音分析: 调用 LLM 获取原始 JSON 数据。

    Args:
        transcript_text: 转写文本
        llm_model: 指定 LLM 模型，None 使用默认模型列表

    Returns:
        原始 JSON 数据 dict
    """
    llm = LLMClient(models=[llm_model] if llm_model else None)
    estimated_tokens = len(transcript_text) // 2

    if estimated_tokens <= settings.max_tokens_per_chunk:
        logger.info(f"通用分析: 文本较短 (~{estimated_tokens} tokens)，整体分析")
        prompt = GENERAL_ANALYSIS_PROMPT.format(transcript=transcript_text)
        return await llm.chat_json(prompt)
    else:
        logger.info(f"通用分析: 文本较长 (~{estimated_tokens} tokens)，分段分析")
        return await _segmented_general(transcript_text, llm)


async def _segmented_general(
    transcript_text: str,
    llm: LLMClient,
) -> dict:
    """通用录音的分段分析 (滑动窗口)"""
    from agent.text_processor import TextProcessor

    processor = TextProcessor()
    # 按句子边界切片，相邻片段重叠 3 句
    chunks = processor.chunk_by_sentences(
        transcript_text, chunk_size=3000, overlap_sentences=3)

    if len(chunks) <= 1:
        prompt = GENERAL_ANALYSIS_PROMPT.format(transcript=transcript_text)
        return await llm.chat_json(prompt)

    segment_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"通用分析: 第 {i + 1}/{len(chunks)} 段...")
        prompt = GENERAL_SEGMENT_PROMPT.format(transcript=chunk)
        try:
            data = await llm.chat_json(prompt)
            segment_results.append(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"第 {i + 1} 段分析失败: {e}")
            continue

    if not segment_results:
        raise ValueError("所有分段分析均失败")

    combined = "\n\n---\n\n".join(
        f"### 片段 {i + 1}\n{s}" for i, s in enumerate(segment_results)
    )
    merge_prompt = GENERAL_MERGE_PROMPT.format(summaries=combined)
    return await llm.chat_json(merge_prompt)


def format_general_result(
    data: dict,
    event_time: str | None = None,
    timing_info: dict | None = None,
) -> str:
    """将通用分析 JSON 格式化为 Markdown"""
    lines = []
    title_suffix = event_time or datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# 内容分析总结 - {title_suffix}")
    lines.append("")
    lines.extend(_build_meta_block(event_time, timing_info))

    # 摘要
    summary = data.get("summary", "")
    if summary:
        lines.append("## 全文摘要")
        lines.append(summary)
        lines.append("")

    # 说话人
    speakers = data.get("speakers", [])
    if speakers:
        lines.append("## 说话人")
        lines.append("、".join(speakers))
        lines.append("")

    # 章节速览
    topics = data.get("topics", [])
    if topics:
        lines.append("## 章节速览")
        for i, topic in enumerate(topics, 1):
            time_range = ""
            st = topic.get("start_time", "")
            et = topic.get("end_time", "")
            if st or et:
                time_range = f" ({st or '?'} - {et or '?'})"
            lines.append(f"### {i}. {topic.get('title', '')}{time_range}")
            lines.append(topic.get("summary", ""))
            lines.append("")

    # 亮点
    highlights = data.get("highlights", [])
    if highlights:
        lines.append("## 关键信息")
        for h in highlights:
            lines.append(f"- {h}")
        lines.append("")

    # 关键词
    keywords = data.get("keywords", [])
    if keywords:
        lines.append("## 关键词")
        lines.append("、".join(keywords))
        lines.append("")

    return "\n".join(lines)


# ---------- 格式化函数 (保留会议) ----------

def format_markdown(
    result: MeetingResult,
    title: str = "会议纪要",
    event_time: str | None = None,
    timing_info: dict | None = None,
) -> str:
    """将 MeetingResult 格式化为 Markdown 输出"""
    lines = []
    title_suffix = event_time or datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# {title} - {title_suffix}")
    lines.append("")
    lines.extend(_build_meta_block(event_time, timing_info))

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
