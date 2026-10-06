"""会议助手 - LangGraph 流水线定义

使用 LangGraph StateGraph 统一 4 条处理路径:
  local_audio / local_video / url_audio / url_video

图结构 (M1 — Function Calling):
    START → extract_audio → preprocess → asr_transcribe → text_clean
          → agent_loop → format_output → END

agent_loop 节点通过 Function Calling 让 LLM 自主决策:
    1. 调用 classify_recording 工具 → 识别录音类型
    2. 调用 analyze_meeting/interview/general 工具 → 生成分析结果
    若 Tool Calling 失败，自动降级为传统硬编码路由。
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from loguru import logger

from config import settings
from agent.context import get_context_manager

# M11: LangSmith 可观测性 (模块加载时自动初始化)
import agent.tracing  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# State 定义
# ═══════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    """LangGraph 流水线状态

    贯穿图的所有节点，每个节点读取所需字段并返回部分更新。
    """

    # ── 输入 ──
    source: str                          # 文件路径 or URL
    # local_audio | local_video | url_audio | url_video
    source_type: str
    asr_model: str | None                # 用户选择的 ASR 模型 (None=自动)
    llm_model: str | None                # 用户选择的 LLM 模型 (None=自动)
    task_id: str                         # 任务 ID (用于进度回调和结果保存)

    # ── 中间产物 ──
    audio_path: str | None               # 从视频提取的音频路径
    preprocessed_path: str | None        # 预处理后的 WAV 路径
    asr_input: str | None                # ASR 输入 (路径或 URL)
    transcript: Any                      # ASR 原始 Transcript 对象
    transcript_text: str                 # 转写文本 (清洗后)
    asr_duration: float                  # ASR 耗时 (秒)
    rec_type: str                        # meeting | interview | general | video

    # ── 分析结果 ──
    output_text: str                     # 最终 Markdown 输出
    result_path: str | None              # 结果文件路径

    # ── 元信息 ──
    audio_stem: str                      # 音频文件名 (不含后缀)
    video_title: str | None              # 视频标题 (URL 视频)
    llm_start: float                     # LLM 分析开始时间戳
    pipeline_start: float                 # 流水线开始时间戳
    error: str | None                    # 错误信息


# ═══════════════════════════════════════════════════════════════
# 进度回调 & 结果处理 & 流式输出 注册表
# ═══════════════════════════════════════════════════════════════
# 避免在 LangGraph state 中存储不可序列化的 callable，
# 改为通过 task_id 查找注册的外部回调函数。

_progress_registry: dict[str, Callable] = {}
_result_handler_registry: dict[str, Callable] = {}
# M9: 流式输出事件队列 (task_id -> asyncio.Queue)
_stream_queue_registry: dict[str, asyncio.Queue] = {}


def register_stream_queue(task_id: str, queue: asyncio.Queue):
    """注册流式输出事件队列"""
    _stream_queue_registry[task_id] = queue
    logger.debug(f"[Stream] 注册队列: task={task_id}")


def unregister_stream_queue(task_id: str):
    """注销流式输出事件队列"""
    _stream_queue_registry.pop(task_id, None)
    logger.debug(f"[Stream] 注销队列: task={task_id}")


async def _push_stream_event(task_id: str, event_type: str, data: dict):
    """推送流式事件到队列 (如果已注册)"""
    queue = _stream_queue_registry.get(task_id)
    if queue:
        await queue.put({"type": event_type, "data": data})


def register_progress_cb(task_id: str, cb: Callable) -> None:
    """注册任务进度回调"""
    _progress_registry[task_id] = cb


def unregister_task(task_id: str) -> None:
    """清理任务相关的注册表项"""
    _progress_registry.pop(task_id, None)
    _result_handler_registry.pop(task_id, None)


def register_result_handler(task_id: str, handler: Callable) -> None:
    """注册任务结果保存处理器"""
    _result_handler_registry[task_id] = handler


def _notify(task_id: str, **kw) -> None:
    """通知进度更新"""
    cb = _progress_registry.get(task_id)
    if cb:
        cb(**kw)


def _build_timing(asr_duration: float, llm_start: float) -> dict:
    """构建耗时信息字典"""
    return {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "asr_duration": asr_duration,
        "llm_duration": time.time() - llm_start,
        "total_duration": time.time() - (llm_start - asr_duration),
    }


# ═══════════════════════════════════════════════════════════════
# 节点函数 (async)
# ═══════════════════════════════════════════════════════════════


async def extract_audio(state: AgentState) -> dict:
    """节点: 从视频提取音频 (音频源直接跳过)"""
    source_type = state["source_type"]
    source = state["source"]
    task_id = state["task_id"]

    if source_type in ("local_video", "url_video"):
        from agent.video_client import extract_audio_from_video

        _notify(task_id, step="正在提取音频...", progress=5,
                source_type=source_type, step_idx=0)
        logger.info(f"[任务 {task_id}] 从视频提取音频...")

        filename_stem = state["audio_stem"]
        updates: dict = {}

        # URL 视频: 获取视频元信息，用标题作为文件名
        if source_type == "url_video":
            from agent.video_client import get_video_info
            from agent.cache import sanitize_stem

            video_info = await get_video_info(source)
            if video_info:
                title = video_info.get("title", "")
                if title:
                    new_stem = sanitize_stem(title)
                    updates["video_title"] = title
                    updates["audio_stem"] = new_stem
                    filename_stem = new_stem
                    duration = video_info.get("duration")
                    _notify(task_id, video_title=title, step_idx=0,
                            audio_duration=duration or 0)
                    logger.info(f"[任务 {task_id}] 视频标题: {title}")
            else:
                logger.warning(f"[任务 {task_id}] 无法获取视频信息，使用 URL stem")

        # 下载进度回调: 实时通知前端下载百分比
        def _dl_progress(percent: float):
            _notify(task_id,
                    step=f"正在下载视频... {percent:.0f}%",
                    progress=5, step_idx=0,
                    download_progress=round(percent, 1))

        audio_path = await extract_audio_from_video(
            source, filename_stem=filename_stem,
            progress_cb=_dl_progress)

        logger.info(f"[任务 {task_id}] 音频提取完成: {audio_path.name}")
        # 获取提取后的音频文件大小
        audio_file_size = audio_path.stat().st_size if audio_path.exists() else 0
        _notify(task_id, step="音频提取完成", progress=20,
                step_idx=0, file_size=audio_file_size)
        updates["audio_path"] = str(audio_path)
        return updates

    # 音频源: 无需提取
    _notify(task_id, step="音频源，跳过提取", progress=5,
            source_type=source_type, step_idx=0)
    return {}


async def preprocess(state: AgentState) -> dict:
    """节点: 预处理音频为 16kHz 单声道 WAV (URL 音频跳过)"""
    source_type = state["source_type"]
    task_id = state["task_id"]

    # URL 音频直接传给 ASR，跳过预处理
    if source_type == "url_audio":
        _notify(task_id, step="URL 音频，跳过预处理", progress=35, step_idx=1)
        return {"asr_input": state["source"]}

    # 本地音频 or 从视频提取的音频 → 预处理
    audio_path = state.get("audio_path") or state["source"]
    _notify(task_id, step="正在预处理音频...", progress=22, step_idx=1)
    logger.info(f"[任务 {task_id}] 预处理音频: {Path(audio_path).name}")

    from utils.audio import preprocess_audio

    preprocessed, audio_duration = preprocess_audio(
        audio_path, sample_rate=settings.asr_sample_rate)

    logger.info(
        f"[任务 {task_id}] 预处理完成: {preprocessed.name}, 时长 {audio_duration:.1f}s")

    # 清理提取的临时音频文件 (视频路径)
    if state.get("audio_path"):
        try:
            Path(state["audio_path"]).unlink()
        except Exception:
            pass

    _notify(task_id, step="预处理完成，准备转写...", progress=35,
            step_idx=1, audio_duration=round(audio_duration, 1))
    return {
        "preprocessed_path": str(preprocessed),
        "asr_input": str(preprocessed),
    }


async def asr_transcribe(state: AgentState) -> dict:
    """节点: ASR 语音转写"""
    task_id = state["task_id"]
    asr_input = state["asr_input"]
    audio_stem = state["audio_stem"]

    logger.info(f"[任务 {task_id}] 开始 ASR 转写...")

    from agent.asr_client import ASRClient
    from agent.cache import load_asr_cache

    asr = ASRClient(models=[state["asr_model"]]
                    if state.get("asr_model") else None)
    asr_model = asr.model

    # 检查 ASR 缓存
    cached_text = load_asr_cache(audio_stem, asr_model)
    if cached_text is not None:
        logger.info(f"[任务 {task_id}] ASR 缓存命中，跳过转写")
        # 解析缓存的格式化文本为 Sentence 对象
        from models.schemas import Transcript, Sentence
        import re
        sentences = []
        for line in cached_text.split('\n'):
            if not line.strip():
                continue
            # 解析格式: [start -> end] speaker: text
            match = re.match(r'\[([^\]]+)\]\s*([^:]+):\s*(.*)', line)
            if match:
                time_range, speaker, text = match.groups()
                # 简单解析时间（缓存中不需要精确时间）
                sentences.append(
                    Sentence(text=text.strip(), speaker=speaker.strip()))
            else:
                # 如果格式不匹配，直接作为纯文本
                sentences.append(Sentence(text=line.strip(), speaker=None))
        cached_transcript = Transcript(sentences=sentences, duration_ms=0)
        _notify(task_id, step="ASR 缓存命中", progress=55,
                asr_model_used=asr_model, step_idx=2,
                asr_sentences=len(sentences),
                asr_duration=0.0)
        return {
            "transcript": cached_transcript,
            "transcript_text": cached_text,
            "asr_duration": 0.0,
        }

    _notify(task_id, step="正在语音转写...", progress=40,
            asr_model_used=asr_model, step_idx=2)

    t0 = time.time()
    transcript = await asr.transcribe(asr_input)
    asr_duration = time.time() - t0

    transcript_text = transcript.formatted_text
    sentence_count = len(transcript.sentences)
    logger.info(
        f"[任务 {task_id}] ASR 转写完成, "
        f"共 {sentence_count} 句, 耗时 {asr_duration:.1f}s"
    )

    # 清理预处理临时文件
    if state.get("preprocessed_path"):
        try:
            Path(state["preprocessed_path"]).unlink()
        except Exception:
            pass

    _notify(task_id, step=f"转写完成 ({sentence_count} 句)",
            progress=55, step_idx=2,
            asr_sentences=sentence_count,
            asr_duration=round(asr_duration, 1),
            asr_chars=len(transcript_text))

    return {
        "transcript": transcript,
        "transcript_text": transcript_text,
        "asr_duration": asr_duration,
    }


async def text_clean(state: AgentState) -> dict:
    """节点: 文本预处理 (去语气词、合并短句)"""
    task_id = state["task_id"]
    transcript_text = state["transcript_text"]

    if not transcript_text.strip():
        raise ValueError("ASR 转写结果为空，请检查音频文件是否包含有效语音")

    from agent.text_processor import TextProcessor
    from agent.cache import save_asr_cache

    processor = TextProcessor()

    # 使用 ASR 返回的完整 Transcript 对象 (保留说话人/时间戳)
    raw_transcript = state["transcript"]
    cleaned = processor.clean(raw_transcript)
    cleaned_text = cleaned.formatted_text

    # 保存 ASR 缓存
    asr_path = save_asr_cache(cleaned_text, state["audio_stem"])
    cleaned_count = len(cleaned.sentences)
    _notify(task_id, asr_result_path=str(asr_path),
            step=f"文本清洗完成 ({cleaned_count} 句)", progress=60,
            step_idx=3, cleaned_sentences=cleaned_count)
    logger.info(f"[任务 {task_id}] 文本预处理完成：{cleaned_count} 句")

    return {"transcript_text": cleaned_text}


async def classify(state: AgentState) -> dict:
    """节点: 录音类型分类 (视频源跳过，直接标记为 general)"""
    task_id = state["task_id"]
    source_type = state["source_type"]

    # 视频源不走分类，直接归为 general
    if source_type in ("local_video", "url_video"):
        logger.info(f"[任务 {task_id}] 视频源，跳过分类 (general)")
        _notify(task_id, step="视频源，跳过分类", progress=65,
                step_idx=4, rec_type="general")
        return {"rec_type": "general"}

    _notify(task_id, step="正在识别录音类型...", progress=62, step_idx=4)
    logger.info(f"[任务 {task_id}] 开始分类...")

    from agent.classifier import classify_recording

    rec_type = await classify_recording(state["transcript_text"])
    logger.info(f"[任务 {task_id}] 分类结果: {rec_type}")
    _notify(task_id, step=f"分类完成: {rec_type}", progress=66,
            step_idx=4, rec_type=rec_type)

    return {"rec_type": rec_type}


async def analyze_dispatch(state: AgentState) -> dict:
    """节点: 根据分类结果分发到对应分析策略"""
    task_id = state["task_id"]
    rec_type = state["rec_type"]
    transcript_text = state["transcript_text"]
    llm_model = state.get("llm_model")

    type_labels = {
        "meeting": "会议录音",
        "interview": "面试录音",
        "general": "其他录音",
        "video": "视频内容",
    }
    _notify(
        task_id,
        step=f"正在分析 ({type_labels.get(rec_type, rec_type)})...",
        progress=70,
        step_idx=5,
        llm_model_used=llm_model or "自动",
    )

    llm_start = time.time()
    timing_info = _build_timing(state["asr_duration"], llm_start)
    audio_stem = state["audio_stem"]

    # ── 会议录音 ──
    if rec_type == "meeting":
        from agent.llm_client import LLMClient
        from agent.prompts import MEETING_ANALYSIS_PROMPT
        from agent.pipeline import format_markdown
        from agent.cache import save_summary_cache
        from models.schemas import MeetingResult

        llm = LLMClient(models=[llm_model] if llm_model else None)
        prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript_text)
        data = await llm.chat_json(prompt)
        result = MeetingResult.model_validate(data)
        output_text = format_markdown(
            result, event_time=None, timing_info=timing_info)
        save_summary_cache(output_text, audio_stem, rec_type="meeting")
        llm_dur = time.time() - llm_start
        _notify(task_id, step="会议分析完成", progress=90,
                step_idx=5, llm_duration=round(llm_dur, 1))

        return {
            "output_text": output_text,
            "llm_start": llm_start,
        }

    # ── 面试录音 ──
    if rec_type == "interview":
        from agent.pipeline import (
            analyze_interview,
            format_interview_questions,
            format_interview_analysis,
        )
        from pathlib import Path

        result_raw = await analyze_interview(transcript_text, llm_model=llm_model)
        q_md = format_interview_questions(
            result_raw["q_data"], None, timing_info)
        a_md = format_interview_analysis(
            result_raw["a_data"], None, timing_info)

        # 面试保存为目录
        result_dir = (
            settings.output_dir
            / "summary"
            / "interview"
            / f"{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "question.md").write_text(q_md, encoding="utf-8")
        (result_dir / "analyze.md").write_text(a_md, encoding="utf-8")
        llm_dur = time.time() - llm_start
        _notify(task_id, step="面试分析完成", progress=90,
                step_idx=5, llm_duration=round(llm_dur, 1))

        return {
            "output_text": a_md,
            "result_path": str(result_dir / "analyze.md"),
            "llm_start": llm_start,
        }

    # ── 通用 / 视频 ──
    from agent.pipeline import analyze_general, format_general_result
    from agent.cache import load_summary_cache, save_summary_cache

    cached = load_summary_cache(audio_stem, rec_type=rec_type)
    if cached is not None:
        output_text = cached
        logger.info(f"[任务 {task_id}] 使用 Summary 缓存")
        _notify(task_id, step="使用缓存，跳过 LLM 分析", progress=90,
                step_idx=5, cache_hit=True, llm_duration=0)
    else:
        data = await analyze_general(transcript_text, llm_model=llm_model)
        output_text = format_general_result(data, None, timing_info)
        save_summary_cache(output_text, audio_stem, rec_type=rec_type)
        llm_dur = time.time() - llm_start
        _notify(task_id, step="通用分析完成", progress=90,
                step_idx=5, cache_hit=False, llm_duration=round(llm_dur, 1))

    return {
        "output_text": output_text,
        "llm_start": llm_start,
    }


async def format_output(state: AgentState) -> dict:
    """节点: 最终输出 — 保存结果文件 (统一由本节点负责)"""
    task_id = state["task_id"]

    # interview 分支已在 analyze_dispatch 中直接写入文件并设置 result_path
    if not state.get("result_path"):
        # meeting / general / video 分支: 通过 result_handler 保存结果
        result_handler = _result_handler_registry.get(task_id)
        if result_handler and state.get("output_text"):
            source_type = state["source_type"]
            rec_type = state["rec_type"]

            if rec_type == "meeting":
                subdir = "meeting"
            elif source_type in ("local_video", "url_video"):
                subdir = "video"
            else:
                subdir = "other"

            result_handler(task_id, state["audio_stem"],
                           subdir, state["output_text"])

    # 计算总耗时并发送完成通知 (所有分支统一处理)
    pipeline_start = state.get("pipeline_start", 0)
    if pipeline_start:
        total_dur = time.time() - pipeline_start
        _notify(task_id, step="分析完成", progress=100,
                step_idx=6, total_duration=round(total_dur, 1))
    else:
        _notify(task_id, step="分析完成", progress=100, step_idx=6)

    logger.info(f"[任务 {task_id}] 流水线执行完成")
    return {}


# ═══════════════════════════════════════════════════════════════
# Agent Loop (Function Calling)
# ═══════════════════════════════════════════════════════════════


AGENT_SYSTEM_PROMPT = """你是一个智能会议分析 Agent。你的任务是分析音视频转写文本并生成结构化报告。

## 工作流程

1. **第一步 — 分类**: 调用 `classify_recording` 工具，根据转写内容判断录音类型
   - meeting: 多人工作会议，有议题讨论和决策
   - interview: 面试场景，有提问-回答模式
   - general: 其他类型（讲座、访谈、日常交流等）

2. **第二步 — 分析**: 根据分类结果，调用对应的分析工具
   - meeting → `analyze_meeting`
   - interview → `analyze_interview`
   - general → `analyze_general`

## 规则

- 必须先分类，再分析，不可跳过任何一步
- 分类依据内容特征判断，不要猜测
- 将转写文本完整传给分析工具，不要截断
"""


async def agent_loop(state: AgentState) -> dict:
    """节点: Agent 智能分析循环 (Function Calling)

    LLM 通过 Tool Use 自主决策:
      1. classify_recording → 识别录音类型
      2. analyze_meeting/interview/general → 生成分析结果

    若 Tool Calling 失败（模型不支持等），自动降级为传统硬编码路由。
    """
    task_id = state["task_id"]
    transcript_text = state["transcript_text"]
    source_type = state["source_type"]
    llm_model = state.get("llm_model")

    llm_start = time.time()
    timing_info = _build_timing(state["asr_duration"], llm_start)

    # ── 视频源跳过分类，直接走传统路径 (避免浪费 Tool 调用) ──
    if source_type in ("local_video", "url_video"):
        logger.info(f"[任务 {task_id}] 视频源，跳过 Agent 分类 (general)")
        _notify(task_id, step="视频源，跳过分类", progress=65,
                step_idx=4, rec_type="general")
        result = await _fallback_analyze(state, "general", llm_start, timing_info)
        result["llm_start"] = llm_start
        return result

    _notify(task_id, step="正在智能分析 (Function Calling)...",
            progress=62, step_idx=4)

    from agent.tools import TOOL_SCHEMAS, TOOL_REGISTRY, ToolContext
    from agent.llm_client import LLMClient

    # ── M3: Token 级别的上下文管理 ──
    ctx_mgr = get_context_manager()
    transcript_tokens = ctx_mgr.count_tokens(transcript_text)
    logger.info(f"[任务 {task_id}] Agent Loop 启动: "
                f"{len(transcript_text)} 字符, {transcript_tokens} tokens, "
                f"{len(TOOL_SCHEMAS)} 个工具")

    # 长文本截断 (为工具调用预留足够上下文空间)
    max_transcript_tokens = 80_000  # 为 system prompt + tool calls 预留空间
    if transcript_tokens > max_transcript_tokens:
        logger.warning(
            f"[任务 {task_id}] 转写文本过长: {transcript_tokens} tokens, "
            f"截断到 {max_transcript_tokens} tokens"
        )
        transcript_text = ctx_mgr.truncate_to_tokens(
            transcript_text, max_transcript_tokens
        )

    llm = LLMClient(models=[llm_model] if llm_model else None)

    # 构建工具上下文
    tool_ctx = ToolContext(
        transcript_text=transcript_text,
        audio_stem=state["audio_stem"],
        llm_model=llm_model,
        timing_info=timing_info,
        task_id=task_id,
    )

    # 初始消息
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下转写文本:\n\n{transcript_text}"},
    ]

    # ── Agent 循环: LLM 自主决策调用工具 ──
    max_iterations = 10
    output_text = ""
    rec_type = "general"

    try:
        for iteration in range(max_iterations):
            logger.debug(
                f"[任务 {task_id}] Agent 迭代 {iteration + 1}/{max_iterations}")

            response = await llm.chat_with_tools(
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            tool_calls = response.get("tool_calls")

            # ── 无 Tool 调用: LLM 给出最终响应 ──
            if not tool_calls:
                content = response.get("content", "")
                if output_text:
                    # 已有分析工具结果，直接返回
                    break
                # LLM 未调用工具就返回了文本，降级到传统路径
                logger.warning(f"[任务 {task_id}] LLM 未调用工具，降级到传统路由")
                rec_type = await _fallback_classify(state)
                _notify(task_id, step=f"分类完成: {rec_type}", progress=66,
                        step_idx=4, rec_type=rec_type)
                result = await _fallback_analyze(state, rec_type, llm_start, timing_info)
                result["rec_type"] = rec_type
                result["llm_start"] = llm_start
                return result

            # ── 有 Tool 调用: 执行工具并继续循环 ──
            messages.append(response)  # assistant + tool_calls

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                tc_id = tc["id"]

                logger.info(
                    f"[任务 {task_id}] Tool Call: {fn_name}({list(fn_args.keys())})")
                _notify(task_id, step=f"Agent 调用: {fn_name}",
                        progress=65, step_idx=4)

                tool_fn = TOOL_REGISTRY.get(fn_name)
                if not tool_fn:
                    tool_result = {"error": f"未知工具: {fn_name}"}
                    logger.warning(f"[任务 {task_id}] 未知工具: {fn_name}")
                else:
                    tool_result = await tool_fn(fn_args, tool_ctx)
                    logger.info(f"[任务 {task_id}] Tool {fn_name} 执行完成")

                # 收集关键结果
                if "rec_type" in tool_result:
                    rec_type = tool_result["rec_type"]
                    _notify(task_id, rec_type=rec_type, step_idx=4)
                if "output_text" in tool_result:
                    output_text = tool_result["output_text"]

                # 追加 tool 结果到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            # 分析工具返回了最终结果，结束循环
            if output_text:
                _notify(task_id, step=f"{rec_type} 分析完成",
                        progress=90, step_idx=4,
                        llm_duration=round(time.time() - llm_start, 1))
                break
        else:
            # 达到最大迭代次数
            logger.warning(f"[任务 {task_id}] Agent 循环达到上限 ({max_iterations})")
            if not output_text:
                rec_type = await _fallback_classify(state)
                result = await _fallback_analyze(state, rec_type, llm_start, timing_info)
                result["rec_type"] = rec_type
                result["llm_start"] = llm_start
                return result

    except Exception as e:
        logger.error(f"[任务 {task_id}] Agent Loop 异常: {e}，降级到传统路由")
        rec_type = await _fallback_classify(state)
        result = await _fallback_analyze(state, rec_type, llm_start, timing_info)
        result["rec_type"] = rec_type
        result["llm_start"] = llm_start
        return result

    return {
        "rec_type": rec_type,
        "output_text": output_text,
        "llm_start": llm_start,
    }


# ── 降级函数: 传统分类 + 分析 (当 Tool Calling 失败时使用) ──


async def _fallback_classify(state: AgentState) -> str:
    """降级分类: 使用传统 classifier.py (非 Tool Calling)"""
    task_id = state["task_id"]
    source_type = state["source_type"]

    if source_type in ("local_video", "url_video"):
        return "general"

    _notify(task_id, step="正在识别录音类型...", progress=62, step_idx=4)
    from agent.classifier import classify_recording
    rec_type = await classify_recording(state["transcript_text"])
    logger.info(f"[任务 {task_id}] 降级分类结果: {rec_type}")
    _notify(task_id, step=f"分类完成: {rec_type}", progress=66,
            step_idx=4, rec_type=rec_type)
    return rec_type


async def _fallback_analyze(
    state: AgentState, rec_type: str,
    llm_start: float, timing_info: dict,
) -> dict:
    """降级分析: 使用传统 pipeline.py 函数 (非 Tool Calling)"""
    task_id = state["task_id"]
    transcript_text = state["transcript_text"]
    llm_model = state.get("llm_model")
    audio_stem = state["audio_stem"]

    type_labels = {
        "meeting": "会议录音", "interview": "面试录音",
        "general": "其他录音", "video": "视频内容",
    }
    _notify(task_id,
            step=f"正在分析 ({type_labels.get(rec_type, rec_type)})...",
            progress=70, step_idx=5,
            llm_model_used=llm_model or "自动")

    # ── 会议 ──
    if rec_type == "meeting":
        from agent.cache import load_summary_cache, save_summary_cache
        from agent.llm_client import LLMClient
        from agent.prompts import MEETING_ANALYSIS_PROMPT
        from agent.pipeline import format_markdown
        from models.schemas import MeetingResult

        # 检查 Summary 缓存
        cached = load_summary_cache(audio_stem, rec_type=rec_type)
        if cached is not None:
            logger.info(f"[任务 {task_id}] 使用 Summary 缓存")
            # 流式输出缓存内容
            await _push_stream_event(task_id, "start", {"rec_type": rec_type})
            for char in cached:
                await _push_stream_event(task_id, "token", {"content": char})
            await _push_stream_event(task_id, "done", {"output_text": cached})
            _notify(task_id, step="会议分析完成 (缓存)", progress=90,
                    step_idx=5, cache_hit=True, llm_duration=0.0)
            return {"output_text": cached}

        llm = LLMClient(models=[llm_model] if llm_model else None)
        prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript_text)

        # M9: 检查是否启用流式输出
        has_stream_queue = task_id in _stream_queue_registry
        if has_stream_queue:
            await _push_stream_event(task_id, "start", {"rec_type": rec_type})
            # 流式调用 LLM
            full_content = ""
            async for token in llm.chat_stream(prompt):
                full_content += token
                await _push_stream_event(task_id, "token", {"content": token})
            # 解析 JSON 并格式化
            data = llm._extract_json(full_content)
            result = MeetingResult.model_validate(data)
            output_text = format_markdown(
                result, event_time=None, timing_info=timing_info)
            await _push_stream_event(task_id, "done", {"output_text": output_text})
        else:
            # 传统非流式调用
            data = await llm.chat_json(prompt)
            result = MeetingResult.model_validate(data)
            output_text = format_markdown(
                result, event_time=None, timing_info=timing_info)

        save_summary_cache(output_text, audio_stem, rec_type="meeting")
        llm_dur = time.time() - llm_start
        _notify(task_id, step="会议分析完成", progress=90,
                step_idx=5, llm_duration=round(llm_dur, 1))
        return {"output_text": output_text}

    # ── 面试 ──
    if rec_type == "interview":
        from agent.pipeline import (
            analyze_interview, format_interview_questions,
            format_interview_analysis,
        )

        result_raw = await analyze_interview(
            transcript_text, llm_model=llm_model)
        q_md = format_interview_questions(
            result_raw["q_data"], None, timing_info)
        a_md = format_interview_analysis(
            result_raw["a_data"], None, timing_info)

        result_dir = (
            settings.output_dir / "summary" / "interview"
            / f"{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "question.md").write_text(q_md, encoding="utf-8")
        (result_dir / "analyze.md").write_text(a_md, encoding="utf-8")
        llm_dur = time.time() - llm_start
        _notify(task_id, step="面试分析完成", progress=90,
                step_idx=5, llm_duration=round(llm_dur, 1))
        return {"output_text": a_md, "result_path": str(result_dir / "analyze.md")}

    # ── 通用 / 视频 ──
    from agent.pipeline import analyze_general, format_general_result
    from agent.cache import load_summary_cache, save_summary_cache

    cached = load_summary_cache(audio_stem, rec_type=rec_type)
    if cached is not None:
        output_text = cached
        logger.info(f"[任务 {task_id}] 使用 Summary 缓存")
        # 流式输出缓存内容
        await _push_stream_event(task_id, "start", {"rec_type": rec_type})
        for char in cached:
            await _push_stream_event(task_id, "token", {"content": char})
        await _push_stream_event(task_id, "done", {"output_text": cached})
        _notify(task_id, step="使用缓存，跳过 LLM 分析", progress=90,
                step_idx=5, cache_hit=True, llm_duration=0)
    else:
        # M9: 检查是否启用流式输出
        has_stream_queue = task_id in _stream_queue_registry
        if has_stream_queue:
            from agent.llm_client import LLMClient
            from agent.prompts import GENERAL_ANALYSIS_PROMPT

            await _push_stream_event(task_id, "start", {"rec_type": rec_type})
            llm = LLMClient(models=[llm_model] if llm_model else None)
            prompt = GENERAL_ANALYSIS_PROMPT.format(transcript=transcript_text)
            # 流式调用 LLM
            full_content = ""
            async for token in llm.chat_stream(prompt):
                full_content += token
                await _push_stream_event(task_id, "token", {"content": token})
            # 解析 JSON 并格式化
            data = llm._extract_json(full_content)
            output_text = format_general_result(data, None, timing_info)
            await _push_stream_event(task_id, "done", {"output_text": output_text})
        else:
            # 传统非流式调用
            data = await analyze_general(transcript_text, llm_model=llm_model)
            output_text = format_general_result(data, None, timing_info)

        save_summary_cache(output_text, audio_stem, rec_type=rec_type)
        llm_dur = time.time() - llm_start
        _notify(task_id, step="通用分析完成", progress=90,
                step_idx=5, cache_hit=False, llm_duration=round(llm_dur, 1))

    return {"output_text": output_text}


# ═══════════════════════════════════════════════════════════════
# 构建 StateGraph
# ═══════════════════════════════════════════════════════════════


def _build_graph():
    """构建并编译 LangGraph 状态图

    M1 重构: classify + analyze_* 合并为 agent_loop (Function Calling)
    降级路径: agent_loop 异常时自动回退到传统硬编码路由
    """
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("extract_audio", extract_audio)
    builder.add_node("preprocess", preprocess)
    builder.add_node("asr_transcribe", asr_transcribe)
    builder.add_node("text_clean", text_clean)
    builder.add_node("agent_loop", agent_loop)       # Function Calling 节点
    builder.add_node("format_output", format_output)

    # 线性边
    builder.add_edge(START, "extract_audio")
    builder.add_edge("extract_audio", "preprocess")
    builder.add_edge("preprocess", "asr_transcribe")
    builder.add_edge("asr_transcribe", "text_clean")
    builder.add_edge("text_clean", "agent_loop")
    builder.add_edge("agent_loop", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile()


# 编译后的图对象 (模块级单例)
graph = _build_graph()


# ═══════════════════════════════════════════════════════════════
# 同步入口 (供 server.py 后台线程调用)
# ═══════════════════════════════════════════════════════════════


async def run_pipeline(initial_state: dict) -> dict:
    """异步执行完整流水线

    Args:
        initial_state: 初始 AgentState 字典

    Returns:
        最终状态字典
    """
    return await graph.ainvoke(initial_state)


def invoke_pipeline(initial_state: dict) -> dict:
    """同步执行完整流水线 (供后台线程调用)

    内部通过 asyncio.run() 驱动异步图执行。
    每个节点中创建的 async 客户端 (ASRClient/LLMClient)
    在各自的事件循环中独立运行，互不干扰。
    """
    task_id = initial_state.get("task_id", "")
    return asyncio.run(run_pipeline(initial_state))
