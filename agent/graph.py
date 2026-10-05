"""会议助手 - LangGraph 流水线定义

使用 LangGraph StateGraph 统一 4 条处理路径:
  local_audio / local_video / url_audio / url_video

图结构:
    START → extract_audio → preprocess → asr_transcribe → text_clean
          → classify → [route] → analyze_* → format_output → END
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from loguru import logger

from config import settings


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
# 进度回调 & 结果处理 注册表
# ═══════════════════════════════════════════════════════════════
# 避免在 LangGraph state 中存储不可序列化的 callable，
# 改为通过 task_id 查找注册的外部回调函数。

_progress_registry: dict[str, Callable] = {}
_result_handler_registry: dict[str, Callable] = {}


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

        _notify(task_id, step="正在提取音频...", progress=10,
                source_type=source_type)
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
                    _notify(task_id, video_title=title)
                    logger.info(f"[任务 {task_id}] 视频标题: {title}")
            else:
                logger.warning(f"[任务 {task_id}] 无法获取视频信息，使用 URL stem")

        audio_path = await extract_audio_from_video(
            source, filename_stem=filename_stem)

        logger.info(f"[任务 {task_id}] 音频提取完成: {audio_path.name}")
        _notify(task_id, step="音频提取完成", progress=22)
        updates["audio_path"] = str(audio_path)
        return updates

    # 音频源: 无需提取
    _notify(task_id, step="音频源，跳过提取", progress=5,
            source_type=source_type)
    return {}


async def preprocess(state: AgentState) -> dict:
    """节点: 预处理音频为 16kHz 单声道 WAV (URL 音频跳过)"""
    source_type = state["source_type"]
    task_id = state["task_id"]

    # URL 音频直接传给 ASR，跳过预处理
    if source_type == "url_audio":
        _notify(task_id, step="URL 音频，跳过预处理", progress=38)
        return {"asr_input": state["source"]}

    # 本地音频 or 从视频提取的音频 → 预处理
    audio_path = state.get("audio_path") or state["source"]
    _notify(task_id, step="正在预处理音频...", progress=25)
    logger.info(f"[任务 {task_id}] 预处理音频: {Path(audio_path).name}")

    from utils.audio import preprocess_audio

    preprocessed = preprocess_audio(
        audio_path, sample_rate=settings.asr_sample_rate)

    logger.info(f"[任务 {task_id}] 预处理完成: {preprocessed.name}")

    # 清理提取的临时音频文件 (视频路径)
    if state.get("audio_path"):
        try:
            Path(state["audio_path"]).unlink()
        except Exception:
            pass

    _notify(task_id, step="预处理完成，准备转写...", progress=38)
    return {
        "preprocessed_path": str(preprocessed),
        "asr_input": str(preprocessed),
    }


async def asr_transcribe(state: AgentState) -> dict:
    """节点: ASR 语音转写"""
    task_id = state["task_id"]
    asr_input = state["asr_input"]

    logger.info(f"[任务 {task_id}] 开始 ASR 转写...")

    from agent.asr_client import ASRClient

    asr = ASRClient(models=[state["asr_model"]]
                    if state.get("asr_model") else None)
    asr_model = asr.model
    _notify(task_id, step="正在语音转写...", progress=40,
            asr_model_used=asr_model)

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
            progress=50, asr_sentences=sentence_count,
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
            step=f"文本清洗完成 ({cleaned_count} 句)", progress=58,
            cleaned_sentences=cleaned_count)
    logger.info(f"[任务 {task_id}] 文本预处理完成：{cleaned_count} 句")

    return {"transcript_text": cleaned_text}


async def classify(state: AgentState) -> dict:
    """节点: 录音类型分类 (视频源跳过，直接标记为 general)"""
    task_id = state["task_id"]
    source_type = state["source_type"]

    # 视频源不走分类，直接归为 general
    if source_type in ("local_video", "url_video"):
        logger.info(f"[任务 {task_id}] 视频源，跳过分类 (general)")
        _notify(task_id, step="视频源，跳过分类", progress=62,
                rec_type="general")
        return {"rec_type": "general"}

    _notify(task_id, step="正在识别录音类型...", progress=62)
    logger.info(f"[任务 {task_id}] 开始分类...")

    from agent.classifier import classify_recording

    rec_type = await classify_recording(state["transcript_text"])
    logger.info(f"[任务 {task_id}] 分类结果: {rec_type}")
    _notify(task_id, step=f"分类完成: {rec_type}", progress=66,
            rec_type=rec_type)

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
                llm_duration=round(llm_dur, 1))

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
                llm_duration=round(llm_dur, 1))

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
                cache_hit=True, llm_duration=0)
    else:
        data = await analyze_general(transcript_text, llm_model=llm_model)
        output_text = format_general_result(data, None, timing_info)
        save_summary_cache(output_text, audio_stem, rec_type=rec_type)
        llm_dur = time.time() - llm_start
        _notify(task_id, step="通用分析完成", progress=90,
                cache_hit=False, llm_duration=round(llm_dur, 1))

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
                total_duration=round(total_dur, 1))
    else:
        _notify(task_id, step="分析完成", progress=100)

    logger.info(f"[任务 {task_id}] 流水线执行完成")
    return {}


# ═══════════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════════


def route_by_type(state: AgentState) -> str:
    """根据录音类型和来源决定分析分支

    - meeting → analyze_meeting (独立会议分析)
    - interview → analyze_interview (面试逐题分析)
    - video / general → analyze_general (通用分析)
    """
    rec_type = state.get("rec_type", "general")
    source_type = state.get("source_type", "")

    if rec_type == "meeting":
        return "analyze_meeting"
    if rec_type == "interview":
        return "analyze_interview"
    # 视频源始终走通用分析; 其他 general 也走通用
    return "analyze_general"


# ═══════════════════════════════════════════════════════════════
# 构建 StateGraph
# ═══════════════════════════════════════════════════════════════


def _build_graph():
    """构建并编译 LangGraph 状态图"""
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("extract_audio", extract_audio)
    builder.add_node("preprocess", preprocess)
    builder.add_node("asr_transcribe", asr_transcribe)
    builder.add_node("text_clean", text_clean)
    builder.add_node("classify", classify)
    builder.add_node("analyze_meeting", analyze_dispatch)
    builder.add_node("analyze_interview", analyze_dispatch)
    builder.add_node("analyze_general", analyze_dispatch)
    builder.add_node("format_output", format_output)

    # 线性边
    builder.add_edge(START, "extract_audio")
    builder.add_edge("extract_audio", "preprocess")
    builder.add_edge("preprocess", "asr_transcribe")
    builder.add_edge("asr_transcribe", "text_clean")
    builder.add_edge("text_clean", "classify")

    # 条件路由: classify → 分析分支
    builder.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "analyze_meeting": "analyze_meeting",
            "analyze_interview": "analyze_interview",
            "analyze_general": "analyze_general",
        },
    )

    # 所有分析分支汇聚到 format_output
    builder.add_edge("analyze_meeting", "format_output")
    builder.add_edge("analyze_interview", "format_output")
    builder.add_edge("analyze_general", "format_output")
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
