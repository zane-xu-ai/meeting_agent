"""会议助手 - Agent Tool 定义与实现

Agent 通过 Function Calling 自主决策调用以下工具:
  - classify_recording:  识别录音类型 (meeting / interview / general)
  - analyze_meeting:     会议纪要分析
  - analyze_interview:   面试逐题分析
  - analyze_general:     通用内容分析

工具注册表 (TOOL_REGISTRY) 供 agent_loop 按名称查找并执行。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from loguru import logger


# ═══════════════════════════════════════════════════════════════
# Tool Schemas (OpenAI tools 格式)
# ═══════════════════════════════════════════════════════════════

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify_recording",
            "description": (
                "根据转写文本内容判断录音类型。"
                "必须在分析之前调用。"
                "类型说明: "
                "meeting=多人工作会议，有议题讨论和决策; "
                "interview=面试场景，有提问-回答模式; "
                "general=其他类型（讲座、访谈、日常交流等）"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rec_type": {
                        "type": "string",
                        "enum": ["meeting", "interview", "general"],
                        "description": "录音类型",
                    },
                    "reason": {
                        "type": "string",
                        "description": "判断依据，简要说明",
                    },
                },
                "required": ["rec_type", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_meeting",
            "description": (
                "分析会议录音转写文本，生成结构化会议纪要。"
                "输出包含: 全文摘要、章节速览、关键决策、待办事项、关键词、参会人。"
                "仅在 classify_recording 判定为 meeting 后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transcript": {
                        "type": "string",
                        "description": "会议转写文本",
                    },
                },
                "required": ["transcript"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_interview",
            "description": (
                "分析面试录音转写文本，生成逐题评估报告。"
                "输出包含: 面试总结、综合评分、每题评分与推荐答案、改进建议。"
                "仅在 classify_recording 判定为 interview 后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transcript": {
                        "type": "string",
                        "description": "面试转写文本",
                    },
                },
                "required": ["transcript"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_general",
            "description": (
                "分析通用录音转写文本（讲座、访谈、日常交流等），生成内容总结。"
                "输出包含: 全文摘要、章节速览、关键信息、关键词。"
                "仅在 classify_recording 判定为 general 后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transcript": {
                        "type": "string",
                        "description": "转写文本",
                    },
                },
                "required": ["transcript"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_past_meetings",
            "description": (
                "搜索历史会议/面试/视频记录。"
                "当用户提到'上次'、'之前'、'历史'、'对比'、'其他会议'时调用。"
                "返回相关历史记录的摘要和关键信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询 (如: 'AI 项目讨论')",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["summary", "asr", "any"],
                        "description": "搜索来源类型 (默认 any)",
                    },
                    "rec_type": {
                        "type": "string",
                        "enum": ["meeting", "interview", "general", "any"],
                        "description": "录音类型过滤 (默认 any)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回数量 (默认 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# 工具执行上下文 (由 agent_loop 注入)
# ═══════════════════════════════════════════════════════════════


class ToolContext:
    """工具执行所需的上下文，由 agent_loop 节点构建并传入工具函数。"""

    def __init__(
        self,
        transcript_text: str,
        audio_stem: str,
        llm_model: str | None = None,
        timing_info: dict | None = None,
        task_id: str = "",
    ):
        self.transcript_text = transcript_text
        self.audio_stem = audio_stem
        self.llm_model = llm_model
        self.timing_info = timing_info or {}
        self.task_id = task_id


# ═══════════════════════════════════════════════════════════════
# Tool 实现
# ═══════════════════════════════════════════════════════════════


async def _tool_classify_recording(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """工具: 录音类型分类"""
    from agent.classifier import classify_recording

    rec_type = await classify_recording(ctx.transcript_text)
    logger.info(f"[Tool] classify_recording → {rec_type}")
    return {"rec_type": rec_type}


async def _tool_analyze_meeting(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """工具: 会议纪要分析"""
    from agent.llm_client import LLMClient
    from agent.prompts import MEETING_ANALYSIS_PROMPT
    from agent.pipeline import format_markdown
    from agent.cache import save_summary_cache
    from models.schemas import MeetingResult

    transcript = args.get("transcript", ctx.transcript_text)
    llm = LLMClient(models=[ctx.llm_model] if ctx.llm_model else None)

    prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript)
    data = await llm.chat_json(prompt)
    result = MeetingResult.model_validate(data)
    output_text = format_markdown(
        result, event_time=None, timing_info=ctx.timing_info
    )
    save_summary_cache(output_text, ctx.audio_stem, rec_type="meeting")
    logger.info("[Tool] analyze_meeting 完成")
    return {"rec_type": "meeting", "output_text": output_text}


async def _tool_analyze_interview(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """工具: 面试逐题分析"""
    from agent.pipeline import (
        analyze_interview,
        format_interview_questions,
        format_interview_analysis,
    )
    from config import settings

    transcript = args.get("transcript", ctx.transcript_text)
    result_raw = await analyze_interview(
        transcript, llm_model=ctx.llm_model
    )
    q_md = format_interview_questions(
        result_raw["q_data"], None, ctx.timing_info
    )
    a_md = format_interview_analysis(
        result_raw["a_data"], None, ctx.timing_info
    )

    # 面试保存为目录
    result_dir = (
        settings.output_dir
        / "summary"
        / "interview"
        / f"{ctx.audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "question.md").write_text(q_md, encoding="utf-8")
    (result_dir / "analyze.md").write_text(a_md, encoding="utf-8")

    logger.info("[Tool] analyze_interview 完成")
    return {"rec_type": "interview", "output_text": a_md}


async def _tool_analyze_general(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """工具: 通用内容分析"""
    from agent.pipeline import analyze_general, format_general_result
    from agent.cache import load_summary_cache, save_summary_cache

    transcript = args.get("transcript", ctx.transcript_text)
    rec_type = "general"

    # 检查缓存
    cached = load_summary_cache(ctx.audio_stem, rec_type=rec_type)
    if cached is not None:
        logger.info("[Tool] analyze_general 使用缓存")
        return {"rec_type": rec_type, "output_text": cached, "cache_hit": True}

    data = await analyze_general(transcript, llm_model=ctx.llm_model)
    output_text = format_general_result(data, None, ctx.timing_info)
    save_summary_cache(output_text, ctx.audio_stem, rec_type=rec_type)
    logger.info("[Tool] analyze_general 完成")
    return {"rec_type": rec_type, "output_text": output_text, "cache_hit": False}


# ═══════════════════════════════════════════════════════════════
# Tool 注册表
# ═══════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, Callable] = {
    "classify_recording": _tool_classify_recording,
    "analyze_meeting": _tool_analyze_meeting,
    "analyze_interview": _tool_analyze_interview,
    "analyze_general": _tool_analyze_general,
    "search_past_meetings": None,  # 由 chat 接口直接调用，非 agent_loop
}
