"""会议助手智能体 - CLI 入口

用法:
    python main.py transcript-only <音频路径或URL>   # 仅 ASR 转写
    python main.py analyze <音频路径或URL>            # 完整分析 (ASR + LLM)
    python main.py video <视频路径或URL>              # 视频总结
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.cache import (
    get_audio_stem,
    load_asr_cache,
    load_interview_cache,
    load_summary_cache,
    sanitize_stem,
    save_asr_cache,
    save_interview_cache,
    save_summary_cache,
)
from agent.pipeline import (
    analyze_general,
    analyze_interview,
    format_markdown,
)
from agent.text_processor import TextProcessor
from config import settings

# 配置 loguru
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | {message}",
)

app = typer.Typer(
    name="meeting-agent",
    help="会议助手智能体 - 音频/视频输入，输出结构化会议纪要",
)
console = Console()


# ---------- 公共: 获取音频文件时间 ----------

def get_event_time(audio_source: str) -> str | None:
    """
    获取音频文件的事件时间(文件修改时间)。

    本地文件: 返回文件修改时间字符串
    URL: 返回 None(无法获取)
    """
    if audio_source.startswith(("http://", "https://")):
        return None
    try:
        path = Path(audio_source)
        if path.exists():
            mtime = path.stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return None


# ---------- 公共: 获取 ASR 转写文本 (带缓存) ----------

async def _get_transcript_text(
    audio_source: str, audio_stem: str
) -> tuple[str, bool, float]:
    """
    获取 ASR 转写文本，优先使用缓存。

    返回: (转写文本, is_from_cache: bool, asr_duration: float)
    """
    from agent.asr_client import ASRClient
    from utils.audio import preprocess_audio

    # 1. 检查 ASR 缓存
    cached = load_asr_cache(audio_stem)
    if cached is not None:
        console.print(
            f"[green]使用 ASR 缓存: {audio_stem}_{settings.asr_models[0]}.txt[/green]")
        return cached, True, 0.0

    # 2. 缓存未命中，执行 ASR 转写
    is_url = audio_source.startswith(("http://", "https://"))
    if is_url:
        asr_input = audio_source
        preprocessed = None
    else:
        preprocessed = preprocess_audio(
            audio_source, sample_rate=settings.asr_sample_rate)
        asr_input = preprocessed

    t0 = time.time()
    try:
        asr = ASRClient()
        transcript = await asr.transcribe(asr_input)
    finally:
        if preprocessed and preprocessed.exists():
            preprocessed.unlink()
    asr_duration = time.time() - t0

    # 3. 文本预处理
    processor = TextProcessor()
    cleaned = processor.clean(transcript)
    transcript_text = cleaned.formatted_text

    # 4. 保存 ASR 缓存
    save_asr_cache(transcript_text, audio_stem)
    console.print(
        f"[green]ASR 结果已保存: {audio_stem}_{settings.asr_models[0]}.txt[/green]")

    return transcript_text, False, asr_duration


# ---------- transcript-only 命令 ----------

@app.command()
def transcript_only(
    audio: str = typer.Argument(..., help="音频文件路径或 URL"),
):
    """仅执行 ASR 转写，不进行 LLM 分析 (用于调试)"""
    if not settings.dashscope_api_key:
        console.print("[red]错误: 未配置 DASHSCOPE_API_KEY[/red]")
        raise typer.Exit(1)

    is_url = audio.startswith(("http://", "https://"))
    display_name = "远程音频" if is_url else Path(audio).name
    audio_stem = get_audio_stem(audio)

    console.print(f"[bold]转写音频:[/bold] {display_name}")

    transcript_text, from_cache, asr_duration = asyncio.run(
        _get_transcript_text(audio, audio_stem))

    console.print("\n[bold]转写结果:[/bold]")
    console.print(transcript_text)


# ---------- analyze 命令 ----------

@app.command()
def analyze(
    audio: str = typer.Argument(
        ...,
        help="音频文件路径或 URL (支持 wav/mp3/m4a/flac/ogg 等格式)",
    ),
    output_format: str = typer.Option(
        "markdown",
        "--output",
        "-o",
        help="输出格式: markdown 或 json",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细日志",
    ),
):
    """分析音频，自动识别类型并生成对应的总结"""
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time:HH:mm:ss} | {level} | {message}",
        )

    # 检查 API Key
    if not settings.dashscope_api_key:
        console.print(
            "[red]错误: 未配置 DASHSCOPE_API_KEY[/red]\n"
            "请复制 .env.example 为 .env 并填入阿里云百炼 API Key\n"
            "获取地址: https://bailian.console.aliyun.com/"
        )
        raise typer.Exit(1)

    is_url = audio.startswith(("http://", "https://"))
    display_name = "远程音频" if is_url else Path(audio).name
    audio_stem = get_audio_stem(audio)

    console.print(
        Panel(
            f"[bold]音频:[/bold] {display_name}\n"
            f"[bold]ASR 模型:[/bold] {settings.asr_models[0]}\n"
            f"[bold]LLM 模型:[/bold] {settings.llm_models[0]}",
            title="会议助手智能体",
            border_style="blue",
        )
    )

    # 获取音频文件的事件时间
    event_time = get_event_time(audio)
    total_start = time.time()

    # 1. 获取 ASR 转写文本 (带缓存)
    logger.info("[1/3] ASR 语音转写...")
    transcript_text, asr_from_cache, asr_duration = asyncio.run(
        _get_transcript_text(audio, audio_stem))

    if not transcript_text.strip():
        console.print("[red]错误: ASR 转写结果为空，请检查音频文件是否包含有效语音[/red]")
        raise typer.Exit(1)

    # 2. 录音类型分类
    logger.info("[2/3] 录音类型分类...")
    from agent.classifier import classify_recording
    rec_type = asyncio.run(classify_recording(transcript_text))
    type_labels = {"meeting": "会议录音", "interview": "面试录音", "general": "其他录音"}
    console.print(
        f"[bold cyan]录音类型:[/bold cyan] {type_labels.get(rec_type, rec_type)}")

    # 3. 根据类型路由到不同分析策略
    logger.info(f"[3/3] 开始分析 ({type_labels.get(rec_type, rec_type)})...")
    llm_start = time.time()

    if rec_type == "interview":
        _handle_interview(audio_stem, transcript_text,
                          event_time, asr_duration, llm_start, total_start)
    elif rec_type == "meeting":
        _handle_meeting(audio_stem, transcript_text, output_format,
                        event_time, asr_duration, llm_start, total_start)
    else:
        _handle_general(audio_stem, transcript_text, event_time,
                        asr_duration, llm_start, total_start)


def _build_timing_info(
    asr_duration: float, llm_start: float, total_start: float
) -> dict:
    """构建耗时信息字典"""
    llm_duration = time.time() - llm_start
    total_duration = time.time() - total_start
    return {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "asr_duration": asr_duration,
        "llm_duration": llm_duration,
        "total_duration": total_duration,
    }


def _handle_meeting(
    audio_stem: str,
    transcript_text: str,
    output_format: str,
    event_time: str | None,
    asr_duration: float,
    llm_start: float,
    total_start: float,
):
    """处理会议录音 -> output/summary/meeting/"""
    from agent.llm_client import LLMClient
    from agent.prompts import MEETING_ANALYSIS_PROMPT
    from models.schemas import MeetingResult

    # 检查 Summary 缓存
    cached = load_summary_cache(audio_stem, rec_type="meeting")
    if cached is not None:
        console.print(
            f"[green]使用 Summary 缓存: "
            f"meeting/{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md[/green]"
        )
        console.print(Markdown(cached))
        return

    llm = LLMClient()
    prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript_text)
    data = asyncio.run(llm.chat_json(prompt))
    result = MeetingResult.model_validate(data)

    timing_info = _build_timing_info(asr_duration, llm_start, total_start)
    output_text = format_markdown(
        result, event_time=event_time, timing_info=timing_info)
    save_summary_cache(output_text, audio_stem, rec_type="meeting")
    console.print(
        f"[green]Summary 已保存: "
        f"meeting/{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md[/green]"
    )

    if output_format == "json":
        console.print_json(result.model_dump_json(indent=2))
    else:
        console.print(Markdown(output_text))


def _handle_interview(
    audio_stem: str,
    transcript_text: str,
    event_time: str | None,
    asr_duration: float,
    llm_start: float,
    total_start: float,
):
    """处理面试录音 -> output/summary/interview/{name}_{asr}_{llm}/"""
    from agent.pipeline import format_interview_analysis, format_interview_questions

    # 检查面试缓存
    cached = load_interview_cache(audio_stem)
    if cached is not None:
        console.print(
            f"[green]使用面试分析缓存: "
            f"interview/{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}/[/green]"
        )
        console.print(Markdown(cached["analyze"]))
        return

    # 执行分析，获取原始 JSON 数据
    result_raw = asyncio.run(analyze_interview(transcript_text))

    # 分析完成后，构建耗时信息并格式化
    timing_info = _build_timing_info(asr_duration, llm_start, total_start)
    question_md = format_interview_questions(
        result_raw["q_data"], event_time, timing_info)
    analyze_md = format_interview_analysis(
        result_raw["a_data"], event_time, timing_info)

    save_interview_cache(question_md, analyze_md, audio_stem)
    subdir = f"interview/{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}"
    console.print(f"[green]面试分析已保存: {subdir}/question.md[/green]")
    console.print(f"[green]面试分析已保存: {subdir}/analyze.md[/green]")

    console.print("\n[bold]--- 面试问题列表 ---[/bold]")
    console.print(Markdown(question_md))
    console.print("\n[bold]--- 面试分析报告 ---[/bold]")
    console.print(Markdown(analyze_md))


def _handle_general(
    audio_stem: str,
    transcript_text: str,
    event_time: str | None,
    asr_duration: float,
    llm_start: float,
    total_start: float,
):
    """处理通用录音 -> output/summary/other/"""
    from agent.pipeline import format_general_result

    # 检查 Summary 缓存
    cached = load_summary_cache(audio_stem, rec_type="general")
    if cached is not None:
        console.print(
            f"[green]使用 Summary 缓存: "
            f"other/{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md[/green]"
        )
        console.print(Markdown(cached))
        return

    # 执行分析，获取原始 JSON 数据
    data = asyncio.run(analyze_general(transcript_text))

    # 分析完成后，构建耗时信息并格式化
    timing_info = _build_timing_info(asr_duration, llm_start, total_start)
    output_text = format_general_result(data, event_time, timing_info)

    save_summary_cache(output_text, audio_stem, rec_type="general")
    console.print(
        f"[green]Summary 已保存: "
        f"other/{audio_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md[/green]"
    )

    console.print(Markdown(output_text))


# ---------- video 命令 ----------

@app.command()
def video(
    source: str = typer.Argument(
        ...,
        help="视频文件路径或 URL (支持 YouTube, Bilibili 等) ",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细日志",
    ),
):
    """分析视频，生成内容总结"""
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time:HH:mm:ss} | {level} | {message}",
        )

    # 检查 API Key
    if not settings.dashscope_api_key:
        console.print(
            "[red]错误: 未配置 DASHSCOPE_API_KEY[/red]\n"
            "请复制 .env.example 为 .env 并填入阿里云百炼 API Key"
        )
        raise typer.Exit(1)

    is_url = source.startswith(("http://", "https://"))
    display_name = "远程视频" if is_url else Path(source).name
    video_stem = get_audio_stem(source)  # 默认用 URL/文件名 作为 stem

    console.print(
        Panel(
            f"[bold]视频:[/bold] {display_name}\n"
            f"[bold]ASR 模型:[/bold] {settings.asr_models[0]}\n"
            f"[bold]LLM 模型:[/bold] {settings.llm_models[0]}",
            title="视频总结",
            border_style="blue",
        )
    )

    total_start = time.time()

    # 1. 获取视频元信息 (仅 URL)
    video_info = None
    if is_url:
        logger.info("[1/4] 获取视频信息...")
        from agent.video_client import get_video_info
        video_info = asyncio.run(get_video_info(source))
        if video_info:
            title = video_info.get('title', '')
            if title:
                # 用视频标题作为所有产物的命名基础
                video_stem = sanitize_stem(title)
                console.print(f"[dim]标题: {title}[/dim]")
            console.print(f"[dim]UP主: {video_info.get('uploader', '')}[/dim]")
            duration = video_info.get("duration")
            if duration:
                console.print(
                    f"[dim]时长: {duration // 60}分{duration % 60}秒[/dim]")

    # 2. 从视频提取音频
    logger.info("[2/4] 从视频提取音频...")
    from agent.video_client import extract_audio_from_video
    try:
        audio_path = asyncio.run(
            extract_audio_from_video(source, filename_stem=video_stem))
    except Exception as e:
        console.print(f"[red]视频处理失败: {e}[/red]")
        raise typer.Exit(1)

    # 3. ASR 转写 (带缓存)
    logger.info("[3/4] ASR 语音转写...")
    transcript_text, asr_from_cache, asr_duration = asyncio.run(
        _get_transcript_text(str(audio_path), video_stem))

    # 清理临时音频文件
    try:
        audio_path.unlink()
    except Exception:
        pass

    if not transcript_text.strip():
        console.print("[red]错误: ASR 转写结果为空[/red]")
        raise typer.Exit(1)

    # 4. LLM 分析
    logger.info("[4/4] LLM 内容分析...")
    llm_start = time.time()

    # 检查 Summary 缓存
    cached = load_summary_cache(video_stem, rec_type="video")
    if cached is not None:
        console.print(
            f"[green]使用 Summary 缓存: "
            f"video/{video_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md[/green]"
        )
        console.print(Markdown(cached))
        return

    from agent.pipeline import analyze_general, format_general_result
    data = asyncio.run(analyze_general(transcript_text))

    timing_info = _build_timing_info(asr_duration, llm_start, total_start)
    output_text = format_general_result(
        data, event_time=None, timing_info=timing_info)

    save_summary_cache(output_text, video_stem, rec_type="video")
    console.print(
        f"[green]Summary 已保存: "
        f"video/{video_stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md[/green]"
    )

    console.print(Markdown(output_text))


if __name__ == "__main__":
    app()
