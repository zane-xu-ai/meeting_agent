"""会议助手智能体 - CLI 入口

用法:
    python main.py transcript-only <音频路径或URL>   # 仅 ASR 转写
    python main.py analyze <音频路径或URL>            # 完整分析 (ASR + LLM)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.cache import (
    get_audio_stem,
    load_asr_cache,
    load_summary_cache,
    save_asr_cache,
    save_summary_cache,
)
from agent.pipeline import format_markdown
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
    help="会议助手智能体 - 音频输入，输出结构化会议纪要",
)
console = Console()


# ---------- 公共: 获取 ASR 转写文本 (带缓存) ----------

async def _get_transcript_text(audio_source: str, audio_stem: str) -> tuple[str, bool]:
    """
    获取 ASR 转写文本，优先使用缓存。

    返回: (转写文本, is_from_cache: bool)
    """
    from agent.asr_client import ASRClient
    from utils.audio import preprocess_audio

    # 1. 检查 ASR 缓存
    cached = load_asr_cache(audio_stem)
    if cached is not None:
        console.print(
            f"[green]使用 ASR 缓存: {audio_stem}_{settings.asr_model}.txt[/green]")
        return cached, True

    # 2. 缓存未命中，执行 ASR 转写
    is_url = audio_source.startswith(("http://", "https://"))
    if is_url:
        asr_input = audio_source
        preprocessed = None
    else:
        preprocessed = preprocess_audio(
            audio_source, sample_rate=settings.asr_sample_rate)
        asr_input = preprocessed

    try:
        asr = ASRClient()
        transcript = await asr.transcribe(asr_input)
    finally:
        if preprocessed and preprocessed.exists():
            preprocessed.unlink()

    # 3. 文本预处理
    processor = TextProcessor()
    cleaned = processor.clean(transcript)
    transcript_text = cleaned.formatted_text

    # 4. 保存 ASR 缓存
    save_asr_cache(transcript_text, audio_stem)
    console.print(
        f"[green]ASR 结果已保存: {audio_stem}_{settings.asr_model}.txt[/green]")

    return transcript_text, False


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

    transcript_text, from_cache = asyncio.run(
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
    """分析会议音频，生成结构化会议纪要"""
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
            f"[bold]ASR 模型:[/bold] {settings.asr_model}\n"
            f"[bold]LLM 模型:[/bold] {settings.llm_model}",
            title="会议助手智能体",
            border_style="blue",
        )
    )

    # 1. 检查 Summary 缓存
    cached_summary = load_summary_cache(audio_stem)
    if cached_summary is not None:
        console.print(
            f"[green]使用 Summary 缓存: "
            f"{audio_stem}_{settings.asr_model}_{settings.llm_model}.md[/green]"
        )
        console.print(Markdown(cached_summary))
        return

    # 2. Summary 缓存未命中，需要生成
    # 2a. 获取 ASR 转写文本 (带缓存)
    logger.info("[1/2] ASR 语音转写...")
    transcript_text, asr_from_cache = asyncio.run(
        _get_transcript_text(audio, audio_stem))

    if not transcript_text.strip():
        console.print("[red]错误: ASR 转写结果为空，请检查音频文件是否包含有效语音[/red]")
        raise typer.Exit(1)

    # 2b. LLM 分析
    logger.info("[2/2] LLM 智能分析...")
    from agent.llm_client import LLMClient
    from agent.prompts import MEETING_ANALYSIS_PROMPT
    from models.schemas import MeetingResult

    llm = LLMClient()
    prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript_text)
    data = asyncio.run(llm.chat_json(prompt))
    result = MeetingResult.model_validate(data)

    # 3. 格式化并保存 Summary 缓存
    output_text = format_markdown(result)
    save_summary_cache(output_text, audio_stem)
    console.print(
        f"[green]Summary 已保存: "
        f"{audio_stem}_{settings.asr_model}_{settings.llm_model}.md[/green]"
    )

    # 4. 展示结果
    if output_format == "json":
        console.print_json(result.model_dump_json(indent=2))
    else:
        console.print(Markdown(output_text))


if __name__ == "__main__":
    app()
