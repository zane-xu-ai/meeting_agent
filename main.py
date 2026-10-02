"""会议助手智能体 - CLI 入口

用法:
    python main.py <音频文件路径>
    python main.py meeting.wav --output markdown
    python main.py meeting.mp3 --output json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.pipeline import MeetingPipeline, format_markdown
from config import settings

# 配置 loguru
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | {message}")

app = typer.Typer(
    name="meeting-agent",
    help="会议助手智能体 - 音频输入，输出结构化会议纪要",
)
console = Console()


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
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="是否保存结果到文件",
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
        logger.add(sys.stderr, level="DEBUG",
                   format="{time:HH:mm:ss} | {level} | {message}")

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

    console.print(
        Panel(
            f"[bold]音频:[/bold] {display_name}\n"
            f"[bold]ASR 模型:[/bold] {settings.asr_model}\n"
            f"[bold]LLM 模型:[/bold] {settings.llm_model}",
            title="会议助手智能体",
            border_style="blue",
        )
    )

    # 执行流水线
    pipeline = MeetingPipeline()
    result = asyncio.run(pipeline.run(audio))

    # 输出结果
    if output_format == "json":
        output_text = result.model_dump_json(indent=2)
        console.print_json(output_text)
    else:
        output_text = format_markdown(result)
        console.print(Markdown(output_text))

    # 保存文件
    if save:
        output_dir = settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_url:
            stem = "remote_audio"
        else:
            stem = Path(audio).stem

        if output_format == "json":
            out_path = output_dir / f"{stem}_meeting.json"
            out_path.write_text(result.model_dump_json(
                indent=2), encoding="utf-8")
        else:
            out_path = output_dir / f"{stem}_meeting.md"
            out_path.write_text(output_text, encoding="utf-8")

        console.print(f"\n[green]结果已保存到: {out_path}[/green]")


@app.command()
def transcript_only(
    audio: str = typer.Argument(..., help="音频文件路径或 URL"),
):
    """仅执行 ASR 转写，不进行 LLM 分析 (用于调试)"""
    if not settings.dashscope_api_key:
        console.print("[red]错误: 未配置 DASHSCOPE_API_KEY[/red]")
        raise typer.Exit(1)

    from agent.asr_client import ASRClient
    from agent.text_processor import TextProcessor
    from utils.audio import preprocess_audio

    is_url = audio.startswith(("http://", "https://"))
    display_name = "远程音频" if is_url else Path(audio).name
    console.print(f"[bold]转写音频:[/bold] {display_name}")

    if is_url:
        asr_input = audio
        preprocessed = None
    else:
        preprocessed = preprocess_audio(
            audio, sample_rate=settings.asr_sample_rate)
        asr_input = preprocessed

    try:
        asr = ASRClient()
        transcript = asyncio.run(asr.transcribe(asr_input))

        processor = TextProcessor()
        cleaned = processor.clean(transcript)

        console.print("\n[bold]转写结果:[/bold]")
        console.print(cleaned.formatted_text)
    finally:
        if preprocessed and preprocessed.exists():
            preprocessed.unlink()


if __name__ == "__main__":
    app()
