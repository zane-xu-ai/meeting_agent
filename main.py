"""会议助手智能体 - CLI 入口

用法:
    python main.py transcript-only <音频路径或URL>   # 仅 ASR 转写
    python main.py analyze <音频路径或URL>            # 完整分析 (ASR + LLM)
    python main.py video <视频路径或URL>              # 视频总结

analyze 和 video 命令使用 LangGraph StateGraph 统一流水线。
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
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
    save_asr_cache,
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


# ---------- 公共函数 ----------


def get_event_time(audio_source: str) -> str | None:
    """获取音频文件的事件时间(文件修改时间)。URL 返回 None。"""
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


def _determine_source_type(source: str, is_video: bool = False) -> str:
    """根据来源判断 source_type"""
    is_url = source.startswith(("http://", "https://"))
    if is_url:
        return "url_video" if is_video else "url_audio"
    return "local_video" if is_video else "local_audio"


def _run_graph_pipeline(
    source: str,
    source_type: str,
    task_id: str,
) -> dict:
    """
    通过 LangGraph 流水线处理音视频，返回 tasks 字典。

    设置 CLI 专用的进度回调 (Rich console 输出)。
    """
    from agent.graph import (
        invoke_pipeline,
        register_progress_cb,
        register_result_handler,
        unregister_task,
    )
    from server import _save_result

    # CLI 进度回调: 在终端打印步骤信息
    def cli_progress(step: str = "", progress: int = 0, **kw):
        if step:
            console.print(f"[dim]  [{progress:3d}%] {step}[/dim]")

    register_progress_cb(task_id, cli_progress)
    register_result_handler(task_id, _save_result)

    initial_state = {
        "source": source,
        "source_type": source_type,
        "asr_model": None,
        "llm_model": None,
        "task_id": task_id,
        "audio_stem": get_audio_stem(source),
        "audio_path": None,
        "preprocessed_path": None,
        "asr_input": None,
        "transcript": None,
        "transcript_text": "",
        "asr_duration": 0.0,
        "rec_type": "general",
        "output_text": "",
        "result_path": None,
        "video_title": None,
        "llm_start": 0.0,
        "error": None,
    }

    tasks: dict = {"status": "processing", "progress": 0}

    try:
        invoke_pipeline(initial_state)
        tasks["status"] = "completed"
    except Exception as e:
        tasks["status"] = "failed"
        tasks["error"] = str(e)
    finally:
        unregister_task(task_id)

    return tasks


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

    # 检查 ASR 缓存
    cached = load_asr_cache(audio_stem)
    if cached is not None:
        console.print(
            f"[green]使用 ASR 缓存: {audio_stem}_{settings.asr_models[0]}.txt[/green]")
        transcript_text = cached
    else:
        # 执行 ASR 转写
        from agent.asr_client import ASRClient
        from utils.audio import preprocess_audio

        is_url = audio.startswith(("http://", "https://"))
        if is_url:
            asr_input = audio
            preprocessed = None
        else:
            preprocessed = preprocess_audio(
                audio, sample_rate=settings.asr_sample_rate)
            asr_input = preprocessed

        t0 = time.time()
        try:
            asr = ASRClient()
            transcript = asyncio.run(asr.transcribe(asr_input))
        finally:
            if preprocessed and preprocessed.exists():
                preprocessed.unlink()
        asr_duration = time.time() - t0

        processor = TextProcessor()
        cleaned = processor.clean(transcript)
        transcript_text = cleaned.formatted_text

        save_asr_cache(transcript_text, audio_stem)
        console.print(
            f"[green]ASR 结果已保存: {audio_stem}_{settings.asr_models[0]}.txt[/green]")

    console.print("\n[bold]转写结果:[/bold]")
    console.print(transcript_text)


# ---------- analyze 命令 ----------


@app.command()
def analyze(
    audio: str = typer.Argument(
        ...,
        help="音频文件路径或 URL (支持 wav/mp3/m4a/flac/ogg 等格式)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细日志",
    ),
):
    """分析音频，自动识别类型并生成对应的总结 (LangGraph 流水线)"""
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time:HH:mm:ss} | {level} | {message}",
        )

    if not settings.dashscope_api_key:
        console.print(
            "[red]错误: 未配置 DASHSCOPE_API_KEY[/red]\n"
            "请复制 .env.example 为 .env 并填入阿里云百炼 API Key\n"
            "获取地址: https://bailian.console.aliyun.com/"
        )
        raise typer.Exit(1)

    is_url = audio.startswith(("http://", "https://"))
    display_name = "远程音频" if is_url else Path(audio).name
    source_type = _determine_source_type(audio)
    task_id = uuid.uuid4().hex[:12]

    console.print(
        Panel(
            f"[bold]音频:[/bold] {display_name}\n"
            f"[bold]ASR 模型:[/bold] {settings.asr_models[0]}\n"
            f"[bold]LLM 模型:[/bold] {settings.llm_models[0]}",
            title="会议助手智能体 (LangGraph)",
            border_style="blue",
        )
    )

    # 通过 LangGraph 流水线执行
    tasks = _run_graph_pipeline(audio, source_type, task_id)

    if tasks["status"] == "failed":
        console.print(f"[red]处理失败: {tasks.get('error', '未知错误')}[/red]")
        raise typer.Exit(1)

    # 读取并显示结果
    result_path = tasks.get("result_path")
    if result_path and Path(result_path).exists():
        text = Path(result_path).read_text(encoding="utf-8")
        console.print(Markdown(text))
    else:
        console.print("[yellow]处理完成但未找到结果文件[/yellow]")


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
    """分析视频，生成内容总结 (LangGraph 流水线)"""
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time:HH:mm:ss} | {level} | {message}",
        )

    if not settings.dashscope_api_key:
        console.print(
            "[red]错误: 未配置 DASHSCOPE_API_KEY[/red]\n"
            "请复制 .env.example 为 .env 并填入阿里云百炼 API Key"
        )
        raise typer.Exit(1)

    is_url = source.startswith(("http://", "https://"))
    display_name = "远程视频" if is_url else Path(source).name
    source_type = _determine_source_type(source, is_video=True)
    task_id = uuid.uuid4().hex[:12]

    console.print(
        Panel(
            f"[bold]视频:[/bold] {display_name}\n"
            f"[bold]ASR 模型:[/bold] {settings.asr_models[0]}\n"
            f"[bold]LLM 模型:[/bold] {settings.llm_models[0]}",
            title="视频总结 (LangGraph)",
            border_style="blue",
        )
    )

    # 获取视频元信息 (仅 URL，用于显示)
    if is_url:
        from agent.video_client import get_video_info
        from agent.cache import sanitize_stem

        video_info = asyncio.run(get_video_info(source))
        if video_info:
            title = video_info.get("title", "")
            if title:
                console.print(f"[dim]标题: {title}[/dim]")
            console.print(f"[dim]UP主: {video_info.get('uploader', '')}[/dim]")
            duration = video_info.get("duration")
            if duration:
                console.print(
                    f"[dim]时长: {duration // 60}分{duration % 60}秒[/dim]")

    # 通过 LangGraph 流水线执行
    tasks = _run_graph_pipeline(source, source_type, task_id)

    if tasks["status"] == "failed":
        console.print(f"[red]处理失败: {tasks.get('error', '未知错误')}[/red]")
        raise typer.Exit(1)

    # 读取并显示结果
    result_path = tasks.get("result_path")
    if result_path and Path(result_path).exists():
        text = Path(result_path).read_text(encoding="utf-8")
        console.print(Markdown(text))
    else:
        console.print("[yellow]处理完成但未找到结果文件[/yellow]")


if __name__ == "__main__":
    app()
