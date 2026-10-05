"""会议助手智能体 - Web 服务 (FastAPI)

使用 LangGraph StateGraph 统一处理流水线，详见 agent/graph.py。
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from loguru import logger
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings

# ── 日志配置 ──────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    str(LOG_DIR / "server_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    rotation="00:00",
    retention="7 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} | {message}",
)


def _content_disposition(filename: str) -> str:
    """生成兼容中文文件名的 Content-Disposition 头 (RFC 5987)"""
    ascii_name = filename.encode("ascii", "replace").decode("ascii")
    utf8_name = quote(filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"

# ── FastAPI App ──────────────────────────────────────────────


app = FastAPI(title="会议助手智能体")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
RESULT_DIR = settings.output_dir / "summary"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Models ───────────────────────────────────────────────────


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending / processing / completed / failed
    progress: int = 0
    step: str = ""
    error: str | None = None
    result_path: str | None = None
    asr_result_path: str | None = None
    filename: str = ""
    file_type: str = ""  # audio / video
    video_title: str | None = None
    # ─ 详细元数据 ──
    source_type: str = ""             # local_audio | local_video | url_audio | url_video
    asr_model_used: str = ""          # 实际使用的 ASR 模型
    llm_model_used: str = ""          # 实际使用的 LLM 模型
    asr_sentences: int = 0            # ASR 转写句数
    asr_duration: float = 0.0         # ASR 耗时 (秒)
    llm_duration: float = 0.0         # LLM 耗时 (秒)
    total_duration: float = 0.0       # 总耗时 (秒)
    rec_type: str = ""                # meeting | interview | general
    cache_hit: bool = False           # Summary 缓存是否命中
    file_size: int = 0                # 文件大小 (字节)
    audio_duration: float = 0.0       # 音视频时长 (秒)
    asr_chars: int = 0                # ASR 转写字符数
    cleaned_sentences: int = 0        # 清洗后句数


class UploadResponse(BaseModel):
    task_id: str
    filename: str
    file_type: str


class UrlRequest(BaseModel):
    url: str
    asr_model: str | None = None
    llm_model: str | None = None


class UploadRequest(BaseModel):
    asr_model: str | None = None
    llm_model: str | None = None


# ── Task Store ───────────────────────────────────────────────

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _update(task_id: str, **kw):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kw)


# ── Graph-based Processing ───────────────────────────────────


def _save_result(task_id: str, stem: str, subdir: str, text: str):
    """保存分析结果文件并更新任务状态 (作为 result_handler 注册到 graph)"""
    result_path = RESULT_DIR / subdir / \
        f"{stem}_{settings.asr_models[0]}_{settings.llm_models[0]}.md"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(text, encoding="utf-8")
    _update(task_id, result_path=str(result_path))


def _process(task_id: str, source: str, source_type: str,
             filename: str, file_type: str,
             asr_model: str | None = None, llm_model: str | None = None):
    """后台线程: 通过 LangGraph 流水线处理音视频

    统一替代原先的 _process_audio / _process_video /
    _process_audio_url / _process_video_url 四个函数。
    """
    logger.info(
        f"[任务 {task_id}] 开始处理: source_type={source_type}, "
        f"asr_model={asr_model}, llm_model={llm_model}"
    )
    try:
        from agent.graph import (
            invoke_pipeline,
            register_progress_cb,
            register_result_handler,
            unregister_task,
        )
        from agent.cache import get_audio_stem

        # 注册进度回调和结果处理器
        register_progress_cb(
            task_id,
            lambda **kw: _update(task_id, **kw),
        )
        register_result_handler(task_id, _save_result)

        # 构建初始状态
        audio_stem = get_audio_stem(source)

        initial_state = {
            "source": source,
            "source_type": source_type,
            "asr_model": asr_model,
            "llm_model": llm_model,
            "task_id": task_id,
            "audio_stem": audio_stem,
            # 中间产物初始值
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
            "pipeline_start": time.time(),
            "error": None,
        }

        invoke_pipeline(initial_state)
        _update(task_id, status="completed", progress=100, step="分析完成")
        logger.info(f"[任务 {task_id}] 处理完成")

    except Exception as e:
        logger.error(f"[任务 {task_id}] 处理失败: {e}", exc_info=True)
        _update(task_id, status="failed", error=str(e), step="处理失败")
    finally:
        from agent.graph import unregister_task
        unregister_task(task_id)


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov",
              ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
ALLOWED_EXTS = VIDEO_EXTS | AUDIO_EXTS

# 已知视频平台域名
VIDEO_DOMAINS = {
    "youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com",
    "bilibili.com", "www.bilibili.com", "b23.tv", "bili2233.cn",
    "v.qq.com", "youku.com", "iqiyi.com", "tudou.com",
    "vimeo.com", "dailymotion.com", "twitter.com", "x.com",
}


# ── API Endpoints ────────────────────────────────────────────


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    asr_model: str | None = None,
    llm_model: str | None = None,
):
    """上传音视频文件，创建后台处理任务"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400, f"不支持的文件格式: {ext}\n支持: {', '.join(sorted(ALLOWED_EXTS))}")

    file_type = "video" if ext in VIDEO_EXTS else "audio"
    task_id = _new_task_id()
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    file_path = task_dir / f"{task_id}{ext}"
    content = await file.read()
    file_path.write_bytes(content)

    source_type = "local_video" if file_type == "video" else "local_audio"

    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "progress": 5,
            "step": "任务已创建",
            "error": None,
            "result_path": None,
            "asr_result_path": None,
            "filename": file.filename or "",
            "file_type": file_type,
            "video_title": None,
            "source_type": source_type,
            "file_size": len(content),
        }

    threading.Thread(
        target=_process,
        args=(task_id, str(file_path), source_type, file.filename or "",
              file_type, asr_model, llm_model),
        daemon=True,
    ).start()

    return UploadResponse(task_id=task_id, filename=file.filename or "", file_type=file_type)


@app.post("/api/submit-url")
async def submit_url(req: UrlRequest):
    """提交音视频链接进行分析"""
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "请输入有效的音视频链接")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "链接需以 http:// 或 https:// 开头")

    # 判断是视频还是音频链接
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    is_video = any(vd in domain for vd in {
        "youtube.com", "youtu.be", "bilibili.com", "b23.tv",
        "v.qq.com", "youku.com", "iqiyi.com", "vimeo.com",
    })
    # 也检查扩展名
    if not is_video:
        ext = Path(parsed.path).suffix.lower()
        is_video = ext in VIDEO_EXTS

    file_type = "video" if is_video else "audio"
    source_type = "url_video" if is_video else "url_audio"
    task_id = _new_task_id()
    display_name = url[:80] + ("..." if len(url) > 80 else "")

    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "progress": 5,
            "step": "任务已创建",
            "error": None,
            "result_path": None,
            "asr_result_path": None,
            "filename": display_name,
            "file_type": file_type,
            "video_title": None,
            "source_type": source_type,
            "file_size": 0,
        }

    threading.Thread(
        target=_process,
        args=(task_id, url, source_type, display_name,
              file_type, req.asr_model, req.llm_model),
        daemon=True,
    ).start()

    return {"task_id": task_id, "filename": display_name, "file_type": file_type}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    return tasks[task_id]


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取分析结果 (Markdown)"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成")
    rp = task.get("result_path")
    if not rp or not Path(rp).exists():
        raise HTTPException(404, "结果文件不存在")
    text = Path(rp).read_text(encoding="utf-8")
    return {"content": text, "filename": Path(rp).name}


@app.get("/api/result/{task_id}/download")
async def download_result(task_id: str):
    """下载结果文件"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成")
    rp = task.get("result_path")
    if not rp or not Path(rp).exists():
        raise HTTPException(404, "结果文件不存在")
    text = Path(rp).read_text(encoding="utf-8")
    return PlainTextResponse(text, media_type="text/markdown",
                             headers={"Content-Disposition": _content_disposition(Path(rp).name)})


@app.get("/api/asr/{task_id}")
async def get_asr_result(task_id: str):
    """获取 ASR 转写结果"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成")
    asr_path = task.get("asr_result_path")
    if not asr_path or not Path(asr_path).exists():
        raise HTTPException(404, "ASR 结果不存在")
    text = Path(asr_path).read_text(encoding="utf-8")
    return {"content": text, "filename": Path(asr_path).name}


@app.get("/api/asr/{task_id}/download")
async def download_asr_result(task_id: str):
    """下载 ASR 转写结果"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成")
    asr_path = task.get("asr_result_path")
    if not asr_path or not Path(asr_path).exists():
        raise HTTPException(404, "ASR 结果不存在")
    text = Path(asr_path).read_text(encoding="utf-8")
    return PlainTextResponse(text, media_type="text/plain",
                             headers={"Content-Disposition": _content_disposition(Path(asr_path).name)})


@app.get("/api/models")
async def get_models():
    """获取可用的 ASR 和 LLM 模型列表"""
    return {
        "asr_models": settings.asr_models,
        "llm_models": settings.llm_models,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "api_key_configured": bool(settings.dashscope_api_key)}


# ── Serve Frontend ───────────────────────────────────────────

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir),
              html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
