"""会议助手智能体 - Web 服务 (FastAPI)

使用 LangGraph StateGraph 统一处理流水线，详见 agent/graph.py。
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

from loguru import logger
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from agent.context import get_context_manager
from agent.memory import get_memory_manager
from agent.quota import QuotaManager
from utils.ingress import RequestLimit, authorized, save_upload
from utils.media_network import UnsafeMediaURL, validate_media_url
from persistence.repository import public_state, repository
from persistence import dispatch

# M5: 长期记忆 (向量检索)
from agent.embedding import get_embedding_client
from agent.reranker import get_reranker_client
from agent.vector_store import get_vector_store
from agent.chunker import chunk_summary, chunk_asr

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
    diagnose=False,
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} | {message}",
)


def _content_disposition(filename: str) -> str:
    """生成兼容中文文件名的 Content-Disposition 头 (RFC 5987)"""
    ascii_name = "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in filename)
    utf8_name = quote(filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"

# ── FastAPI App ──────────────────────────────────────────────


app = FastAPI(title="会议助手智能体")

app.add_middleware(RequestLimit, limit=lambda: settings.max_upload_bytes + 1024 * 1024)


@app.middleware("http")
async def require_workspace(request: Request, call_next):
    if request.url.path != "/api/health":
        try:
            allowed = authorized(request, settings.workspace_token)
        except ValueError:
            allowed = False
        if not allowed:
            headers = {"WWW-Authenticate": 'Basic realm="Meeting Agent"'} if settings.workspace_token else None
            return JSONResponse({"detail": "需要工作区访问权限"}, 401, headers=headers)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin.rstrip("/") != expected:
                return JSONResponse({"detail": "请求来源与工作区不同"}, 403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path == "/" or request.url.path.endswith((".html", ".js")):
        response.headers["Cache-Control"] = "no-store"
    return response


UPLOAD_DIR = Path("./uploads")
RESULT_DIR = settings.output_dir / "summary"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── M10: 多用户配额管理器 ──────────────────────────────────────
quota_manager = QuotaManager(
    data_dir=settings.quota_data_dir,
    daily_limit=settings.daily_task_limit,
    whitelist=set(settings.quota_whitelist_ips),
)


def get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"

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


class ChatRequest(BaseModel):
    message: str


# ── Task Store ───────────────────────────────────────────────

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()

# ── M4: Short-term Memory Store ─────────────────────────────────
# 使用 MemoryManager 管理对话记忆 (替代原有的 chat_histories 字典)
# 支持滑动窗口、历史压缩等功能


def _asr_models():
    from agent.local_asr import LOCAL_MODEL, available
    return [*settings.asr_models, LOCAL_MODEL] if available() else settings.asr_models


def _validate_models(asr_model, llm_model):
    if (asr_model and asr_model not in _asr_models()
            or llm_model and llm_model not in settings.llm_models):
        raise HTTPException(422, "模型不在当前可用列表")


def _new_task_id() -> str:
    return uuid.uuid4().hex


def _read_task(task_id):
    repo = repository()
    task = repo.task(task_id) if repo else tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return task


def _update(task_id: str, **kw):
    repo = repository()
    if repo:
        repo.update(task_id, kw)
    else:
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id].update(kw)


def _register_task(state, location, asr_model, llm_model, *, modified_ms=None, batch_id=None):
    repo = repository()
    if repo:
        repo.create(state, location, {"asr_model": asr_model, "llm_model": llm_model},
                    modified_ms=modified_ms, batch_id=batch_id, queued=dispatch.enabled())
    elif dispatch.enabled():
        raise HTTPException(503, "任务队列需要 PostgreSQL")
    else:
        with tasks_lock:
            tasks[state["task_id"]] = state


# ── Graph-based Processing ───────────────────────────────────


async def _index_task_to_vector_store(task_id: str, task: dict):
    """M5: 后台索引任务 — 将分析结果和 ASR 转写入向量库"""
    try:
        logger.info(f"[M5-Index] 任务 {task_id} 开始索引...")

        # 获取 Embedding 客户端和向量存储
        embed_client = get_embedding_client()
        vector_store = get_vector_store()

        # 先删除旧数据 (幂等)
        await vector_store.delete_by_task_id(task_id)

        # 按源文件名去重: 删除同一源文件的旧索引 (防止重复上传导致重复数据)
        source_files_to_delete = set()
        result_path = task.get("result_path")
        asr_path = task.get("asr_result_path")
        if result_path and Path(result_path).exists():
            source_files_to_delete.add(Path(result_path).name)
        if asr_path and Path(asr_path).exists():
            source_files_to_delete.add(Path(asr_path).name)
        for sf in source_files_to_delete:
            deleted = await vector_store.delete_by_source_file(sf)
            if deleted > 0:
                logger.info(f"[M5-Index] 去重删除旧索引: {sf} ({deleted} 子块)")

        all_parent_chunks = []
        all_child_texts = []

        # 1. 索引分析报告 (Summary)
        result_path = task.get("result_path")
        if result_path and Path(result_path).exists():
            summary_text = Path(result_path).read_text(encoding="utf-8")
            rec_type = task.get("rec_type", "general")
            source_file = Path(result_path).name

            logger.debug(
                f"[M5-Index] Summary 分块: {len(summary_text)} 字符"
            )
            summary_chunks = chunk_summary(
                summary_text, task_id, rec_type, source_file
            )
            all_parent_chunks.extend(summary_chunks)
            for parent in summary_chunks:
                for child in parent.children:
                    all_child_texts.append(child.content)

            logger.info(
                f"[M5-Index] Summary: {len(summary_chunks)} 父块, "
                f"{sum(len(p.children) for p in summary_chunks)} 子块"
            )

        # 2. 索引 ASR 转写
        asr_path = task.get("asr_result_path")
        if asr_path and Path(asr_path).exists():
            asr_text = Path(asr_path).read_text(encoding="utf-8")
            rec_type = task.get("rec_type", "general")
            source_file = Path(asr_path).name

            logger.debug(
                f"[M5-Index] ASR 分块: {len(asr_text)} 字符"
            )
            asr_chunks = chunk_asr(
                asr_text, task_id, rec_type, source_file
            )
            all_parent_chunks.extend(asr_chunks)
            for parent in asr_chunks:
                for child in parent.children:
                    all_child_texts.append(child.content)

            logger.info(
                f"[M5-Index] ASR: {len(asr_chunks)} 父块, "
                f"{sum(len(p.children) for p in asr_chunks)} 子块"
            )

        if not all_parent_chunks:
            logger.warning(f"[M5-Index] 任务 {task_id} 无可索引内容")
            return

        # 3. Embedding 向量化
        logger.debug(
            f"[M5-Index] 开始 Embedding: {len(all_child_texts)} 个子块"
        )
        embeddings = await embed_client.embed_batch(all_child_texts)

        # 4. 写入向量库
        written = await vector_store.add_chunks(
            all_parent_chunks, embeddings
        )

        stats = vector_store.get_stats()
        logger.info(
            f"[M5-Index] 任务 {task_id} 索引完成: "
            f"{len(all_parent_chunks)} 父块, {written} 子块, "
            f"向量库总计: {stats['child_count']} 子块"
        )

    except Exception as e:
        logger.error(
            f"[M5-Index] 任务 {task_id} 索引失败: {e}", exc_info=True
        )
        # 降级: 索引失败不影响主流程


def _save_result(task_id: str, stem: str, subdir: str, text: str,
                 asr_model_used: str | None = None,
                 llm_model_used: str | None = None):
    """保存分析结果文件并更新任务状态 (作为 result_handler 注册到 graph)"""
    asr_name = asr_model_used or settings.asr_models[0]
    llm_name = llm_model_used or settings.llm_models[0]
    result_path = RESULT_DIR / subdir / \
        f"{stem}_{asr_name}_{llm_name}.md"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(text, encoding="utf-8")
    _update(task_id, result_path=str(result_path))


def _process(task_id: str, source: str, source_type: str,
             filename: str, file_type: str,
             asr_model: str | None = None, llm_model: str | None = None,
             source_sha256: str | None = None):
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
        # 本地文件用原始文件名做缓存 key，URL 用 source 路径
        if source_type.startswith("local_") and filename:
            audio_stem = f"{get_audio_stem(filename)[:40]}_{source_sha256 or task_id}"
        else:
            audio_stem = f"url_{task_id}"

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

        result = invoke_pipeline(initial_state)
        repo = repository()
        if repo:
            repo.record_result(task_id, result, _read_task(task_id))
        else:
            _update(task_id, status="completed", progress=100, step="分析完成")
        logger.info(f"[任务 {task_id}] 处理完成")

        # M5: 后台线程索引到向量库 (独立线程 + 独立 event loop)
        if settings.enable_vector_store:
            import threading
            task_data = dict(_read_task(task_id))

            def _run_index():
                import asyncio
                asyncio.run(_index_task_to_vector_store(task_id, task_data))
            threading.Thread(target=_run_index, daemon=True,
                             name=f"index-{task_id}").start()

    except Exception as e:
        from agent.llm_client import LLMJSONError

        if isinstance(e, LLMJSONError):
            code, message = "invalid_model_output", str(e)
        elif isinstance(e, UnsafeMediaURL):
            code, message = "unsafe_media_url", str(e)
        else:
            code, message = "processing_failed", "处理失败，请按任务编号查看服务日志"
        logger.error("[任务 {}] {} ({})", task_id, code, type(e).__name__)
        _update(task_id, status="failed", error=message, error_code=code, step="处理失败")
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
    request: Request,
    file: UploadFile = File(...),
    asr_model: str | None = None,
    llm_model: str | None = None,
    modified_ms: int | None = None,
):
    return await _submit_upload(request, file, asr_model, llm_model, modified_ms)


async def _submit_upload(request, file, asr_model=None, llm_model=None, modified_ms=None, batch_id=None):
    """上传音视频文件，创建后台处理任务"""
    # M10: 配额检查
    client_ip = get_client_ip(request)
    allowed, quota_info = quota_manager.check_quota(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"今日任务数量已达上限 ({quota_info['used']}/{quota_info['limit']})，明天再试吧！",
                "used": quota_info["used"],
                "limit": quota_info["limit"],
                "reset_time": "明天 00:00",
            }
        )

    _validate_models(asr_model, llm_model)
    original_name = Path((file.filename or "").replace("\\", "/")).name
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400, f"不支持的文件格式: {ext}\n支持: {', '.join(sorted(ALLOWED_EXTS))}")

    file_type = "video" if ext in VIDEO_EXTS else "audio"
    task_id = _new_task_id()
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(mode=0o700, parents=True, exist_ok=False)

    file_path = task_dir / f"{task_id}{ext}"
    try:
        size, digest = await save_upload(file, file_path, settings.max_upload_bytes)
    except BaseException:
        task_dir.rmdir()
        raise

    source_type = "local_video" if file_type == "video" else "local_audio"

    if not quota_manager.record_usage(client_ip, task_id):
        file_path.unlink()
        task_dir.rmdir()
        raise HTTPException(429, "今日任务数量已达上限")

    state = {
        "task_id": task_id,
        "status": "processing",
        "progress": 5,
        "step": "任务已创建",
        "error": None,
        "result_path": None,
        "asr_result_path": None,
        "filename": original_name,
        "file_type": file_type,
        "video_title": None,
        "source_type": source_type,
        "file_size": size,
        "content_sha256": digest,
    }
    queued = dispatch.enabled()
    if queued:
        state.update(status="pending", progress=0, step="等待派发")
    try:
        await asyncio.to_thread(_register_task, state, str(file_path), asr_model, llm_model,
                                modified_ms=modified_ms, batch_id=batch_id)
    except Exception:
        file_path.unlink(missing_ok=True)
        task_dir.rmdir()
        raise
    if queued:
        file_path.unlink()
        task_dir.rmdir()
    else:
        threading.Thread(
            target=_process,
            args=(task_id, str(file_path), source_type, original_name,
                  file_type, asr_model, llm_model, digest),
            daemon=True,
        ).start()

    return UploadResponse(task_id=task_id, filename=original_name, file_type=file_type)


@app.post("/api/submit-url")
async def submit_url(request: Request, req: UrlRequest):
    return await _submit_url(request, req)


async def _submit_url(request, req, batch_id=None):
    """提交音视频链接进行分析"""
    # M10: 配额检查
    client_ip = get_client_ip(request)
    allowed, quota_info = quota_manager.check_quota(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"今日任务数量已达上限 ({quota_info['used']}/{quota_info['limit']})，明天再试吧！",
                "used": quota_info["used"],
                "limit": quota_info["limit"],
                "reset_time": "明天 00:00",
            }
        )

    _validate_models(req.asr_model, req.llm_model)
    url = req.url.strip()
    if len(url) > 4096:
        raise HTTPException(400, "链接超过长度限制")
    try:
        parsed = await asyncio.to_thread(validate_media_url, url)
    except (UnsafeMediaURL, OSError) as error:
        raise HTTPException(400, "请输入可解析的公开 HTTP(S) 音视频链接") from error
    domain = parsed.hostname
    is_video = any(domain == host or domain.endswith("." + host) for host in VIDEO_DOMAINS)
    # 也检查扩展名
    if not is_video:
        ext = Path(parsed.path).suffix.lower()
        is_video = ext in VIDEO_EXTS

    file_type = "video" if is_video else "audio"
    source_type = "url_video" if is_video else "url_audio"
    task_id = _new_task_id()
    display_name = f"{domain}{parsed.path}"[:120]

    if not quota_manager.record_usage(client_ip, task_id):
        raise HTTPException(429, "今日任务数量已达上限")

    state = {
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
    queued = dispatch.enabled()
    if queued:
        state.update(status="pending", progress=0, step="等待派发")
    await asyncio.to_thread(_register_task, state, url, req.asr_model, req.llm_model, batch_id=batch_id)

    if not queued:
        threading.Thread(
            target=_process,
            args=(task_id, url, source_type, display_name, file_type, req.asr_model, req.llm_model),
            daemon=True,
        ).start()

    return {"task_id": task_id, "filename": display_name, "file_type": file_type}


@app.post("/api/batches", status_code=202)
async def submit_batch(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(default=""),
    asr_model: str | None = Form(default=None),
    llm_model: str | None = Form(default=None),
):
    repo = _queue_repository()
    _validate_models(asr_model, llm_model)
    links = [line.strip() for line in urls.splitlines() if line.strip()]
    if not 1 <= len(files) + len(links) <= 20:
        raise HTTPException(400, "每批可提交 1–20 个文件或链接")
    from persistence.stage_cache import fingerprint
    # Multipart files are spooled and seekable. Hash without loading all bytes into memory.
    def file_digest(file):
        import hashlib
        result = hashlib.file_digest(file.file, "sha256").hexdigest()
        file.file.seek(0)
        return result
    key = request.headers.get("idempotency-key")
    metadata = [[file.filename, await asyncio.to_thread(file_digest, file)] for file in files] if key else []
    try:
        batch, previous, created = await asyncio.to_thread(
            dispatch.create_batch, repo, fingerprint([metadata, links, asr_model, llm_model]), key,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if not created:
        return previous or await asyncio.to_thread(dispatch.batch_status, repo, batch)

    result = {"batch_id": str(batch), "accepted": [], "rejected": []}
    for index, value in enumerate([*files, *links]):
        try:
            if isinstance(value, str):
                item = await _submit_url(request, UrlRequest(url=value, asr_model=asr_model, llm_model=llm_model), batch)
            else:
                item = (await _submit_upload(request, value, asr_model, llm_model, batch_id=batch)).model_dump()
            result["accepted"].append({"index": index, **item})
        except HTTPException as error:
            result["rejected"].append({"index": index, "detail": error.detail})
    await asyncio.to_thread(dispatch.save_batch_receipt, repo, batch, result)
    return result


def _queue_repository():
    if not dispatch.enabled():
        raise HTTPException(503, "尚未启用 RabbitMQ 任务队列")
    return _history_repository()


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: uuid.UUID):
    value = dispatch.batch_status(_queue_repository(), batch_id)
    if value is None:
        raise HTTPException(404, "批次不存在")
    return value


@app.post("/api/status/{task_id}/cancel")
def cancel_task(task_id: str):
    if not dispatch.cancel(_queue_repository(), task_id):
        raise HTTPException(409, "任务不存在或已结束")
    return {"accepted": True}


@app.post("/api/status/{task_id}/resume")
def resume_task(task_id: str):
    if not dispatch.resume(_queue_repository(), task_id):
        raise HTTPException(409, "任务当前无法续跑")
    return {"accepted": True}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """查询任务状态"""
    return public_state(await asyncio.to_thread(_read_task, task_id))


# ═══════════════════════════════════════════════════════════════
# M10: 多用户配额管理
# ═══════════════════════════════════════════════════════════════


@app.get("/api/quota")
async def get_quota(request: Request):
    """查询当前用户的配额使用情况"""
    client_ip = get_client_ip(request)
    info = quota_manager.get_usage_stats(client_ip)
    return {
        "ip": client_ip,
        "used": info["used"],
        "limit": info["limit"],
        "remaining": info["remaining"],
        "is_whitelisted": info["is_whitelisted"],
    }


# ═══════════════════════════════════════════════════════════════
# M9: 流式输出 SSE 端点
# ═══════════════════════════════════════════════════════════════


@app.get("/api/stream/{task_id}")
async def stream_task_output(task_id: str, request: Request):
    """
    M9: SSE 流式输出任务结果

    事件类型:
    - start: 开始流式输出 {"rec_type": "meeting"|"general"|...}
    - token: 增量 token {"content": "..."}
    - done: 完成 {"output_text": "完整内容"}
    - error: 错误 {"message": "..."}
    - heartbeat: 心跳 (每 15 秒)
    """
    task = await asyncio.to_thread(_read_task, task_id)

    if repository():
        raw_cursor = request.headers.get("last-event-id", "0")
        after = int(raw_cursor) if raw_cursor.isdecimal() and len(raw_cursor) < 19 else 0
        return _history_event_stream(task_id, after)

    # 如果任务已完成，直接返回 SSE 格式的完成事件
    if task["status"] == "completed":
        result_path = task.get("result_path")
        if result_path and Path(result_path).exists():
            content = Path(result_path).read_text(encoding="utf-8")

            async def completed_stream():
                # 直接发送完成事件，不模拟流式
                yield f"data: {json.dumps({'type': 'start', 'data': {'rec_type': task.get('rec_type', 'general')}})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'data': {'output_text': content}})}\n\n"

            return StreamingResponse(
                completed_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

    # 任务未完成或正在进行，创建流式队列
    from agent.graph import register_stream_queue, unregister_stream_queue

    queue = asyncio.Queue()
    register_stream_queue(task_id, queue)

    async def event_generator():
        try:
            last_heartbeat = time.time()

            while True:
                try:
                    # 等待事件，超时 15 秒发心跳
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"

                    # 如果收到 done 或 error 事件，结束流
                    if event["type"] in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat', 'data': {'time': time.time()}})}\n\n"

                    # 检查任务是否已完成
                    latest = await asyncio.to_thread(_read_task, task_id)
                    if latest.get("status") == "completed":
                        result_path = latest.get("result_path")
                        if result_path and Path(result_path).exists():
                            content = Path(result_path).read_text(
                                encoding="utf-8")
                            yield f"data: {json.dumps({'type': 'done', 'data': {'output_text': content}})}\n\n"
                            break
        finally:
            unregister_stream_queue(task_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def _history_event_stream(task_id, after):
    async def generate():
        cursor = after
        repo = repository()
        heartbeat = time.monotonic()
        while True:
            events = await asyncio.to_thread(dispatch.events, repo, task_id, cursor)
            for event in events:
                cursor = event["seq"]
                payload = json.dumps({"type": "status", "data": event["payload"]}, ensure_ascii=False)
                yield f"id: {cursor}\ndata: {payload}\n\n"
            if len(events) == 100:
                continue
            state = await asyncio.to_thread(_read_task, task_id)
            if state["status"] in dispatch.TERMINAL:
                if state["status"] == "completed":
                    content = await asyncio.to_thread(repo.content, task_id, "result")
                    payload = {"type": "done", "data": {"output_text": content or ""}}
                else:
                    payload = {"type": "error", "data": {"message": state.get("error") or state["status"]}}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            if time.monotonic() - heartbeat > 15:
                yield ": heartbeat\n\n"
                heartbeat = time.monotonic()
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _get_content(task_id, kind):
    task = await asyncio.to_thread(_read_task, task_id)
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成")
    repo = repository()
    suffix = {"result": ".md", "questions": "-questions.md", "asr": "-asr.txt"}[kind]
    if repo:
        content = await asyncio.to_thread(repo.content, task_id, kind)
        if content is not None:
            return {"content": content, "filename": (task.get("filename") or task_id) + suffix}
    key = "asr_result_path" if kind == "asr" else "result_path"
    path = Path(task[key]) if task.get(key) else None
    if path and kind == "questions":
        path = path.with_name("question.md")
    if not path or not path.is_file():
        raise HTTPException(404, "结果文件不存在")
    return {"content": await asyncio.to_thread(path.read_text, encoding="utf-8"), "filename": path.name}


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    result = await _get_content(task_id, "result")
    task = await asyncio.to_thread(_read_task, task_id)
    try:
        questions = await _get_content(task_id, "questions") if task.get("rec_type") == "interview" else None
    except HTTPException as error:
        if error.status_code != 404:
            raise
        questions = None
    return {
        **result, "rec_type": task.get("rec_type"),
        "questions_available": bool(questions and questions["content"].strip()),
    }


@app.get("/api/result/{task_id}/download")
async def download_result(task_id: str):
    return await _download_content(task_id, "result")


@app.get("/api/asr/{task_id}")
async def get_asr_result(task_id: str):
    return await _get_content(task_id, "asr")


@app.get("/api/asr/{task_id}/download")
async def download_asr_result(task_id: str):
    return await _download_content(task_id, "asr")


@app.get("/api/questions/{task_id}/download")
async def download_questions(task_id: str):
    return await _download_content(task_id, "questions")


async def _download_content(task_id, kind):
    result = await _get_content(task_id, kind)
    return PlainTextResponse(result["content"], media_type="text/plain" if kind == "asr" else "text/markdown",
                             headers={"Content-Disposition": _content_disposition(result["filename"])})


def _history_repository():
    repo = repository()
    if not repo:
        raise HTTPException(503, "尚未配置 PostgreSQL 历史存储")
    return repo


@app.get("/api/history")
def list_history(limit: int = 30, cursor: str | None = None, status: str | None = None,
                 category: str | None = None, search: str | None = None):
    limit = min(100, max(1, limit))
    position = None
    if cursor:
        try:
            timestamp, task_id = cursor.rsplit("/", 1)
            position = (datetime.fromisoformat(timestamp), task_id)
        except ValueError as error:
            raise HTTPException(422, "分页游标无效") from error
    rows = _history_repository().history(limit, position, status=status, category=category, search=search)
    next_cursor = None
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = f"{last['created_at'].isoformat()}/{last['id']}"
    return {"items": rows[:limit], "next_cursor": next_cursor}


@app.get("/api/history/{task_id}")
def history_detail(task_id: str):
    result = _history_repository().detail(task_id)
    if result is None:
        raise HTTPException(404, "任务不存在")
    return result


@app.get("/api/history/{task_id}/media")
def history_media(task_id: str):
    entry = _history_repository().media_file(task_id)
    if not entry:
        raise HTTPException(404, "媒体文件不存在")
    return FileResponse(entry[0], media_type=entry[1] or "application/octet-stream")


class CachePolicyRequest(BaseModel):
    read: bool
    write: bool
    ttl: int = Field(ge=60, le=365 * 86400)


@app.get("/api/cache")
def cache_list(stage: str | None = None):
    from persistence.stage_cache import list_cache
    return list_cache(_history_repository(), stage)


@app.put("/api/cache/policies/{stage}")
def cache_settings(stage: str, policy: CachePolicyRequest):
    from persistence.stage_cache import LAYERS, set_policy
    if stage not in LAYERS:
        raise HTTPException(404, "缓存阶段不存在")
    set_policy(_history_repository(), stage, policy.read, policy.write, policy.ttl)
    return policy


@app.get("/api/cache/entries/{entry_id}")
def cache_entry(entry_id: uuid.UUID):
    from persistence.stage_cache import cache_preview
    preview = cache_preview(_history_repository(), entry_id)
    if preview is None:
        raise HTTPException(404, "缓存不存在")
    return {"preview": preview}


@app.get("/api/cache/entries/{entry_id}/media")
def cache_media(entry_id: uuid.UUID):
    from persistence.stage_cache import cache_file
    path = cache_file(_history_repository(), entry_id)
    if not path:
        raise HTTPException(404, "缓存媒体不存在")
    return FileResponse(path, filename="cached-media" + path.suffix)


@app.delete("/api/cache/entries/{entry_id}")
def invalidate_cache_entry(entry_id: uuid.UUID):
    from persistence.stage_cache import invalidate
    return {"invalidated": invalidate(_history_repository(), identifier=entry_id)}


@app.delete("/api/cache/stages/{stage}")
def invalidate_cache_stage(stage: str):
    from persistence.stage_cache import LAYERS, invalidate
    if stage not in LAYERS:
        raise HTTPException(404, "缓存阶段不存在")
    return {"invalidated": invalidate(_history_repository(), stage=stage)}


@app.get("/api/gpu")
def gpu_status():
    from gpu_ops.scheduler import configured
    scheduler = configured()
    if not scheduler:
        raise HTTPException(503, "GPU 调度未启用")
    try:
        return scheduler.snapshot()
    except Exception as error:
        raise HTTPException(503, "GPU 监测暂不可用") from error


@app.get("/api/gpu/metrics")
def gpu_metrics():
    data = gpu_status()
    oldest = max((item["wait_seconds"] for item in data["waiting"]), default=0)
    metrics = {
        "meeting_gpu_free_mib": data["free_mib"], "meeting_gpu_total_mib": data["total_mib"],
        "meeting_gpu_compute_processes": data["compute_processes"],
        "meeting_gpu_waiting_jobs": len(data["waiting"]), "meeting_gpu_oldest_wait_seconds": oldest,
        "meeting_gpu_active_lease": int(data["lease"] is not None),
        "meeting_gpu_alert": {"warning": 1, "critical": 2}.get((data["alert"] or {}).get("level"), 0),
    }
    return PlainTextResponse("".join(f"{name} {value}\n" for name, value in metrics.items()))


@app.get("/api/models")
async def get_models():
    """获取可用的 ASR 和 LLM 模型列表"""
    return {
        "asr_models": _asr_models(),
        "llm_models": settings.llm_models,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "api_key_configured": bool(settings.dashscope_api_key)}


# ── Chat (Multi-turn Dialogue) ────────────────────────────────

# M5: 检索意图关键词
RETRIEVAL_KEYWORDS = [
    "上次", "之前", "历史", "对比", "其他会议", "最近", "上个",
    "以前", "过往", "过去", "比较", "回顾", "曾经",
]


def _has_retrieval_intent(text: str) -> bool:
    """M5: 检测用户消息是否包含检索意图"""
    return any(kw in text for kw in RETRIEVAL_KEYWORDS)


async def _retrieve_relevant_history(
    query: str,
    top_k: int = 5,
    coarse_top_k: int = 20,
):
    """
    M5: 三阶段检索 — 粗检索 → Rerank 精排 → 取回父块

    Args:
        query: 用户查询
        top_k: 最终返回数量
        coarse_top_k: 粗检索数量

    Returns:
        SearchResult 列表
    """
    try:
        embed_client = get_embedding_client()
        reranker = get_reranker_client()
        vector_store = get_vector_store()

        # 检查向量库是否为空
        stats = vector_store.get_stats()
        if stats["child_count"] == 0:
            logger.debug("[M5-RAG] 向量库为空，跳过检索")
            return []

        # 1. Embedding 查询
        query_embedding = await embed_client.embed_text(query)

        # 2. 三阶段检索
        results = await vector_store.search_with_rerank(
            query=query,
            query_embedding=query_embedding,
            reranker=reranker,
            coarse_top_k=coarse_top_k,
            final_top_k=top_k,
        )

        logger.debug(
            f"[M5-RAG] 检索完成: {len(results)} 条结果"
        )
        return results

    except Exception as e:
        logger.error(f"[M5-RAG] 检索失败: {e}", exc_info=True)
        return []


CHAT_SYSTEM_PROMPT = """你是一个智能会议分析助手。用户刚刚完成了一段音视频的分析，现在想要基于分析结果进行追问。

## 你的上下文

你拥有以下信息来回答用户的问题:
1. **转写文本**: 音视频的完整转写内容
2. **分析报告**: 已经生成的结构化分析结果 (会议纪要/面试评估/内容总结)
3. **对话历史**: 之前的追问和回答

## 回答规则

- 基于转写文本和分析报告回答，不要编造信息
- 如果用户问的内容在转写文本中没有，如实告知
- 回答要简洁、准确、有条理
- 支持中英文回答"""


@app.post("/api/chat/{task_id}")
async def chat_with_task(task_id: str, req: ChatRequest):
    """基于已完成任务的分析结果进行多轮追问"""
    task = await asyncio.to_thread(_read_task, task_id)
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成，无法追问")

    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(400, "消息不能为空")

    # 获取转写文本作为上下文
    transcript_text = ""
    asr_path = task.get("asr_result_path")
    if asr_path and Path(asr_path).exists():
        transcript_text = Path(asr_path).read_text(encoding="utf-8")

    # 获取分析报告作为上下文
    analysis_text = ""
    result_path = task.get("result_path")
    if result_path and Path(result_path).exists():
        analysis_text = Path(result_path).read_text(encoding="utf-8")

    # M5: RAG 检索增强 — 检测检索意图并注入历史上下文
    rag_context = ""
    if settings.enable_vector_store and _has_retrieval_intent(user_message):
        try:
            rag_results = await _retrieve_relevant_history(user_message)
            if rag_results:
                rag_parts = []
                for r in rag_results:
                    source_label = f"[{r.source_type}/{r.section}]"
                    rag_parts.append(f"{source_label} {r.parent_content}")
                rag_context = "\n\n".join(rag_parts)
                logger.info(
                    f"[M5-RAG] 任务 {task_id}: "
                    f"检索到 {len(rag_results)} 条历史记录"
                )
        except Exception as e:
            logger.warning(f"[M5-RAG] 检索失败: {e}")
            # 降级: 检索失败不影响主流程

    # 使用 ContextManager 进行 token 级别的上下文管理
    ctx_mgr = get_context_manager()

    # M4: 使用 MemoryManager 获取对话历史 (基于 token 预算)
    mem_mgr = get_memory_manager()
    recent_history = mem_mgr.get_recent_messages(task_id, token_budget=20_000)

    # 追加当前用户消息到历史 (用于 token 计算)
    current_messages = recent_history + \
        [{"role": "user", "content": user_message}]

    # 构建完整上下文 (token 预算自动分配)
    # M5: 如果有 RAG 检索结果，注入到 system prompt
    system_prompt = CHAT_SYSTEM_PROMPT
    if rag_context:
        system_prompt += f"\n\n## 相关历史记录\n{rag_context}"

    messages = ctx_mgr.build_chat_context(
        base_system_prompt=system_prompt,
        transcript=transcript_text,
        analysis=analysis_text,
        history=current_messages,
        response_reserve=4000,
    )

    # 调用 LLM
    try:
        from agent.llm_client import LLMClient
        llm = LLMClient()
        reply = await llm.chat_messages(messages, temperature=0.5, max_tokens=4096)

        # 检查空回复
        if not reply or not reply.strip():
            logger.warning(f"[任务 {task_id}] LLM 返回空回复")
            reply = "抱歉，我暂时无法回答这个问题。请稍后重试，或者尝试换一种提问方式。"

        # M4: 保存对话到 MemoryManager (自动触发 L1 蒸馏)
        mem_mgr.add_message(task_id, "user", user_message)
        mem_mgr.add_message(task_id, "assistant", reply)

        # L2: 高低水位线检查 (超过 high_watermark 时自动压缩)
        if mem_mgr.needs_compression(task_id):
            # 生成简单摘要 (后续可升级为 LLM 摘要)
            memory = mem_mgr.get_memory(task_id)
            early_msgs = memory.get_messages_to_compress()
            if early_msgs:
                summary_parts = []
                for m in early_msgs[:6]:
                    text = m["content"][:100]
                    summary_parts.append(f"{m['role']}: {text}")
                auto_summary = "早期对话摘要: " + "; ".join(summary_parts)
                mem_mgr.compress(task_id, auto_summary)
                logger.info(f"[任务 {task_id}] L2 压缩已执行")

        stats = mem_mgr.get_stats(task_id)
        logger.info(
            f"[任务 {task_id}] 追问对话: user='{user_message[:50]}...' → "
            f"reply={len(reply)} chars, "
            f"rounds={stats['rounds']}, messages={stats['messages']}, "
            f"tokens={stats['tokens']}"
        )

        return {"reply": reply}

    except Exception as e:
        logger.error(f"[任务 {task_id}] 追问失败: {e}")
        raise HTTPException(500, f"对话失败: {e}")


# M9: 任务追问对话流式输出版本
@app.post("/api/chat/{task_id}/stream")
async def chat_stream(task_id: str, req: ChatRequest):
    """
    M9: 任务追问对话流式输出版本

    使用 SSE 流式输出 LLM 回答
    """
    task = await asyncio.to_thread(_read_task, task_id)
    if task["status"] != "completed":
        raise HTTPException(400, "任务尚未完成，无法追问")

    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(400, "消息不能为空")

    logger.info(f"[任务 {task_id}-Stream] 追问对话: user='{user_message[:50]}...'")

    # 获取转写文本作为上下文
    transcript_text = ""
    asr_path = task.get("asr_result_path")
    if asr_path and Path(asr_path).exists():
        transcript_text = Path(asr_path).read_text(encoding="utf-8")

    # 获取分析报告作为上下文
    analysis_text = ""
    result_path = task.get("result_path")
    if result_path and Path(result_path).exists():
        analysis_text = Path(result_path).read_text(encoding="utf-8")

    # M5: RAG 检索增强
    rag_context = ""
    if settings.enable_vector_store and _has_retrieval_intent(user_message):
        try:
            rag_results = await _retrieve_relevant_history(user_message)
            if rag_results:
                rag_parts = []
                for r in rag_results:
                    source_label = f"[{r.source_type}/{r.section}]"
                    rag_parts.append(f"{source_label} {r.parent_content}")
                rag_context = "\n\n".join(rag_parts)
                logger.info(
                    f"[任务 {task_id}-Stream] 检索到 {len(rag_results)} 条历史记录")
        except Exception as e:
            logger.warning(f"[任务 {task_id}-Stream] RAG 检索失败: {e}")

    # 使用 ContextManager 进行 token 级别的上下文管理
    ctx_mgr = get_context_manager()
    mem_mgr = get_memory_manager()
    recent_history = mem_mgr.get_recent_messages(task_id, token_budget=20_000)
    current_messages = recent_history + \
        [{"role": "user", "content": user_message}]

    # 构建完整上下文
    system_prompt = CHAT_SYSTEM_PROMPT
    if rag_context:
        system_prompt += f"\n\n## 相关历史记录\n{rag_context}"

    messages = ctx_mgr.build_chat_context(
        base_system_prompt=system_prompt,
        transcript=transcript_text,
        analysis=analysis_text,
        history=current_messages,
        response_reserve=4000,
    )

    async def event_generator():
        try:
            from agent.llm_client import LLMClient
            llm = LLMClient()

            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # 流式调用 LLM
            full_content = ""
            async for token in llm.chat_messages_stream(messages, temperature=0.5, max_tokens=4096):
                full_content += token
                yield f"data: {json.dumps({'type': 'token', 'data': {'content': token}})}\n\n"

            # 检查空回复
            if not full_content or not full_content.strip():
                logger.warning(f"[任务 {task_id}-Stream] LLM 返回空回复")
                full_content = "抱歉，我暂时无法回答这个问题。请稍后重试，或者尝试换一种提问方式。"
                yield f"data: {json.dumps({'type': 'token', 'data': {'content': full_content}})}\n\n"

            # 保存对话历史
            mem_mgr.add_message(task_id, "user", user_message)
            mem_mgr.add_message(task_id, "assistant", full_content)

            # L2: 高低水位线检查
            if mem_mgr.needs_compression(task_id):
                memory = mem_mgr.get_memory(task_id)
                early_msgs = memory.get_messages_to_compress()
                if early_msgs:
                    summary_parts = []
                    for m in early_msgs[:6]:
                        text = m["content"][:100]
                        summary_parts.append(f"{m['role']}: {text}")
                    auto_summary = "早期对话摘要: " + "; ".join(summary_parts)
                    mem_mgr.compress(task_id, auto_summary)
                    logger.info(f"[任务 {task_id}-Stream] L2 压缩已执行")

            # 发送完成事件
            yield f"data: {json.dumps({'type': 'done', 'data': {'reply': full_content}})}\n\n"

            stats = mem_mgr.get_stats(task_id)
            logger.info(
                f"[任务 {task_id}-Stream] 追问对话完成: "
                f"reply={len(full_content)} chars, "
                f"rounds={stats['rounds']}, messages={stats['messages']}"
            )
        except Exception as e:
            logger.error(f"[任务 {task_id}-Stream] 对话失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/chat/{task_id}/history")
async def get_chat_history(task_id: str):
    """获取任务的对话历史 (M4: 三层分级压缩)"""
    await asyncio.to_thread(_read_task, task_id)
    mem_mgr = get_memory_manager()
    memory = mem_mgr.get_memory(task_id)
    return {
        "messages": memory.messages,
        "summary": memory.summary,
        "stats": mem_mgr.get_stats(task_id),
        "milestones": [
            {"label": m.label, "token_count": m.token_count}
            for m in memory.milestones
        ],
    }


class MilestoneRequest(BaseModel):
    label: str
    summary: str = ""


@app.post("/api/chat/{task_id}/milestone")
async def mark_milestone(task_id: str, req: MilestoneRequest):
    """
    L3: 标记里程碑，归档当前对话

    当一个子任务完成时调用 (如 "分析完成", "待办提取完成")
    将当前对话压缩为摘要，只保留最近的少量消息。
    """
    await asyncio.to_thread(_read_task, task_id)
    mem_mgr = get_memory_manager()
    milestone = mem_mgr.mark_milestone(task_id, req.label, req.summary)
    return {
        "milestone": {
            "label": milestone.label,
            "token_count": milestone.token_count,
        },
        "stats": mem_mgr.get_stats(task_id),
    }


# ── Global Chat (跨任务 RAG 对话) ───────────────────────────────

GLOBAL_CHAT_SYSTEM_PROMPT = """你是一个智能会议分析助手。你可以通过搜索历史记录来回答用户的问题。

## 你的上下文

你拥有以下信息来回答用户的问题:
1. **相关历史记录**: 从过去的会议/面试/视频分析中检索到的相关内容
2. **对话历史**: 之前的追问和回答

## 回答规则

- 基于检索到的历史记录回答，不要编造信息
- 如果检索结果中没有相关信息，如实告知
- 回答要简洁、准确、有条理
- 支持中英文回答
- 引用来源时说明是来自哪个会议/视频的分析"""


class GlobalChatRequest(BaseModel):
    message: str
    session_id: str = "default"


# 全局对话历史存储 (按 session_id 隔离)
_global_chat_histories: dict[str, list[dict]] = {}


@app.post("/api/global-chat")
async def global_chat(req: GlobalChatRequest):
    """
    全局 RAG 对话 — 无需上传文件，直接查询历史分析记录

    流程:
    1. 检测检索意图
    2. 向量检索 + Rerank 精排
    3. 注入检索结果到 system prompt
    4. LLM 生成回答
    """
    user_message = req.message.strip()
    session_id = req.session_id or "default"

    if not user_message:
        raise HTTPException(400, "消息不能为空")

    logger.info(
        f"[GlobalChat] session={session_id}: "
        f"user='{user_message[:60]}'"
    )

    # 获取/初始化会话历史
    if session_id not in _global_chat_histories:
        _global_chat_histories[session_id] = []
    history = _global_chat_histories[session_id]

    # M5: RAG 检索
    rag_context = ""
    if settings.enable_vector_store:
        try:
            rag_results = await _retrieve_relevant_history(user_message)
            if rag_results:
                rag_parts = []
                for r in rag_results:
                    source_label = f"[{r.source_type}/{r.section}]"
                    rag_parts.append(f"{source_label} {r.parent_content}")
                rag_context = "\n\n".join(rag_parts)
                logger.info(
                    f"[GlobalChat] 检索到 {len(rag_results)} 条历史记录"
                )
        except Exception as e:
            logger.warning(f"[GlobalChat] 检索失败: {e}")

    # 构建 system prompt
    system_prompt = GLOBAL_CHAT_SYSTEM_PROMPT
    if rag_context:
        system_prompt += f"\n\n## 相关历史记录\n{rag_context}"
    else:
        return {"reply": "当前没有可引用的会议记录。请先确认记录范围和知识构建状态。", "status": "no_match", "evidence": []}

    # 构建 messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-20:])  # 最近 10 轮 (20 条消息)
    messages.append({"role": "user", "content": user_message})

    # 调用 LLM
    try:
        from agent.llm_client import LLMClient
        llm = LLMClient()
        reply = await llm.chat_messages(messages, temperature=0.5, max_tokens=4096)

        # 检查空回复
        if not reply or not reply.strip():
            logger.warning(f"[任务 {task_id}] LLM 返回空回复")
            reply = "抱歉，我暂时无法回答这个问题。请稍后重试，或者尝试换一种提问方式。"

        # 保存历史
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        # 限制历史长度 (保留最近 40 条)
        if len(history) > 40:
            _global_chat_histories[session_id] = history[-40:]

        logger.info(
            f"[GlobalChat] session={session_id}: "
            f"reply={len(reply)} chars, history={len(_global_chat_histories[session_id])} msgs"
        )
        return {"reply": reply}

    except Exception as e:
        logger.error(f"[GlobalChat] 对话失败: {e}")
        raise HTTPException(500, f"对话失败: {e}")


# M9: 全局聊天流式输出版本
@app.post("/api/global-chat/stream")
async def global_chat_stream(req: GlobalChatRequest):
    """
    M9: 全局 RAG 对话流式输出版本

    使用 SSE 流式输出 LLM 回答
    """
    user_message = req.message.strip()
    session_id = req.session_id or "default"

    if not user_message:
        raise HTTPException(400, "消息不能为空")

    logger.info(
        f"[GlobalChat-Stream] session={session_id}: user='{user_message[:60]}'")

    # 获取/初始化会话历史
    if session_id not in _global_chat_histories:
        _global_chat_histories[session_id] = []
    history = _global_chat_histories[session_id]

    # M5: RAG 检索
    rag_context = ""
    try:
        rag_results = await _retrieve_relevant_history(user_message)
        if rag_results:
            rag_parts = []
            for r in rag_results:
                source_label = f"[{r.source_type}/{r.section}]"
                rag_parts.append(f"{source_label} {r.parent_content}")
            rag_context = "\n\n".join(rag_parts)
            logger.info(f"[GlobalChat-Stream] 检索到 {len(rag_results)} 条历史记录")
    except Exception as e:
        logger.warning(f"[GlobalChat-Stream] 检索失败: {e}")

    # 构建 system prompt
    system_prompt = GLOBAL_CHAT_SYSTEM_PROMPT
    if rag_context:
        system_prompt += f"\n\n## 相关历史记录\n{rag_context}"
    else:
        async def missing_evidence():
            payload = {"type": "done", "data": {"reply": "当前没有可引用的会议记录。请先确认记录范围和知识构建状态。", "status": "no_match"}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return StreamingResponse(missing_evidence(), media_type="text/event-stream")

    # 构建 messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": user_message})

    async def event_generator():
        try:
            from agent.llm_client import LLMClient
            llm = LLMClient()

            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # 流式调用 LLM
            full_content = ""
            async for token in llm.chat_messages_stream(messages, temperature=0.5, max_tokens=4096):
                full_content += token
                yield f"data: {json.dumps({'type': 'token', 'data': {'content': token}})}\n\n"

            # 检查空回复
            if not full_content or not full_content.strip():
                logger.warning(f"[GlobalChat-Stream] LLM 返回空回复")
                full_content = "抱歉，我暂时无法回答这个问题。请稍后重试，或者尝试换一种提问方式。"
                yield f"data: {json.dumps({'type': 'token', 'data': {'content': full_content}})}\n\n"

            # 保存历史
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_content})

            # 限制历史长度
            if len(history) > 40:
                _global_chat_histories[session_id] = history[-40:]

            # 发送完成事件
            yield f"data: {json.dumps({'type': 'done', 'data': {'reply': full_content}})}\n\n"

            logger.info(
                f"[GlobalChat-Stream] session={session_id}: "
                f"reply={len(full_content)} chars, history={len(_global_chat_histories[session_id])} msgs"
            )
        except Exception as e:
            logger.error(f"[GlobalChat-Stream] 对话失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/global-chat/history")
async def get_global_chat_history(session_id: str = "default"):
    """获取全局对话历史"""
    history = _global_chat_histories.get(session_id, [])
    return {"messages": history, "session_id": session_id}


@app.delete("/api/global-chat/history")
async def clear_global_chat_history(session_id: str = "default"):
    """清空全局对话历史"""
    if session_id in _global_chat_histories:
        del _global_chat_histories[session_id]
    logger.info(f"[GlobalChat] session={session_id} 历史已清空")
    return {"status": "cleared"}


@app.get("/api/vector-store/stats")
async def vector_store_stats():
    """获取向量库统计信息 (调试用)"""
    try:
        store = get_vector_store()
        return store.get_stats()
    except Exception as e:
        return {"error": str(e), "child_count": 0, "parent_count": 0}


# ── Serve Frontend ───────────────────────────────────────────

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir),
              html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9090, proxy_headers=False)
