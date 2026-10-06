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
from urllib.parse import quote

from loguru import logger
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from agent.context import get_context_manager
from agent.memory import get_memory_manager

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


class ChatRequest(BaseModel):
    message: str


# ── Task Store ───────────────────────────────────────────────

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()

# ── M4: Short-term Memory Store ─────────────────────────────────
# 使用 MemoryManager 管理对话记忆 (替代原有的 chat_histories 字典)
# 支持滑动窗口、历史压缩等功能


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _update(task_id: str, **kw):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kw)


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
        # 本地文件用原始文件名做缓存 key，URL 用 source 路径
        if source_type.startswith("local_") and filename:
            audio_stem = get_audio_stem(filename)
        else:
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

        # M5: 后台线程索引到向量库 (独立线程 + 独立 event loop)
        import threading
        task_data = dict(tasks.get(task_id, {}))

        def _run_index():
            import asyncio
            asyncio.run(_index_task_to_vector_store(task_id, task_data))
        threading.Thread(target=_run_index, daemon=True,
                         name=f"index-{task_id}").start()

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


# ═══════════════════════════════════════════════════════════════
# M9: 流式输出 SSE 端点
# ═══════════════════════════════════════════════════════════════


@app.get("/api/stream/{task_id}")
async def stream_task_output(task_id: str):
    """
    M9: SSE 流式输出任务结果

    事件类型:
    - start: 开始流式输出 {"rec_type": "meeting"|"general"|...}
    - token: 增量 token {"content": "..."}
    - done: 完成 {"output_text": "完整内容"}
    - error: 错误 {"message": "..."}
    - heartbeat: 心跳 (每 15 秒)
    """
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")

    task = tasks[task_id]

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
    import asyncio
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
                    if tasks.get(task_id, {}).get("status") == "completed":
                        result_path = tasks[task_id].get("result_path")
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
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
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
    if _has_retrieval_intent(user_message):
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
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
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
    if _has_retrieval_intent(user_message):
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
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
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
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
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
        system_prompt += "\n\n## 提示\n当前向量库中无相关历史记录，请基于你的知识回答。"

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
        system_prompt += "\n\n## 提示\n当前向量库中无相关历史记录，请基于你的知识回答。"

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
    uvicorn.run(app, host="0.0.0.0", port=9090)
