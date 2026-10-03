"""会议助手智能体 - Web 服务 (FastAPI)"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings

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


class UploadResponse(BaseModel):
    task_id: str
    filename: str
    file_type: str


class UrlRequest(BaseModel):
    url: str


# ── Task Store ───────────────────────────────────────────────

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _update(task_id: str, **kw):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kw)


# ── Audio Processing (background thread) ─────────────────────


def _process_audio(task_id: str, file_path: Path):
    """后台线程: 处理音频文件"""
    try:
        from agent.cache import get_audio_stem, save_asr_cache, save_summary_cache, load_asr_cache, load_summary_cache
        from agent.asr_client import ASRClient
        from agent.text_processor import TextProcessor
        from agent.classifier import classify_recording
        from agent.llm_client import LLMClient
        from agent.prompts import MEETING_ANALYSIS_PROMPT
        from agent.pipeline import (
            analyze_interview, analyze_general,
            format_interview_questions, format_interview_analysis,
            format_general_result,
        )
        from models.schemas import MeetingResult
        from utils.audio import preprocess_audio

        audio_stem = get_audio_stem(str(file_path))
        _update(task_id, step="正在预处理音频...", progress=10)

        # 1. 预处理
        preprocessed = preprocess_audio(
            file_path, sample_rate=settings.asr_sample_rate)

        # 2. ASR
        _update(task_id, step="正在语音转写...", progress=20)
        asr = ASRClient()
        t0 = time.time()
        transcript = asyncio.run(asr.transcribe(str(preprocessed)))
        asr_duration = time.time() - t0

        # 清理预处理文件
        try:
            preprocessed.unlink()
        except Exception:
            pass

        processor = TextProcessor()
        cleaned = processor.clean(transcript)
        transcript_text = cleaned.formatted_text

        if not transcript_text.strip():
            _update(task_id, status="failed", error="ASR 转写结果为空")
            return

        # 保存 ASR 缓存
        asr_path = save_asr_cache(transcript_text, audio_stem)
        _update(task_id, asr_result_path=str(asr_path))

        # 3. 分类
        _update(task_id, step="正在识别录音类型...", progress=50)
        rec_type = asyncio.run(classify_recording(transcript_text))

        # 4. 分析
        llm_start = time.time()
        total_start = time.time() - asr_duration  # 近似
        type_labels = {"meeting": "会议录音",
                       "interview": "面试录音", "general": "其他录音"}
        _update(
            task_id, step=f"正在分析 ({type_labels.get(rec_type, rec_type)})...", progress=60)

        llm = LLMClient()

        if rec_type == "meeting":
            prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript_text)
            data = asyncio.run(llm.chat_json(prompt))
            result = MeetingResult.model_validate(data)
            timing_info = _build_timing(asr_duration, llm_start)
            from agent.pipeline import format_markdown
            output_text = format_markdown(
                result, event_time=None, timing_info=timing_info)
            save_summary_cache(output_text, audio_stem, rec_type="meeting")
            _save_result(task_id, audio_stem, "meeting", output_text)

        elif rec_type == "interview":
            result_raw = asyncio.run(analyze_interview(transcript_text))
            timing_info = _build_timing(asr_duration, llm_start)
            q_md = format_interview_questions(
                result_raw["q_data"], None, timing_info)
            a_md = format_interview_analysis(
                result_raw["a_data"], None, timing_info)
            # 面试保存为目录
            result_dir = RESULT_DIR / "interview" / \
                f"{audio_stem}_{settings.asr_model}_{settings.llm_model}"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "question.md").write_text(q_md, encoding="utf-8")
            (result_dir / "analyze.md").write_text(a_md, encoding="utf-8")
            _update(task_id, status="completed", progress=100,
                    step="分析完成", result_path=str(result_dir / "analyze.md"))
            return

        else:  # general
            data = asyncio.run(analyze_general(transcript_text))
            timing_info = _build_timing(asr_duration, llm_start)
            output_text = format_general_result(data, None, timing_info)
            save_summary_cache(output_text, audio_stem, rec_type="general")
            _save_result(task_id, audio_stem, "other", output_text)

        _update(task_id, status="completed", progress=100, step="分析完成")

    except Exception as e:
        _update(task_id, status="failed", error=str(e), step="处理失败")


# ── Video Processing (background thread) ─────────────────────


def _process_video(task_id: str, file_path: Path):
    """后台线程: 处理视频文件"""
    try:
        from agent.cache import get_audio_stem, sanitize_stem, save_asr_cache, save_summary_cache, load_asr_cache, load_summary_cache
        from agent.video_client import get_video_info, extract_audio_from_video
        from agent.asr_client import ASRClient
        from agent.text_processor import TextProcessor
        from agent.pipeline import analyze_general, format_general_result
        from utils.audio import preprocess_audio

        source = str(file_path)
        video_stem = get_audio_stem(source)
        video_title = None

        # 1. 视频信息
        _update(task_id, step="正在获取视频信息...", progress=5)
        video_info = asyncio.run(get_video_info(source))
        if video_info:
            video_title = video_info.get("title", "")
            if video_title:
                video_stem = sanitize_stem(video_title)
            _update(task_id, video_title=video_title)

        # 2. 提取音频
        _update(task_id, step="正在提取音频...", progress=15)
        audio_path = asyncio.run(extract_audio_from_video(
            source, filename_stem=video_stem))

        # 3. 预处理
        _update(task_id, step="正在预处理音频...", progress=25)
        preprocessed = preprocess_audio(
            audio_path, sample_rate=settings.asr_sample_rate)
        try:
            audio_path.unlink()
        except Exception:
            pass

        # 4. ASR
        _update(task_id, step="正在语音转写...", progress=35)
        asr = ASRClient()
        t0 = time.time()
        transcript = asyncio.run(asr.transcribe(str(preprocessed)))
        asr_duration = time.time() - t0

        try:
            preprocessed.unlink()
        except Exception:
            pass

        processor = TextProcessor()
        cleaned = processor.clean(transcript)
        transcript_text = cleaned.formatted_text

        if not transcript_text.strip():
            _update(task_id, status="failed", error="ASR 转写结果为空")
            return

        asr_path = save_asr_cache(transcript_text, video_stem)
        _update(task_id, asr_result_path=str(asr_path))

        # 5. LLM 分析
        _update(task_id, step="正在分析内容...", progress=65)
        llm_start = time.time()

        cached = load_summary_cache(video_stem, rec_type="video")
        if cached is not None:
            output_text = cached
        else:
            data = asyncio.run(analyze_general(transcript_text))
            timing_info = _build_timing(asr_duration, llm_start)
            output_text = format_general_result(data, None, timing_info)
            save_summary_cache(output_text, video_stem, rec_type="video")

        _save_result(task_id, video_stem, "video", output_text)
        _update(task_id, status="completed", progress=100, step="分析完成")

    except Exception as e:
        _update(task_id, status="failed", error=str(e), step="处理失败")


# ── Helpers ──────────────────────────────────────────────────


def _build_timing(asr_duration: float, llm_start: float) -> dict:
    return {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "asr_duration": asr_duration,
        "llm_duration": time.time() - llm_start,
        "total_duration": time.time() - (llm_start - asr_duration),
    }


def _save_result(task_id: str, stem: str, subdir: str, text: str):
    result_path = RESULT_DIR / subdir / \
        f"{stem}_{settings.asr_model}_{settings.llm_model}.md"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(text, encoding="utf-8")
    _update(task_id, result_path=str(result_path))


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
async def upload_file(file: UploadFile = File(...)):
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

    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "step": "任务已创建",
            "error": None,
            "result_path": None,
            "asr_result_path": None,
            "filename": file.filename or "",
            "file_type": file_type,
            "video_title": None,
        }

    target = _process_video if file_type == "video" else _process_audio
    threading.Thread(target=target, args=(
        task_id, file_path), daemon=True).start()

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
    task_id = _new_task_id()
    display_name = url[:80] + ("..." if len(url) > 80 else "")

    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "step": "任务已创建",
            "error": None,
            "result_path": None,
            "asr_result_path": None,
            "filename": display_name,
            "file_type": file_type,
            "video_title": None,
        }

    if is_video:
        threading.Thread(target=_process_video_url, args=(
            task_id, url), daemon=True).start()
    else:
        threading.Thread(target=_process_audio_url, args=(
            task_id, url), daemon=True).start()

    return {"task_id": task_id, "filename": display_name, "file_type": file_type}


# ── URL Processing (background threads) ─────────────────────


def _process_video_url(task_id: str, url: str):
    """后台线程: 处理视频 URL"""
    try:
        from agent.cache import get_audio_stem, sanitize_stem, save_asr_cache, save_summary_cache, load_summary_cache
        from agent.video_client import get_video_info, extract_audio_from_video
        from agent.asr_client import ASRClient
        from agent.text_processor import TextProcessor
        from agent.pipeline import analyze_general, format_general_result
        from utils.audio import preprocess_audio

        video_stem = get_audio_stem(url)
        video_title = None

        # 1. 视频信息
        _update(task_id, step="正在获取视频信息...", progress=5)
        video_info = asyncio.run(get_video_info(url))
        if video_info:
            video_title = video_info.get("title", "")
            if video_title:
                video_stem = sanitize_stem(video_title)
            _update(task_id, video_title=video_title,
                    filename=video_title or url[:60])

        # 2. 提取音频
        _update(task_id, step="正在提取音频...", progress=15)
        audio_path = asyncio.run(
            extract_audio_from_video(url, filename_stem=video_stem))

        # 3. 预处理
        _update(task_id, step="正在预处理音频...", progress=25)
        preprocessed = preprocess_audio(
            audio_path, sample_rate=settings.asr_sample_rate)
        try:
            audio_path.unlink()
        except Exception:
            pass

        # 4. ASR
        _update(task_id, step="正在语音转写...", progress=35)
        asr = ASRClient()
        t0 = time.time()
        transcript = asyncio.run(asr.transcribe(str(preprocessed)))
        asr_duration = time.time() - t0
        try:
            preprocessed.unlink()
        except Exception:
            pass

        processor = TextProcessor()
        cleaned = processor.clean(transcript)
        transcript_text = cleaned.formatted_text
        if not transcript_text.strip():
            _update(task_id, status="failed", error="ASR 转写结果为空")
            return
        asr_path = save_asr_cache(transcript_text, video_stem)
        _update(task_id, asr_result_path=str(asr_path))

        # 5. LLM 分析
        _update(task_id, step="正在分析内容...", progress=65)
        llm_start = time.time()
        cached = load_summary_cache(video_stem, rec_type="video")
        if cached is not None:
            output_text = cached
        else:
            data = asyncio.run(analyze_general(transcript_text))
            timing_info = _build_timing(asr_duration, llm_start)
            output_text = format_general_result(data, None, timing_info)
            save_summary_cache(output_text, video_stem, rec_type="video")

        _save_result(task_id, video_stem, "video", output_text)
        _update(task_id, status="completed", progress=100, step="分析完成")

    except Exception as e:
        _update(task_id, status="failed", error=str(e), step="处理失败")


def _process_audio_url(task_id: str, url: str):
    """后台线程: 处理音频 URL"""
    try:
        from agent.cache import get_audio_stem, save_asr_cache, save_summary_cache, load_asr_cache
        from agent.asr_client import ASRClient
        from agent.text_processor import TextProcessor
        from agent.classifier import classify_recording
        from agent.llm_client import LLMClient
        from agent.prompts import MEETING_ANALYSIS_PROMPT
        from agent.pipeline import (
            analyze_interview, analyze_general,
            format_interview_questions, format_interview_analysis,
            format_general_result, format_markdown,
        )
        from models.schemas import MeetingResult

        audio_stem = get_audio_stem(url)
        _update(task_id, step="正在语音转写...", progress=20)

        asr = ASRClient()
        t0 = time.time()
        transcript = asyncio.run(asr.transcribe(url))
        asr_duration = time.time() - t0

        processor = TextProcessor()
        cleaned = processor.clean(transcript)
        transcript_text = cleaned.formatted_text
        if not transcript_text.strip():
            _update(task_id, status="failed", error="ASR 转写结果为空")
            return
        asr_path = save_asr_cache(transcript_text, audio_stem)
        _update(task_id, asr_result_path=str(asr_path))

        _update(task_id, step="正在识别录音类型...", progress=50)
        rec_type = asyncio.run(classify_recording(transcript_text))

        llm_start = time.time()
        type_labels = {"meeting": "会议录音",
                       "interview": "面试录音", "general": "其他录音"}
        _update(
            task_id, step=f"正在分析 ({type_labels.get(rec_type, rec_type)})...", progress=60)

        llm = LLMClient()
        if rec_type == "meeting":
            prompt = MEETING_ANALYSIS_PROMPT.format(transcript=transcript_text)
            data = asyncio.run(llm.chat_json(prompt))
            result = MeetingResult.model_validate(data)
            timing_info = _build_timing(asr_duration, llm_start)
            output_text = format_markdown(
                result, event_time=None, timing_info=timing_info)
            save_summary_cache(output_text, audio_stem, rec_type="meeting")
            _save_result(task_id, audio_stem, "meeting", output_text)
        elif rec_type == "interview":
            result_raw = asyncio.run(analyze_interview(transcript_text))
            timing_info = _build_timing(asr_duration, llm_start)
            a_md = format_interview_analysis(
                result_raw["a_data"], None, timing_info)
            result_dir = RESULT_DIR / "interview" / \
                f"{audio_stem}_{settings.asr_model}_{settings.llm_model}"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "analyze.md").write_text(a_md, encoding="utf-8")
            _update(task_id, status="completed", progress=100, step="分析完成",
                    result_path=str(result_dir / "analyze.md"))
            return
        else:
            data = asyncio.run(analyze_general(transcript_text))
            timing_info = _build_timing(asr_duration, llm_start)
            output_text = format_general_result(data, None, timing_info)
            save_summary_cache(output_text, audio_stem, rec_type="general")
            _save_result(task_id, audio_stem, "other", output_text)

        _update(task_id, status="completed", progress=100, step="分析完成")

    except Exception as e:
        _update(task_id, status="failed", error=str(e), step="处理失败")


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
                             headers={"Content-Disposition": f'attachment; filename="{Path(rp).name}"'})


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
                             headers={"Content-Disposition": f'attachment; filename="{Path(asr_path).name}"'})


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
