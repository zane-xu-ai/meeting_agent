"""会议助手 - 结果缓存管理

缓存目录结构:
    output/asr/{audio_stem}_{asr_model}.txt                                  -- ASR 转写结果
    output/summary/meeting/{audio_stem}_{asr_model}_{llm_model}.md           -- 会议总结
    output/summary/interview/{audio_stem}_{asr_model}_{llm_model}/question.md -- 面试问题列表
    output/summary/interview/{audio_stem}_{asr_model}_{llm_model}/analyze.md  -- 面试逐题分析
    output/summary/other/{audio_stem}_{asr_model}_{llm_model}.md             -- 其他录音总结
    output/summary/video/                                                     -- 视频总结(预留)
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from loguru import logger

from config import settings

# 录音类型 -> summary 子目录名
_TYPE_TO_SUBDIR = {
    "meeting": "meeting",
    "interview": "interview",
    "general": "other",
    "video": "video",
}


def get_audio_stem(audio_source: str) -> str:
    """
    从音频路径或 URL 中提取文件名(不含后缀)。

    本地路径: /path/to/meeting.wav -> meeting
    URL: https://.../test/one_sent.wav?Expires=... -> one_sent
    """
    if audio_source.startswith(("http://", "https://")):
        parsed = urlparse(audio_source)
        # 取路径部分的最后一段，并解码 URL 编码
        path_stem = Path(unquote(parsed.path)).stem
        return path_stem
    else:
        return Path(audio_source).stem


def _rec_type_to_subdir(rec_type: str) -> str:
    """将录音类型映射为 summary 子目录名"""
    return _TYPE_TO_SUBDIR.get(rec_type, "other")


# ---------- ASR 缓存 ----------

def _asr_cache_path(audio_stem: str, asr_model: str | None = None) -> Path:
    """ASR 缓存文件路径"""
    model = asr_model or settings.asr_model
    return settings.output_dir / "asr" / f"{audio_stem}_{model}.txt"


def load_asr_cache(audio_stem: str, asr_model: str | None = None) -> str | None:
    """
    尝试加载 ASR 缓存。命中返回转写文本，未命中返回 None。
    """
    cache_path = _asr_cache_path(audio_stem, asr_model)
    if cache_path.exists():
        logger.info(f"[缓存命中] ASR 结果: {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")
    return None


def save_asr_cache(
    transcript_text: str, audio_stem: str, asr_model: str | None = None
) -> Path:
    """保存 ASR 转写文本到缓存"""
    cache_path = _asr_cache_path(audio_stem, asr_model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(transcript_text, encoding="utf-8")
    logger.info(f"[缓存保存] ASR 结果: {cache_path}")
    return cache_path


# ---------- Summary 缓存 (会议/通用: 单文件) ----------

def _summary_cache_path(
    audio_stem: str,
    rec_type: str = "meeting",
    asr_model: str | None = None,
    llm_model: str | None = None,
) -> Path:
    """Summary 缓存文件路径 (会议/通用类型)"""
    am = asr_model or settings.asr_model
    lm = llm_model or settings.llm_model
    subdir = _rec_type_to_subdir(rec_type)
    return settings.output_dir / "summary" / subdir / f"{audio_stem}_{am}_{lm}.md"


def load_summary_cache(
    audio_stem: str,
    rec_type: str = "meeting",
    asr_model: str | None = None,
    llm_model: str | None = None,
) -> str | None:
    """
    尝试加载 Summary 缓存 (会议/通用)。命中返回 Markdown 文本，未命中返回 None。
    """
    cache_path = _summary_cache_path(audio_stem, rec_type, asr_model, llm_model)
    if cache_path.exists():
        logger.info(f"[缓存命中] Summary 结果: {cache_path}")
        return cache_path.read_text(encoding="utf-8")
    return None


def save_summary_cache(
    markdown_text: str,
    audio_stem: str,
    rec_type: str = "meeting",
    asr_model: str | None = None,
    llm_model: str | None = None,
) -> Path:
    """保存 Summary 结果到缓存 (会议/通用类型)"""
    cache_path = _summary_cache_path(audio_stem, rec_type, asr_model, llm_model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(markdown_text, encoding="utf-8")
    logger.info(f"[缓存保存] Summary 结果: {cache_path}")
    return cache_path


# ---------- Summary 缓存 (面试: 目录结构) ----------

def _interview_cache_dir(
    audio_stem: str,
    asr_model: str | None = None,
    llm_model: str | None = None,
) -> Path:
    """面试录音缓存目录路径"""
    am = asr_model or settings.asr_model
    lm = llm_model or settings.llm_model
    return settings.output_dir / "summary" / "interview" / f"{audio_stem}_{am}_{lm}"


def load_interview_cache(
    audio_stem: str,
    asr_model: str | None = None,
    llm_model: str | None = None,
) -> dict[str, str] | None:
    """
    尝试加载面试录音缓存。

    命中返回 {"question": "...", "analyze": "..."}，未命中返回 None。
    """
    cache_dir = _interview_cache_dir(audio_stem, asr_model, llm_model)
    q_path = cache_dir / "question.md"
    a_path = cache_dir / "analyze.md"

    if q_path.exists() and a_path.exists():
        logger.info(f"[缓存命中] 面试分析结果: {cache_dir}/")
        return {
            "question": q_path.read_text(encoding="utf-8"),
            "analyze": a_path.read_text(encoding="utf-8"),
        }
    return None


def save_interview_cache(
    question_md: str,
    analyze_md: str,
    audio_stem: str,
    asr_model: str | None = None,
    llm_model: str | None = None,
) -> Path:
    """保存面试录音分析结果到缓存目录"""
    cache_dir = _interview_cache_dir(audio_stem, asr_model, llm_model)
    cache_dir.mkdir(parents=True, exist_ok=True)

    q_path = cache_dir / "question.md"
    a_path = cache_dir / "analyze.md"
    q_path.write_text(question_md, encoding="utf-8")
    a_path.write_text(analyze_md, encoding="utf-8")

    logger.info(f"[缓存保存] 面试分析结果: {cache_dir}/")
    return cache_dir
