"""会议助手 - 视频处理客户端

支持:
- 本地视频文件 (mp4/mkv/avi/mov/webm/flv 等)
- 网络视频链接 (YouTube, Bilibili 等 1000+ 站点，由 yt-dlp 支持)
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

# 视频文件扩展名
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".wmv", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts",
}

# 已知的视频平台 URL 模式
VIDEO_URL_PATTERNS = [
    r"(?:www\.)?youtube\.com",
    r"(?:www\.)?youtu\.be",
    r"(?:www\.)?bilibili\.com",
    r"(?:www\.)?b23\.tv",
    r"bili2233\.cn",
]

# 默认 HTTP 头 (模拟浏览器，绕过反爬虫)
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 默认 cookies 文件路径 (Netscape 格式)
# 支持两种位置:
#   1. data/cookies.txt
#   2. data/cookies/*.txt (按域名匹配)
DEFAULT_COOKIES_FILE = Path(__file__).parent.parent / "data" / "cookies.txt"
COOKIES_DIR = Path(__file__).parent.parent / "data" / "cookies"


def is_video_file(path: str | Path) -> bool:
    """判断本地文件是否为视频"""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_video_url(url: str) -> bool:
    """判断 URL 是否为已知视频平台链接"""
    for pattern in VIDEO_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def is_video_source(source: str) -> bool:
    """
    判断输入源是否为视频。

    - 本地文件: 根据扩展名判断
    - URL: 根据域名模式匹配判断
    """
    if source.startswith(("http://", "https://")):
        return is_video_url(source)
    else:
        return is_video_file(source)


async def extract_audio_from_video(
    source: str,
    output_dir: Path | None = None,
    filename_stem: str | None = None,
    progress_cb: callable | None = None,
) -> Path:
    """
    从视频提取音频，返回 WAV 文件路径。

    Args:
        source: 本地视频路径 或 视频 URL
        output_dir: 输出目录，默认使用临时目录
        filename_stem: 输出文件名 (不含后缀)，默认用视频标题
        progress_cb: 下载/提取进度回调 (percent: float 0-100)

    Returns:
        提取的 WAV 音频文件路径 (调用方负责清理)
    """
    if source.startswith(("http://", "https://")):
        return await _extract_from_url(source, output_dir, filename_stem, progress_cb)
    else:
        return _extract_from_local(Path(source), output_dir, filename_stem)


def _extract_from_local(
    video_path: Path,
    output_dir: Path | None = None,
    filename_stem: str | None = None,
) -> Path:
    """从本地视频文件提取音频"""
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = filename_stem or video_path.stem
    output_path = output_dir / f"{stem}_audio.wav"

    logger.info(f"从本地视频提取音频: {video_path.name}")
    logger.info(f"输出路径: {output_path}")

    # ffmpeg: 提取音频 -> 单声道 16kHz WAV
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                    # 不要视频
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz 采样率
        "-ac", "1",               # 单声道
        "-y",                     # 覆盖输出
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr[:500]}")

    logger.info(f"音频提取完成: {output_path.name}")
    return output_path


def _find_cookies_file(domain_hint: str = "") -> Path | None:
    """
    查找可用的 cookies 文件。

    搜索顺序:
    1. data/cookies.txt
    2. data/cookies/*{domain_hint}*.txt
    3. data/cookies/*.txt (第一个)
    """
    # 1. 默认路径
    if DEFAULT_COOKIES_FILE.exists():
        return DEFAULT_COOKIES_FILE

    # 2. 在 cookies 目录中搜索
    if COOKIES_DIR.exists():
        # 按域名匹配
        if domain_hint:
            matches = list(COOKIES_DIR.glob(f"*{domain_hint}*.txt"))
            if matches:
                return matches[0]
        # 任意 cookies 文件
        all_files = list(COOKIES_DIR.glob("*.txt"))
        if all_files:
            return all_files[0]

    return None


def _clean_url(url: str) -> str:
    """清洗 URL，去除 zsh 转义字符"""
    return url.replace("\\", "")


def _is_bilibili_url(url: str) -> bool:
    """判断是否为 Bilibili 链接"""
    return bool(re.search(r"bilibili\.com|b23\.tv|bili2233\.cn", url, re.IGNORECASE))


def _get_ydl_opts(url: str, base_opts: dict) -> dict:
    """
    构建 yt-dlp 配置，对 Bilibili 链接自动添加 cookies。

    优先级: cookies 文件 > 浏览器 cookies
    """
    opts = {**base_opts, "http_headers": {**DEFAULT_HTTP_HEADERS}}

    if _is_bilibili_url(url):
        # 1. 优先使用 cookies 文件
        cookies_path = _find_cookies_file("bilibili")
        if cookies_path:
            opts["cookiefile"] = str(cookies_path)
            logger.info(f"使用 cookies 文件: {cookies_path}")
            return opts

        # 2. 尝试从浏览器提取 cookies
        for browser in ["chrome", "safari", "firefox", "edge"]:
            try:
                opts["cookiesfrombrowser"] = (browser,)
                logger.debug(f"尝试从 {browser} 提取 cookies...")
                # 测试 cookies 是否有效
                import yt_dlp
                test_opts = {**opts, "quiet": True, "skip_download": True}
                with yt_dlp.YoutubeDL(test_opts) as ydl:
                    ydl.extract_info(url, download=False)
                logger.info(f"使用 {browser} 浏览器 cookies")
                return opts
            except Exception:
                continue

        # 所有方式都失败
        opts.pop("cookiesfrombrowser", None)
        logger.warning(
            f"未能获取 Bilibili cookies，请将浏览器 cookies 导出到:\n"
            f"  {DEFAULT_COOKIES_FILE}\n"
            f"可使用 Chrome 扩展 'Get cookies.txt LOCALLY' 导出"
        )

    return opts


async def _extract_from_url(
    url: str,
    output_dir: Path | None = None,
    filename_stem: str | None = None,
    progress_cb: callable | None = None,
) -> Path:
    """从网络视频 URL 下载并提取音频 (使用 yt-dlp Python 模块)"""
    import yt_dlp

    url = _clean_url(url)  # 清洗 URL

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp 输出模板: 优先使用 filename_stem
    if filename_stem:
        output_template = str(output_dir / f"{filename_stem}_audio.%(ext)s")
    else:
        output_template = str(output_dir / "%(title)s_audio.%(ext)s")

    logger.info(f"从网络视频提取音频: {url[:80]}...")

    # 下载进度回调
    def _progress_hook(d):
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                progress_cb(round(downloaded / total * 100, 1))
        elif progress_cb and d.get("status") == "finished":
            progress_cb(100.0)

    # yt-dlp 基础配置
    base_opts = {
        "extractaudio": True,           # 只提取音频
        "audioformat": "wav",           # 转为 WAV
        "audioquality": "0",            # 最佳质量
        "outtmpl": output_template,
        "noplaylist": True,             # 不下载播放列表
        "nocheckcertificate": True,
        "quiet": True,                  # 静默模式
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
    }

    # 构建配置 (Bilibili 自动添加浏览器 cookies)
    ydl_opts = _get_ydl_opts(url, base_opts)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"yt-dlp 下载失败: {e}")

    # 查找生成的文件
    wav_files = list(output_dir.glob("*_audio.wav"))
    if not wav_files:
        # 尝试其他扩展名
        wav_files = list(output_dir.glob("*_audio.*"))

    if not wav_files:
        raise RuntimeError(f"未找到提取的音频文件，输出目录: {output_dir}")

    output_path = wav_files[0]
    logger.info(f"音频下载完成: {output_path.name}")
    return output_path


async def get_video_info(source: str) -> dict | None:
    """
    获取视频元信息 (标题、描述、时长等)。

    使用 yt-dlp 的 extract_info 功能。
    仅支持 URL 输入，本地文件返回 None。
    """
    if source.startswith(("http://", "https://")):
        return await _get_url_video_info(source)
    return None


async def _get_url_video_info(url: str) -> dict | None:
    """获取网络视频的元信息 (使用 yt-dlp Python 模块)"""
    import yt_dlp

    url = _clean_url(url)  # 清洗 URL

    base_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    ydl_opts = _get_ydl_opts(url, base_opts)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", ""),
                "description": (info.get("description") or "")[:500],
                "duration": info.get("duration"),
                "uploader": info.get("uploader", ""),
                "upload_date": info.get("upload_date", ""),
                "view_count": info.get("view_count"),
                "webpage_url": info.get("webpage_url", url),
            }
    except Exception as e:
        logger.warning(f"获取视频信息失败: {e}")
        return None


async def extract_subtitles(source: str) -> str | None:
    """
    尝试提取视频字幕 (如果有内嵌或平台字幕)。

    Returns:
        字幕文本，无字幕返回 None
    """
    if not source.startswith(("http://", "https://")):
        return None

    import yt_dlp

    output_dir = Path(tempfile.mkdtemp())

    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["zh", "zh-Hans", "zh-CN", "en"],
        "subtitlesformat": "srt",
        "skip_download": True,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "http_headers": DEFAULT_HTTP_HEADERS,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source])

        # 查找字幕文件
        srt_files = list(output_dir.glob("*.srt"))
        if not srt_files:
            logger.info("未找到视频字幕，将使用 ASR 转写")
            return None

        # 读取字幕并转换为纯文本
        srt_content = srt_files[0].read_text(encoding="utf-8")
        text = _srt_to_text(srt_content)
        logger.info(f"提取字幕成功: {srt_files[0].name} ({len(text)} 字符)")
        return text

    except Exception as e:
        logger.warning(f"提取字幕失败: {e}")
        return None


def _srt_to_text(srt_content: str) -> str:
    """将 SRT 字幕格式转换为纯文本"""
    lines = []
    for line in srt_content.split("\n"):
        line = line.strip()
        # 跳过序号行和时间行
        if not line or line.isdigit() or "-->" in line:
            continue
        # 去除 HTML 标签
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    return "\n".join(lines)
