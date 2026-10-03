"""会议助手 - ASR 语音识别模块 (阿里云百炼 Paraformer)"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from loguru import logger

from config import settings
from models.schemas import Sentence, Transcript


class ASRClient:
    """阿里云百炼 Paraformer 离线语音识别客户端

    支持两种音频输入方式:
    1. 本地文件路径 -> 自动上传到 DashScope 临时 OSS 获取 URL
    2. 直接传入可访问的 HTTP/HTTPS URL
    """

    TRANSCRIPTION_URL = (
        "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    )
    UPLOAD_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.dashscope_api_key
        if not self.api_key:
            raise ValueError(
                "请配置 DASHSCOPE_API_KEY，可在 .env 文件或环境变量中设置"
            )
        self.model = settings.asr_model

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    async def transcribe(self, audio_source: str | Path) -> Transcript:
        """
        转写音频。

        Args:
            audio_source: 本地音频文件路径 或 可访问的音频 URL

        Returns:
            Transcript 对象，包含带时间戳和说话人标签的句子列表
        """
        audio_str = str(audio_source)

        # 判断是 URL 还是本地文件
        if audio_str.startswith(("http://", "https://")):
            audio_url = audio_str
            logger.info(f"使用远程音频 URL 转写")
        else:
            audio_path = Path(audio_source)
            if not audio_path.exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            logger.info(f"上传本地文件获取临时 URL: {audio_path.name}")
            audio_url = await self._upload_file(audio_path)

        logger.info(f"开始 ASR 转写, 模型: {self.model}")

        # 提交转写任务
        payload = {
            "model": self.model,
            "input": {
                "file_urls": [audio_url],
            },
            "parameters": {
                "diarization_enabled": True,
                "timestamp_enabled": True,
                "language_hints": ["zh", "en"],
            },
        }

        headers = {**self._headers(), "Content-Type": "application/json"}
        # DashScope 异步任务需要设置 X-DashScope-Async: enable
        headers["X-DashScope-Async"] = "enable"
        # 如果音频 URL 是 oss:// 前缀 (DashScope 临时存储), 需要此头
        if audio_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"

        async with httpx.AsyncClient(timeout=300.0) as client:
            logger.info("提交转写任务到 DashScope...")
            resp = await client.post(
                self.TRANSCRIPTION_URL, json=payload, headers=headers
            )
            resp.raise_for_status()
            result = resp.json()

            # 获取 task_id 并轮询结果
            task_id = result.get("output", {}).get("task_id")
            if not task_id:
                # 可能是同步返回了结果
                return self._parse_response(result)

            logger.info(f"异步任务已提交，task_id: {task_id}")
            result = await self._poll_task(task_id, client)

        transcript = self._parse_response(result)
        logger.info(
            f"转写完成: 共 {len(transcript.sentences)} 句, "
            f"时长 {transcript.duration_ms / 1000:.1f}s"
        )
        return transcript

    async def _upload_file(self, audio_path: Path) -> str:
        """
        上传本地文件到 DashScope 临时 OSS 存储，返回 oss:// URL。

        流程:
        1. GET 获取上传策略 (getPolicy)
        2. 上传文件到 OSS
        3. 返回 oss:// 临时 URL (48h 有效)

        注意: 使用 oss:// URL 调用 ASR 时需添加
              X-DashScope-OssResourceResolve: enable 请求头
        """
        headers = self._headers()

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Step 1: GET 获取上传策略
            logger.debug("获取 DashScope 上传策略...")
            resp = await client.get(
                self.UPLOAD_URL,
                params={"action": "getPolicy", "model": self.model},
                headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()
            # DashScope 返回结构可能是 {"data": {...}} 或直接 {...}
            policy_data = raw.get("data") or raw

            upload_host = policy_data.get(
                "host", "") or policy_data.get("upload_host", "")
            upload_dir = policy_data.get(
                "dir", "") or policy_data.get("upload_dir", "")
            policy = policy_data.get("policy", "")
            access_key_id = policy_data.get("oss_access_key_id", "")
            signature = policy_data.get("signature", "")
            x_oss_object_acl = policy_data.get("x_oss_object_acl", "")
            x_oss_forbid_overwrite = policy_data.get(
                "x_oss_forbid_overwrite", "")

            if not upload_host:
                raise RuntimeError(f"获取上传策略失败: {policy_data}")

            # Step 2: 上传文件到 OSS
            file_name = audio_path.name
            oss_key = f"{upload_dir}/{file_name}"

            logger.debug(f"上传文件到 OSS: {oss_key}")
            with open(audio_path, "rb") as f:
                form_data = {
                    "key": (None, oss_key),
                    "policy": (None, policy),
                    "OSSAccessKeyId": (None, access_key_id),
                    "signature": (None, signature),
                    "x-oss-object-acl": (None, x_oss_object_acl),
                    "x-oss-forbid-overwrite": (None, x_oss_forbid_overwrite),
                    "file": (file_name, f, "audio/wav"),
                    "success_action_status": (None, "200"),
                }
                resp = await client.post(upload_host, files=form_data)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"上传文件到 OSS 失败: {resp.status_code} {resp.text}"
                    )

            # Step 3: 返回 oss:// 临时 URL (无需 getFileUrl 步骤)
            oss_url = f"oss://{oss_key}"
            logger.info(f"文件上传成功，获取临时 OSS URL")
            return oss_url

    async def _poll_task(
        self,
        task_id: str,
        client: httpx.AsyncClient,
        interval: int = 3,
        max_retries: int = 200,
    ) -> dict:
        """轮询异步转写任务状态"""
        import asyncio

        # DashScope 统一任务查询接口
        task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = self._headers()

        for i in range(max_retries):
            await asyncio.sleep(interval)
            resp = await client.get(task_url, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            status = result.get("output", {}).get("task_status", "")
            if status == "SUCCEEDED":
                logger.info("转写任务完成")
                return result
            elif status == "FAILED":
                error_msg = result.get("output", {}).get("message", "未知错误")
                raise RuntimeError(f"转写任务失败: {error_msg}")
            else:
                if i % 10 == 0:
                    logger.info(f"转写进行中... 状态: {status}")

        raise TimeoutError(f"转写任务超时: task_id={task_id}")

    def _parse_response(self, result: dict) -> Transcript:
        """解析 DashScope 异步任务返回结果，提取 transcription_url 并下载解析"""
        sentences = []
        output = result.get("output", {})
        duration_ms = 0

        # 从 results 中找到 transcription_url
        results_list = output.get("results", [])
        transcription_url = None

        for item in results_list:
            # 每个 item 可能包含 transcription_url
            if isinstance(item, dict):
                t_url = item.get("transcription_url")
                if t_url:
                    transcription_url = t_url
                    break
                # 也可能嵌套在 output.results 中
                sub_results = item.get("results", [])
                for sub in sub_results:
                    if isinstance(sub, dict):
                        t_url = sub.get("transcription_url")
                        if t_url:
                            transcription_url = t_url
                            break

        if transcription_url:
            sub_result = self._parse_from_url(transcription_url)
            sentences = sub_result.sentences
            duration_ms = sub_result.duration_ms

        return Transcript(sentences=sentences, duration_ms=int(duration_ms))

    def _parse_from_url(self, url: str) -> Transcript:
        """从 DashScope 返回的 transcription_url 下载并解析结果

        JSON 格式:
        {
          "properties": {"original_duration_in_milliseconds": 5546, ...},
          "transcripts": [{
            "sentences": [{
              "begin_time": 260,
              "end_time": 5540,
              "text": "...",
              "speaker_id": 0,
              ...
            }]
          }]
        }
        """
        logger.debug(f"下载转写结果 JSON...")
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        sentences = []
        duration_ms = 0

        # 获取音频时长
        props = data.get("properties", {})
        duration_ms = props.get("original_duration_in_milliseconds", 0)

        # 解析 transcripts -> sentences
        for t in data.get("transcripts", []):
            for s in t.get("sentences", []):
                text = s.get("text", "")
                if not text:
                    continue

                speaker = s.get("speaker_id", None)
                begin_time = s.get("begin_time", 0)
                end_time = s.get("end_time", 0)

                sentences.append(
                    Sentence(
                        text=text.strip(),
                        speaker=f"Speaker-{speaker:02d}" if speaker is not None else None,
                        begin_time=int(begin_time),
                        end_time=int(end_time),
                    )
                )

        return Transcript(sentences=sentences, duration_ms=int(duration_ms))
