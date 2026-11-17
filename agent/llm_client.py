"""会议助手 - LLM 调用封装 (OpenAI 兼容接口)"""

from __future__ import annotations

import json
import re
from typing import cast
from urllib.parse import urlsplit

import httpx
from loguru import logger

from config import settings


class LLMJSONError(ValueError):
    """The model did not complete a usable JSON response."""

    def __init__(self, finish_reason: str | None = None):
        self.truncated = finish_reason == "length"
        message = "LLM 的 JSON 输出达到长度上限，未能生成完整结果" if self.truncated else "LLM 输出无法解析为 JSON"
        super().__init__(message)


class LLMClient:
    """LLM 客户端，使用 OpenAI 兼容接口"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        models: list[str] | None = None,
    ):
        self.api_key = api_key or settings.llm_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.models = models or settings.llm_models
        if not self.models:
            raise ValueError("LLM 模型列表为空")
        self.model = self.models[0]  # 默认使用第一个

        if not self.api_key:
            raise ValueError(
                "请配置 LLM API Key: 设置 DASHSCOPE_API_KEY(ASR+LLM 共用) "
                "或单独设置 LLM_API_KEY，可在 .env 文件或环境变量中设置"
            )

        self.chat_url = f"{self.base_url}/chat/completions"
        logger.info(
            f"LLM 客户端初始化: models={self.models}, base_url={self.base_url}")

    async def chat(
        self,
        prompt: str,
        system: str = "你是一名专业的会议纪要助手，擅长从会议记录中提取关键信息。",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """发送单轮对话，按配置尝试模型并返回文本。"""
        result = await self._chat(prompt, system, temperature, max_tokens)
        return cast(str, result)

    async def _chat(
        self, prompt: str, system: str, temperature: float, max_tokens: int,
        *, json_output: bool = False,
    ) -> str | dict:
        last_error = None
        for model in self.models:
            self.model = model
            budget = max_tokens
            timeout_retried = False
            use_json_mode = json_output
            while True:
                try:
                    choice = await self._chat_with_model(
                        prompt, system, temperature, budget,
                        use_json_mode=use_json_mode, extend_timeout=timeout_retried,
                    )
                    content = choice["message"].get("content")
                    if not json_output:
                        return content
                    if choice.get("finish_reason") == "length":
                        raise LLMJSONError("length")
                    return self._extract_json(content)
                except LLMJSONError as error:
                    last_error = error
                    if error.truncated and budget < 16384:
                        budget = min(budget * 2, 16384)
                        logger.warning(f"LLM {model} 的 JSON 输出被截断，以 max_tokens={budget} 重试")
                        continue
                    break
                except httpx.HTTPStatusError as error:
                    last_error = error
                    # 部分兼容接口仅支持通过提示词约束 JSON。
                    if (use_json_mode and error.response.status_code in (400, 422)
                            and "response_format" in error.response.text.lower()):
                        use_json_mode = False
                        logger.info(f"LLM {model} 不接受 response_format，改用提示词约束 JSON")
                        continue
                    break
                except httpx.TimeoutException as error:
                    last_error = error
                    if not timeout_retried:
                        timeout_retried = True
                        logger.warning(f"LLM 模型 {model} 超时，延长超时后重试")
                        continue
                    break
                except Exception as error:
                    last_error = error
                    break
            logger.warning(f"LLM 模型 {model} 失败: {last_error}")
        if isinstance(last_error, LLMJSONError):
            raise last_error
        raise RuntimeError(
            f"所有 LLM 模型均失败。最后一个错误 ({self.model}): {last_error}"
        ) from last_error

    def _reasoning_options(self) -> dict:
        host = urlsplit(self.base_url).hostname or ""
        if self.model == "glm-5.3" and (
            host == "dashscope.aliyuncs.com" or host.endswith(".maas.aliyuncs.com")
        ):
            return {"reasoning_effort": "low"}
        return {}

    async def _chat_with_model(
        self, prompt: str, system: str, temperature: float, max_tokens: int,
        *, use_json_mode: bool, extend_timeout: bool,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._reasoning_options(),
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = max(60.0 + (max_tokens / 4096) * 60.0, 120.0)
        if extend_timeout:
            timeout *= 2

        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"调用 LLM: {self.model}")
            resp = await client.post(self.chat_url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        choice = result["choices"][0]
        usage = result.get("usage", {})
        logger.info(
            f"LLM 响应: model={self.model}, finish_reason={choice.get('finish_reason')}, "
            f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
            f"completion_tokens={usage.get('completion_tokens', '?')}"
        )
        return choice

    async def chat_messages(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        发送多轮对话请求 (完整 messages 列表)，支持模型列表自动 fallback。

        用于多轮对话场景，messages 包含完整的对话历史 (system/user/assistant)。

        Args:
            messages: 完整对话消息列表 [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大输出 token 数

        Returns:
            模型输出文本
        """
        last_error = None
        for model in self.models:
            self.model = model
            timeout_override: float | None = None
            for attempt in range(2):
                try:
                    return await self._chat_messages_with_model(
                        messages, temperature, max_tokens,
                        timeout_override=timeout_override)
                except httpx.TimeoutException:
                    if attempt == 0:
                        timeout_override = 60.0 + (max_tokens / 4096) * 60.0
                        timeout_override = max(timeout_override, 120.0) * 2
                        logger.warning(
                            f"LLM 模型 {model} (messages) 超时，"
                            f"以 timeout={timeout_override:.0f}s 重试...")
                        continue
                    last_error = httpx.TimeoutException(f"请求超时 (已重试)")
                except Exception as e:
                    last_error = e
                    break
            logger.warning(f"LLM 模型 {model} (messages) 失败: {last_error}，尝试下一个...")
            if model != self.models[-1]:
                continue
            raise RuntimeError(
                f"所有 LLM 模型 (messages) 均失败。最后一个错误 ({model}): {last_error}"
            ) from last_error

    async def _chat_messages_with_model(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_override: float | None = None,
    ) -> str:
        """使用当前 self.model 发送多轮对话"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if timeout_override is not None:
            timeout = timeout_override
        else:
            # 根据 max_tokens 动态计算超时 (基础60s + 每4096 tokens额外60s)
            timeout = 60.0 + (max_tokens / 4096) * 60.0
            timeout = max(timeout, 120.0)  # 最少 120s

        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(
                f"调用 LLM (messages): {self.model}, turns={len(messages)}")
            resp = await client.post(self.chat_url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        logger.info(
            f"LLM 响应 (messages): model={self.model}, "
            f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
            f"completion_tokens={usage.get('completion_tokens', '?')}"
        )
        return content

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> dict:
        """
        发送对话请求并支持 Function Calling (Tool Use)。

        支持模型列表自动 fallback: 若当前模型不支持 tools，尝试下一个。

        Args:
            messages: 完整对话消息列表 (含 system/user/tool roles)
            tools: OpenAI tools 格式的工具定义
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            tool_choice: 工具选择策略 ("auto" | "none" | "required")

        Returns:
            模型响应 message dict:
            - 有 tool_calls 时: {"role": "assistant", "tool_calls": [...]}
            - 无 tool_calls 时: {"role": "assistant", "content": "..."}
        """
        last_error = None
        for model in self.models:
            self.model = model
            timeout_override: float | None = None
            for attempt in range(2):
                try:
                    return await self._chat_with_model_tools(
                        messages, tools, temperature, max_tokens, tool_choice,
                        timeout_override=timeout_override
                    )
                except httpx.TimeoutException:
                    if attempt == 0:
                        timeout_override = 60.0 + (max_tokens / 4096) * 60.0
                        timeout_override = max(timeout_override, 120.0) * 2
                        logger.warning(
                            f"LLM 模型 {model} (tools) 超时，"
                            f"以 timeout={timeout_override:.0f}s 重试...")
                        continue
                    last_error = httpx.TimeoutException(f"请求超时 (已重试)")
                except Exception as e:
                    last_error = e
                    break
            logger.warning(f"LLM 模型 {model} (tools) 失败: {last_error}，尝试下一个...")
            if model != self.models[-1]:
                continue
            raise RuntimeError(
                f"所有 LLM 模型 (tools) 均失败。最后一个错误 ({model}): {last_error}"
            ) from last_error

    async def _chat_with_model_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
        tool_choice: str,
        timeout_override: float | None = None,
    ) -> dict:
        """使用当前 self.model 执行带 tools 的对话"""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            **self._reasoning_options(),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if timeout_override is not None:
            timeout = timeout_override
        else:
            # 根据 max_tokens 动态计算超时 (与 _chat_with_model 保持一致)
            timeout = 60.0 + (max_tokens / 4096) * 60.0
            timeout = max(timeout, 120.0)  # 最少 120s

        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"调用 LLM (tools): {self.model}, tools={len(tools)}")
            resp = await client.post(self.chat_url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        message = result["choices"][0]["message"]
        usage = result.get("usage", {})
        has_tool_calls = bool(message.get("tool_calls"))
        logger.info(
            f"LLM 响应 (tools): model={self.model}, "
            f"tool_calls={'yes' if has_tool_calls else 'no'}, "
            f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
            f"completion_tokens={usage.get('completion_tokens', '?')}"
        )
        return message

    async def chat_stream(
        self,
        prompt: str,
        system: str = "你是一名专业的会议纪要助手，擅长从会议记录中提取关键信息。",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        流式对话输出，逐 token 返回。

        Args:
            prompt: 用户 prompt
            system: 系统 prompt
            temperature: 温度参数
            max_tokens: 最大输出 token 数

        Yields:
            str: 每个 token 片段
        """
        last_error = None
        for model in self.models:
            self.model = model
            try:
                async for token in self._chat_stream_with_model(prompt, system, temperature, max_tokens):
                    yield token
                return  # 成功完成，退出
            except Exception as e:
                last_error = e
                logger.warning(f"LLM 模型 {model} (stream) 失败: {e}，尝试下一个...")
                if model != self.models[-1]:
                    continue
                raise RuntimeError(
                    f"所有 LLM 模型 (stream) 均失败。最后一个错误 ({model}): {last_error}"
                ) from last_error

    async def _chat_stream_with_model(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ):
        """使用当前 self.model 执行流式对话"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,  # 启用流式输出
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 根据 max_tokens 动态计算超时 (基础60s + 每4096 tokens额外60s)
        timeout = 60.0 + (max_tokens / 4096) * 60.0
        timeout = max(timeout, 120.0)  # 最少 120s

        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"调用 LLM (stream): {self.model}")
            async with client.stream("POST", self.chat_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                # 逐行读取 SSE 响应
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[
                                0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue

    async def chat_messages_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        流式多轮对话输出。

        Args:
            messages: 完整对话消息列表
            temperature: 温度参数
            max_tokens: 最大输出 token 数

        Yields:
            str: 每个 token 片段
        """
        last_error = None
        for model in self.models:
            self.model = model
            try:
                async for token in self._chat_messages_stream_with_model(messages, temperature, max_tokens):
                    yield token
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM 模型 {model} (messages stream) 失败: {e}，尝试下一个...")
                if model != self.models[-1]:
                    continue
                raise RuntimeError(
                    f"所有 LLM 模型 (messages stream) 均失败。最后一个错误 ({model}): {last_error}"
                ) from last_error

    async def _chat_messages_stream_with_model(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        """使用当前 self.model 执行流式多轮对话"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 根据 max_tokens 动态计算超时
        timeout = 60.0 + (max_tokens / 4096) * 60.0
        timeout = max(timeout, 120.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(
                f"调用 LLM (messages stream): {self.model}, turns={len(messages)}")
            async with client.stream("POST", self.chat_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[
                                0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue

    async def chat_json(
        self,
        prompt: str,
        system: str = "你是一名专业的会议纪要助手，擅长从会议记录中提取关键信息。请始终返回有效的 JSON。",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """请求 JSON 对象。按模型回退，只在明确截断时增加输出额度。"""
        result = await self._chat(prompt, system, temperature, max_tokens, json_output=True)
        return cast(dict, result)

    @staticmethod
    def _extract_json(text: str | None) -> dict:
        """解析 JSON 对象，兼容 Markdown 代码块和外围说明文字。"""
        if not isinstance(text, str) or not text.strip():
            raise LLMJSONError()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                blocks.append(text[start:end + 1])
            for block in blocks:
                try:
                    data = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    return data
        else:
            if isinstance(data, dict):
                return data
        raise LLMJSONError()
