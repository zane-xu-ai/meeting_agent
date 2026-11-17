"""Bounds and identities for the single-workspace web deployment."""

import base64
import hashlib
import hmac
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import anyio
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers


def local_host(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def authorized(request: Request, token: str) -> bool:
    host = urlsplit("//" + request.headers.get("host", "")).hostname
    if not token:
        return local_host(host) and local_host(request.client.host if request.client else None)
    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except ValueError:
        return False
    return hmac.compare_digest(decoded, ("meeting:" + token).encode())


class RequestLimit:
    """Limit multipart bodies before Starlette spools them to disk."""

    def __init__(self, app, limit):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        limit = self.limit()
        length = Headers(scope=scope).get("content-length")
        if length and length.isdecimal() and int(length) > limit:
            await JSONResponse({"detail": "上传超过大小限制"}, 413)(scope, receive, send)
            return
        seen = 0

        async def bounded():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > limit:
                raise HTTPException(413, "上传超过大小限制")
            return message

        await self.app(scope, bounded, send)


async def save_upload(file, path: Path, limit: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        async with await anyio.open_file(path, "xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "上传超过大小限制")
                digest.update(chunk)
                await output.write(chunk)
        if not size:
            raise HTTPException(400, "上传文件为空")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return size, digest.hexdigest()
