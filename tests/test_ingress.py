"""Exercise upload limits, authorization and media destination pinning."""

import base64
import hashlib
import io
import socket
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from yt_dlp.networking.exceptions import RequestError

import server
from utils.ingress import save_upload
from utils.media_network import PublicYoutubeDL, UnsafeMediaURL, _connect_public, validate_media_url


def address(host):
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (host, 443))


class IngressTests(unittest.IsolatedAsyncioTestCase):
    async def test_hash_and_cleanup(self):
        with TemporaryDirectory() as root:
            target = Path(root) / "input.wav"
            size, actual = await save_upload(UploadFile(io.BytesIO(b"voice")), target, 5)
            self.assertEqual((size, actual), (5, hashlib.sha256(b"voice").hexdigest()))
            target.unlink()
            from fastapi import HTTPException

            with self.assertRaises(HTTPException) as error:
                await save_upload(UploadFile(io.BytesIO(b"longer")), target, 5)
            self.assertEqual(error.exception.status_code, 413)
            self.assertFalse(target.exists())

    async def test_private_during_connect_cannot_rebind_initial_public_url(self):
        with patch("socket.getaddrinfo", side_effect=[[address("8.8.8.8")], [address("127.0.0.1")]]):
            validate_media_url("https://example.test/media")
            with patch("socket.socket") as sock:
                with self.assertRaises(UnsafeMediaURL):
                    _connect_public(("example.test", 443), 1)
                sock.assert_not_called()

    async def test_yt_dlp_never_opens_private_destination(self):
        with (
            PublicYoutubeDL({"quiet": True}) as downloader,
            patch("socket.getaddrinfo", return_value=[address("127.0.0.1")]),
            patch("socket.socket") as sock,
        ):
            with self.assertRaises(RequestError) as error:
                downloader.urlopen("http://example.test/media")
            self.assertIsInstance(error.exception.__cause__, UnsafeMediaURL)
            sock.assert_not_called()


class WorkspaceTests(unittest.TestCase):
    def test_host_spoof_and_cross_origin(self):
        with patch.object(server.settings, "workspace_token", ""):
            with TestClient(server.app, base_url="http://localhost", client=("127.0.0.1", 1234)) as client:
                self.assertEqual(client.get("/api/models").status_code, 200)
                self.assertEqual(client.get("/api/models", headers={"Host": "hostile.example"}).status_code, 401)
                response = client.post("/api/submit-url", headers={"Origin": "https://hostile.example"}, json={})
                self.assertEqual(response.status_code, 403)

    def test_token_required_for_task_download(self):
        value = base64.b64encode(b"meeting:secret-test").decode()
        with (
            patch.object(server.settings, "workspace_token", "secret-test"),
            TestClient(server.app) as client,
        ):
            self.assertEqual(client.get("/api/result/xyz/download").status_code, 401)
            self.assertEqual(client.get("/api/result/xyz/download", headers={"Authorization": f"Basic {value}"}).status_code, 404)
