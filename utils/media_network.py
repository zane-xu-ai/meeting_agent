"""Public HTTP transport for untrusted media URLs."""

import http.client
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from yt_dlp import YoutubeDL
from yt_dlp.networking._helper import create_connection
from yt_dlp.networking._urllib import HTTPHandler, RedirectHandler, UrllibRH


class UnsafeMediaURL(ValueError):
    """A media destination did not pass the public network policy."""


def parse_media_url(url: str):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise UnsafeMediaURL("无效的音视频链接") from error
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or port not in {None, 80, 443} or "\\" in url
        or any(ord(character) < 32 for character in url)
    ):
        raise UnsafeMediaURL("音视频链接须为公开 HTTP(S) 地址")
    return parsed


def _require_public(host: str) -> None:
    ip = ipaddress.ip_address(host.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast or ip.is_reserved or getattr(ip, "is_site_local", False):
        raise UnsafeMediaURL("音视频链接指向非公网地址")


def validate_media_url(url: str):
    parsed = parse_media_url(url)
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    for address in addresses:
        _require_public(address[4][0])
    return parsed


def _connect_public(address, timeout, source_address=None):
    """Resolve once and connect to the validated numeric sockaddr, retaining TLS SNI."""
    def connect(resolved, timeout, source_address):
        family, kind, protocol, _, sockaddr = resolved
        _require_public(sockaddr[0])
        with_socket = socket.socket(family, kind, protocol)
        try:
            with_socket.settimeout(timeout)
            if source_address:
                with_socket.bind(source_address)
            with_socket.connect(sockaddr)
            return with_socket
        except BaseException:
            with_socket.close()
            raise

    return create_connection(address, timeout, source_address, _create_socket_func=connect)


class PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_public


class PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_public


class _PublicHTTPHandler(HTTPHandler):
    def http_open(self, request):
        parse_media_url(request.full_url)
        return self.do_open(PublicHTTPConnection, request)

    def https_open(self, request):
        parse_media_url(request.full_url)
        return self.do_open(PublicHTTPSConnection, request, context=self._context)


class _PublicRedirectHandler(RedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parse_media_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicUrllibRH(UrllibRH):
    _SUPPORTED_URL_SCHEMES = ("http", "https")
    _SUPPORTED_PROXY_SCHEMES = ()

    RH_KEY = "Urllib"

    def _create_instance(self, proxies, cookiejar, legacy_ssl_support=None):
        opener = super()._create_instance({}, cookiejar, legacy_ssl_support)
        # Install guarded handlers before the stock handlers of equal priority.
        handler = _PublicHTTPHandler(context=self._make_sslcontext(legacy_ssl_support=legacy_ssl_support))
        handler.handler_order = HTTPHandler.handler_order - 1
        redirect = _PublicRedirectHandler()
        redirect.handler_order = RedirectHandler.handler_order - 1
        opener.add_handler(handler)
        opener.add_handler(redirect)
        return opener


class PublicYoutubeDL(YoutubeDL):
    def __init__(self, options):
        super().__init__({
            **options, "proxy": "", "nocheckcertificate": False, "enable_file_urls": False,
            "external_downloader": {"default": "native"}, "socket_timeout": 20,
            "hls_prefer_native": True,
        })

    def build_request_director(self, handlers, preferences=None):
        return super().build_request_director([PublicUrllibRH])

    def dl(self, name, info, subtitle=False, test=False):
        protocol = info.get("protocol") or urlsplit(info.get("url", "")).scheme
        if protocol not in {"http", "https", "m3u8_native", "http_dash_segments"}:
            raise UnsafeMediaURL("不支持该媒体传输协议")
        return super().dl(name, info, subtitle, test)


def download_public_audio(url: str, target: Path, max_bytes: int) -> None:
    for _ in range(6):
        parsed = parse_media_url(url)
        connection_type = PublicHTTPSConnection if parsed.scheme == "https" else PublicHTTPConnection
        connection = connection_type(parsed.hostname, parsed.port, timeout=30)
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.request("GET", path, headers={"Accept-Encoding": "identity"})
            with connection.getresponse() as response:
                if response.status in {301, 302, 303, 307, 308} and response.getheader("Location"):
                    url = urljoin(url, response.getheader("Location"))
                    continue
                if response.status != 200:
                    raise ValueError(f"媒体下载失败（HTTP {response.status}）")
                size = 0
                with target.open("xb") as output:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError("远程音频超过大小限制")
                        output.write(block)
                if not size:
                    raise ValueError("远程音频为空")
                return
        finally:
            connection.close()
    raise UnsafeMediaURL("音视频链接重定向次数过多")
