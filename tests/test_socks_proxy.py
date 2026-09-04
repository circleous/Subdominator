from __future__ import annotations

import asyncio
import socket
import struct
import unittest
from time import perf_counter

import aiohttp
from aiohttp_socks import ProxyConnector, ProxyType
from revoltlogger import LogLevel, Logger

from subdominator.http.retryable import (
    RetryableHttpClient,
    build_socks_connector,
    is_socks_proxy,
)


def _logger() -> Logger:
    return Logger(name="test", level=LogLevel.NONE)


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def _serve_socks(*, stall: bool) -> tuple[asyncio.AbstractServer, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            _version, method_count = await reader.readexactly(2)
            await reader.readexactly(method_count)
            writer.write(b"\x05\x00")
            await writer.drain()

            _version, _command, _reserved, address_type = await reader.readexactly(4)
            if address_type == 1:
                host = socket.inet_ntoa(await reader.readexactly(4))
            elif address_type == 3:
                length = (await reader.readexactly(1))[0]
                host = (await reader.readexactly(length)).decode()
            else:
                host = socket.inet_ntop(socket.AF_INET6, await reader.readexactly(16))
            port = struct.unpack("!H", await reader.readexactly(2))[0]

            if stall:
                # A proxy that answers the greeting and then never answers
                # CONNECT, which is what a proxy does while it waits on an
                # upstream host that swallows TCP connections.
                await asyncio.Event().wait()

            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
            writer.write(
                b"\x05\x00\x00\x01" + socket.inet_aton("0.0.0.0") + struct.pack("!H", 0)
            )
            await writer.drain()
            await asyncio.gather(
                _relay(reader, upstream_writer),
                _relay(upstream_reader, writer),
            )
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.CancelledError):
            pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _serve_http(body: str) -> tuple[asyncio.AbstractServer, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read(65536)
            payload = body.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(payload)
                + payload
            )
            await writer.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


class SocksProxyDetectionTests(unittest.TestCase):
    def test_socks_schemes_are_detected(self) -> None:
        for proxy in (
            "socks5://127.0.0.1:1080",
            "SOCKS5H://127.0.0.1:1080",
            "socks4://127.0.0.1:1080",
            "socks4a://user:pass@127.0.0.1:1080",
        ):
            self.assertTrue(is_socks_proxy(proxy), proxy)

    def test_other_proxies_are_not_socks(self) -> None:
        for proxy in ("http://127.0.0.1:8080", "https://127.0.0.1:8080", "", None):
            self.assertFalse(is_socks_proxy(proxy), proxy)


class SocksSchemeMappingTests(unittest.TestCase):
    def test_scheme_maps_to_proxy_type_and_remote_dns(self) -> None:
        expected = {
            "socks4://127.0.0.1:1080": (ProxyType.SOCKS4, None),
            "socks4a://127.0.0.1:1080": (ProxyType.SOCKS4, True),
            "socks5://127.0.0.1:1080": (ProxyType.SOCKS5, None),
            "socks5h://127.0.0.1:1080": (ProxyType.SOCKS5, True),
        }

        async def run_check() -> dict[str, tuple[ProxyType, bool | None]]:
            built = {}
            for proxy in expected:
                connector = build_socks_connector(proxy)
                built[proxy] = (connector._proxy_type, connector._rdns)
                await connector.close()
            return built

        self.assertEqual(asyncio.run(run_check()), expected)

    def test_credentials_and_default_port_are_carried_over(self) -> None:
        async def run_check() -> tuple[str, int, str | None, str | None]:
            connector = build_socks_connector("socks5://user:secret@proxy.internal")
            try:
                return (
                    connector._proxy_host,
                    connector._proxy_port,
                    connector._proxy_username,
                    connector._proxy_password,
                )
            finally:
                await connector.close()

        self.assertEqual(asyncio.run(run_check()), ("proxy.internal", 1080, "user", "secret"))


class ConnectorSelectionTests(unittest.TestCase):
    def test_socks_proxy_uses_a_proxy_connector(self) -> None:
        async def run_check() -> tuple[type, bool]:
            async with RetryableHttpClient(
                logger=_logger(), proxy="socks5://127.0.0.1:1080"
            ) as client:
                return type(client._session.connector), client._session.trust_env

        connector_type, trust_env = asyncio.run(run_check())
        self.assertIs(connector_type, ProxyConnector)
        self.assertFalse(trust_env)

    def test_http_proxy_uses_a_tcp_connector(self) -> None:
        async def run_check() -> tuple[type, bool]:
            async with RetryableHttpClient(
                logger=_logger(), proxy="http://127.0.0.1:8080"
            ) as client:
                return type(client._session.connector), client._session.trust_env

        connector_type, trust_env = asyncio.run(run_check())
        self.assertIs(connector_type, aiohttp.TCPConnector)
        self.assertTrue(trust_env)


class SocksTimeoutTests(unittest.TestCase):
    def test_stalled_handshake_is_bounded_by_the_request_timeout(self) -> None:
        async def run_check() -> float:
            server, port = await _serve_socks(stall=True)
            try:
                async with RetryableHttpClient(
                    logger=_logger(),
                    timeout=1.0,
                    retries=1,
                    proxy=f"socks5://127.0.0.1:{port}",
                ) as client:
                    started = perf_counter()
                    with self.assertRaises(RuntimeError):
                        await client.request("GET", "http://stalled.invalid/")
                    return perf_counter() - started
            finally:
                server.close()

        self.assertLess(asyncio.run(run_check()), 6.0)


class SocksRequestTests(unittest.TestCase):
    def test_request_succeeds_through_a_socks_proxy(self) -> None:
        async def run_check() -> str:
            upstream, upstream_port = await _serve_http("a.example.com\nb.example.com\n")
            proxy_server, proxy_port = await _serve_socks(stall=False)
            try:
                async with RetryableHttpClient(
                    logger=_logger(),
                    timeout=5.0,
                    retries=1,
                    proxy=f"socks5://127.0.0.1:{proxy_port}",
                ) as client:
                    return await client.request("GET", f"http://127.0.0.1:{upstream_port}/")
            finally:
                proxy_server.close()
                upstream.close()

        self.assertEqual(asyncio.run(run_check()), "a.example.com\nb.example.com\n")

    def test_request_succeeds_with_the_socks5h_spelling(self) -> None:
        async def run_check() -> str:
            upstream, upstream_port = await _serve_http("c.example.com\n")
            proxy_server, proxy_port = await _serve_socks(stall=False)
            try:
                async with RetryableHttpClient(
                    logger=_logger(),
                    timeout=5.0,
                    retries=1,
                    proxy=f"socks5h://127.0.0.1:{proxy_port}",
                ) as client:
                    return await client.request("GET", f"http://127.0.0.1:{upstream_port}/")
            finally:
                proxy_server.close()
                upstream.close()

        self.assertEqual(asyncio.run(run_check()), "c.example.com\n")


if __name__ == "__main__":
    unittest.main()
