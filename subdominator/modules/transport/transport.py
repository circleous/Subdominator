from __future__ import annotations

import httpcore
import httpx

from subdominator.modules.logger.logger import logger

SOCKS_SCHEMES = ("socks5://", "socks5h://")


class HandshakeBoundStream(httpcore.AsyncNetworkStream):
    """Delegates to a real stream and supplies a timeout where httpcore omits one.

    httpcore builds the argument dict for its SOCKS5 handshake without a timeout
    key (encode/httpcore#1015, unfixed in 1.0.9, PR #1097 unmerged), so
    `_init_socks5_connection` reads the CONNECT reply through
    `anyio.fail_after(None)`, which never expires. A proxied request to a host
    that accepts no TCP connection then blocks until the proxy abandons its own
    upstream attempt, so the wall time comes from the proxy and no client
    setting caps it.

    Reads and writes outside the handshake always carry an explicit per-request
    timeout from `request.extensions`, so `fallback` applies to the handshake
    only.
    """

    def __init__(self, stream: httpcore.AsyncNetworkStream, fallback: float) -> None:
        self._stream = stream
        self._fallback = fallback

    def _bounded(self, timeout: float | None) -> float | None:
        return self._fallback if timeout is None else timeout

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return await self._stream.read(max_bytes, self._bounded(timeout))

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self._stream.write(buffer, self._bounded(timeout))

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._stream.start_tls(
            ssl_context, server_hostname, self._bounded(timeout)
        )
        return HandshakeBoundStream(stream, self._fallback)

    def get_extra_info(self, info: str):
        return self._stream.get_extra_info(info)


class HandshakeBoundBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, backend: httpcore.AsyncNetworkBackend, fallback: float) -> None:
        self._backend = backend
        self._fallback = fallback

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._backend.connect_tcp(
            host, port, timeout, local_address, socket_options
        )
        return HandshakeBoundStream(stream, self._fallback)

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options=None
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._backend.connect_unix_socket(
            path, timeout, socket_options
        )
        return HandshakeBoundStream(stream, self._fallback)

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def is_socks(proxy: str | None) -> bool:
    return bool(proxy) and proxy.lower().startswith(SOCKS_SCHEMES)


def bind_socks_handshake(client: httpx.AsyncClient, fallback: float) -> bool:
    """Wrap the network backend of every SOCKS pool the client can route to.

    httpx constructs the `httpcore.AsyncSOCKSProxy` pool itself and offers no
    argument for its network backend, so the wrap happens through private
    attributes after construction. A pool passes its backend to each connection
    it creates, so this must run before the first request, and it also covers
    proxies that httpx resolved from the environment. Every attribute access is
    guarded: an httpx internal rename costs the bound and leaves the run
    working.
    """
    mounts = getattr(client, "_mounts", None) or {}
    transports = [getattr(client, "_transport", None), *mounts.values()]
    bound = False
    for transport in transports:
        pool = getattr(transport, "_pool", None)
        if isinstance(pool, httpcore.AsyncSOCKSProxy):
            backend = getattr(pool, "_network_backend", None)
            if backend is None:
                continue
            pool._network_backend = HandshakeBoundBackend(backend, fallback)
            bound = True
    return bound


def http_client(args, **kwargs) -> httpx.AsyncClient:
    """Build the client every source shares, with -t enforced on SOCKS5 proxies."""
    kwargs.setdefault("verify", False)
    kwargs.setdefault("proxy", args.proxy)
    kwargs.setdefault("timeout", httpx.Timeout(args.timeout, connect=args.timeout))
    client = httpx.AsyncClient(**kwargs)
    if not bind_socks_handshake(client, args.timeout) and is_socks(args.proxy):
        if args.verbose:
            logger(
                "Unable to bound the SOCKS5 handshake, a request through the proxy "
                "may outlast the timeout set by -t",
                "warn",
                args.no_color,
            )
    return client
