from __future__ import annotations

import asyncio
import unittest

from revoltlogger import LogLevel, Logger

from subdominator.resources.providers.crtsh import CrtShResource


class _CrtShClient:
    """Stands in for RetryableHttpClient with a proxy configured."""

    def __init__(self, proxy: str | None) -> None:
        self.logger = Logger(name="test", level=LogLevel.NONE)
        self.proxy = proxy
        self.sql_calls = 0
        self.http_calls = 0

    async def get_json(self, url: str, *, expected_status: set[int]) -> list[dict[str, str]]:
        self.http_calls += 1
        return [{"name_value": "a.example.com"}]


class CrtShProxyBypassTests(unittest.TestCase):
    def test_postgres_path_is_skipped_when_a_proxy_is_configured(self) -> None:
        client = _CrtShClient("socks5://127.0.0.1:1080")
        resource = CrtShResource(client, None)  # type: ignore[arg-type]

        def fail_sql(_target: str) -> list[str]:
            raise AssertionError("postgres path must not run behind a proxy")

        resource._get_from_sql = fail_sql  # type: ignore[method-assign]
        result = asyncio.run(resource.enumerate("example.com", 0))

        self.assertEqual(result.findings, ["a.example.com"])
        self.assertEqual(client.http_calls, 1)


if __name__ == "__main__":
    unittest.main()
