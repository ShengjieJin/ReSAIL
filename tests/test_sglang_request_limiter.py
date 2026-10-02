from __future__ import annotations

import asyncio

import pytest

from slime.utils import http_utils


NUM_GPUS = 0


@pytest.mark.unit
def test_generate_request_limiter_bounds_actual_http_calls(monkeypatch):
    active = 0
    observed = 0

    async def fake_transport(url, payload, max_retries=60, headers=None):
        nonlocal active, observed
        active += 1
        observed = max(observed, active)
        await asyncio.sleep(0.002)
        active -= 1
        return {"url": url, "payload": payload, "headers": headers}

    monkeypatch.setattr(http_utils, "_post_with_transport", fake_transport)
    http_utils.configure_generate_request_limiter(4)

    async def run():
        return await asyncio.gather(
            *(http_utils.post("http://router/generate", {"identity": index}) for index in range(29))
        )

    try:
        outputs = asyncio.run(run())
        metrics = http_utils.get_generate_request_metrics()
        assert [row["payload"]["identity"] for row in outputs] == list(range(29))
        assert observed == 4
        assert metrics["rollout/sglang_request/configured_concurrency"] == 4
        assert metrics["rollout/sglang_request/submitted"] == 29
        assert metrics["rollout/sglang_request/completed"] == 29
        assert metrics["rollout/sglang_request/observed_max_inflight"] == 4
        assert metrics["rollout/sglang_request/errors"] == 0
        assert metrics["rollout/sglang_request/service_p99_seconds"] > 0
        assert metrics["rollout/sglang_request/queue_wait_p99_seconds"] > 0
    finally:
        http_utils.configure_generate_request_limiter(512)


@pytest.mark.unit
def test_generate_request_limiter_releases_permit_and_ignores_non_generate(monkeypatch):
    attempts = 0

    async def fake_transport(url, payload, max_retries=60, headers=None):
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0)
        if payload.get("fail"):
            raise RuntimeError("injected request failure")
        return payload

    monkeypatch.setattr(http_utils, "_post_with_transport", fake_transport)
    http_utils.configure_generate_request_limiter(1)

    async def run():
        with pytest.raises(RuntimeError, match="injected"):
            await http_utils.post("http://router/generate", {"fail": True})
        assert await http_utils.post("http://router/generate", {"ok": True}) == {"ok": True}
        assert await http_utils.post("http://router/health", {"health": True}) == {"health": True}

    try:
        asyncio.run(run())
        metrics = http_utils.get_generate_request_metrics()
        assert attempts == 3
        assert metrics["rollout/sglang_request/submitted"] == 2
        assert metrics["rollout/sglang_request/completed"] == 1
        assert metrics["rollout/sglang_request/errors"] == 1
        assert metrics["rollout/sglang_request/observed_max_inflight"] == 1
        http_utils.reset_generate_request_metrics()
        assert http_utils.get_generate_request_metrics()["rollout/sglang_request/submitted"] == 0
    finally:
        http_utils.configure_generate_request_limiter(512)


@pytest.mark.unit
def test_generate_request_limiter_cancel_reset_fifo_and_headers(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    start_order = []

    async def fake_transport(url, payload, max_retries=60, headers=None):
        start_order.append(payload["identity"])
        if payload["identity"] == 0:
            entered.set()
            await release.wait()
        return {"identity": payload["identity"], "headers": headers}

    monkeypatch.setattr(http_utils, "_post_with_transport", fake_transport)
    http_utils.configure_generate_request_limiter(1)

    async def run():
        first = asyncio.create_task(
            http_utils.post(
                "http://router/generate",
                {"identity": 0},
                headers={"X-SMG-Routing-Key": "session-0"},
            )
        )
        await entered.wait()
        queued = [
            asyncio.create_task(
                http_utils.post(
                    "http://router/generate",
                    {"identity": identity},
                    headers={"X-SMG-Routing-Key": f"session-{identity}"},
                )
            )
            for identity in range(1, 5)
        ]
        with pytest.raises(RuntimeError, match="while requests are in flight"):
            http_utils.reset_generate_request_metrics()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        return await asyncio.gather(*queued)

    try:
        outputs = asyncio.run(run())
        metrics = http_utils.get_generate_request_metrics()
        assert start_order == list(range(5))
        assert [row["identity"] for row in outputs] == list(range(1, 5))
        assert [row["headers"]["X-SMG-Routing-Key"] for row in outputs] == [
            f"session-{identity}" for identity in range(1, 5)
        ]
        assert metrics["rollout/sglang_request/submitted"] == 5
        assert metrics["rollout/sglang_request/completed"] == 4
        assert metrics["rollout/sglang_request/cancelled"] == 1
        assert metrics["rollout/sglang_request/observed_max_inflight"] == 1
        http_utils.reset_generate_request_metrics()
        assert http_utils.get_generate_request_metrics()["rollout/sglang_request/submitted"] == 0
    finally:
        http_utils.configure_generate_request_limiter(512)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
