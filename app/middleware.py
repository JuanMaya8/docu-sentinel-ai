"""Pure ASGI middlewares (no FastAPI imports): request id + access log, and a request body size limit.

Framework-free on purpose: they work with any ASGI app and do not buffer bodies like BaseHTTPMiddleware.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Awaitable, Callable

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

logger = logging.getLogger("docu_sentinel_ai.access")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class RequestContextMiddleware:
    """Propagates (or creates) `X-Request-ID`, adds it to the response and logs one JSON line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        request_id = incoming if incoming and _SAFE_ID.match(incoming) else uuid.uuid4().hex[:16]
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status["code"],
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
            )


class _TooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects bodies larger than `max_bytes` with 413 `PAYLOAD_TOO_LARGE` (Spanish message)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send: Send) -> None:
        mb = self.max_bytes / (1024 * 1024)
        shown = f"{mb:.0f} MB" if mb >= 1 else f"{self.max_bytes} bytes"
        body = json.dumps(
            {
                "code": "PAYLOAD_TOO_LARGE",
                "message": f"La solicitud supera el tamaño máximo permitido ({shown}).",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return
        received = 0
        response_started = False

        async def guarded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, tracking_send)
        except _TooLarge:
            if not response_started:
                await self._reject(send)


# --------------------------------------------------------------------------- metrics

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Metrics:
    """Minimal Prometheus registry (text format 0.0.4): requests by method/route/status and a latency histogram."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: dict[tuple[str, str, int], int] = {}
        self.hist: dict[tuple[str, str], list[float]] = {}  # [*bucket counts, sum, count]
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(method, route, status)] = self.requests.get((method, route, status), 0) + 1
            h = self.hist.setdefault((method, route), [0.0] * (len(_BUCKETS) + 2))
            for i, b in enumerate(_BUCKETS):
                if seconds <= b:
                    h[i] += 1
            h[-2] += seconds
            h[-1] += 1

    def inc(self, name: str, labels: dict[str, str] | None = None, by: float = 1) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + by

    def render(self, gauges: dict[str, float] | None = None) -> str:
        def lab(**kv: str) -> str:
            esc = {k: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\n") for k, v in kv.items()}
            return "{" + ",".join(f'{k}="{esc[k]}"' for k in sorted(esc)) + "}"

        out = ["# HELP ds_ai_http_requests_total HTTP requests.", "# TYPE ds_ai_http_requests_total counter"]
        with self._lock:
            for (m, r, s), n in sorted(self.requests.items()):
                out.append(f"ds_ai_http_requests_total{lab(method=m, route=r, status=str(s))} {n}")
            out += ["# HELP ds_ai_http_request_duration_seconds HTTP request duration.", "# TYPE ds_ai_http_request_duration_seconds histogram"]
            for (m, r), h in sorted(self.hist.items()):
                for i, b in enumerate(_BUCKETS):
                    out.append(f"ds_ai_http_request_duration_seconds_bucket{lab(method=m, route=r, le=str(b))} {int(h[i])}")
                out.append(f"ds_ai_http_request_duration_seconds_bucket{lab(method=m, route=r, le='+Inf')} {int(h[-1])}")
                out.append(f"ds_ai_http_request_duration_seconds_sum{lab(method=m, route=r)} {h[-2]:.6f}")
                out.append(f"ds_ai_http_request_duration_seconds_count{lab(method=m, route=r)} {int(h[-1])}")
            names = sorted({k[0] for k in self.counters})
            for name in names:
                out += [f"# HELP {name} Counter.", f"# TYPE {name} counter"]
                for (n, labels), v in sorted(self.counters.items()):
                    if n == name:
                        out.append(f"{name}{lab(**dict(labels)) if labels else ''} {v:g}")
        all_gauges = {"ds_ai_process_uptime_seconds": time.time() - self.started, **(gauges or {})}
        for name, value in all_gauges.items():
            out += [f"# HELP {name} Gauge.", f"# TYPE {name} gauge", f"{name} {value:g}"]
        return "\n".join(out) + "\n"


class MetricsMiddleware:
    """Times every HTTP request; the route label is the route TEMPLATE (e.g. /v1/models/{model_id}/score)."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None) or "unmatched"
            self.metrics.observe(scope.get("method", "GET"), template, status["code"], time.perf_counter() - started)


# --------------------------------------------------------------------------- rate limit


class RateLimitMiddleware:
    """Fixed-window limit per client (API key if sent, else IP) for /v1/* routes. 429 RATE_LIMITED in Spanish.
    `trust_proxy=True` takes the client IP from the first X-Forwarded-For entry (Render, Railway, Fly…)."""

    def __init__(self, app: ASGIApp, per_minute: int, trust_proxy: bool = False, window_seconds: float = 60.0) -> None:
        self.app = app
        self.per_minute = per_minute
        self.trust_proxy = trust_proxy
        self.window = window_seconds
        self._hits: dict[str, tuple[float, int]] = {}

    def _client(self, scope: Scope) -> str:
        key = _header(scope, b"x-api-key")
        if key:
            return "key:" + key[:16]
        if self.trust_proxy:
            fwd = _header(scope, b"x-forwarded-for")
            if fwd:
                return "ip:" + fwd.split(",")[0].strip()
        client = scope.get("client") or ("unknown", 0)
        return f"ip:{client[0]}"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.per_minute <= 0 or not str(scope.get("path", "")).startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        now = time.monotonic()
        who = self._client(scope)
        start, count = self._hits.get(who, (now, 0))
        if now - start >= self.window:
            start, count = now, 0
        count += 1
        self._hits[who] = (start, count)
        if len(self._hits) > 10_000:  # keep memory bounded
            self._hits = {k: v for k, v in self._hits.items() if now - v[0] < self.window}
        if count > self.per_minute:
            retry = max(1, int(self.window - (now - start)))
            body = json.dumps(
                {"code": "RATE_LIMITED", "message": "Demasiadas solicitudes. Intenta de nuevo en un momento."},
                ensure_ascii=False,
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"content-length", str(len(body)).encode()),
                        (b"retry-after", str(retry).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
