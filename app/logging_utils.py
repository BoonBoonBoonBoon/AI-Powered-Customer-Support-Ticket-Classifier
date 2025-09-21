import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

REQUEST_ID_CTX_KEY = "request_id"
request_id_ctx_var: ContextVar[Optional[str]] = ContextVar(REQUEST_ID_CTX_KEY, default=None)


def get_request_id() -> Optional[str]:
    return request_id_ctx_var.get()


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": int(time.time() * 1000),
        }
        rid = get_request_id()
        if rid:
            log["request_id"] = rid
        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)
        for key in ("path", "method", "status_code", "latency_ms"):
            if hasattr(record, key):
                log[key] = getattr(record, key)
        return json.dumps(log, ensure_ascii=False)


def configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    structured = os.getenv("STRUCTURED_LOGS", "true").lower() in {"1", "true", "yes", "on"}
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    if structured:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").handlers = []


class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rid = headers.get("x-request-id", str(uuid.uuid4()))
        token = request_id_ctx_var.set(rid)
        start = time.time()

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                headers_list = message.setdefault("headers", [])
                headers_list.append((b"x-request-id", rid.encode()))
                status_code = message.get("status")
                latency_ms = int((time.time() - start) * 1000)
                logging.getLogger("request").info(
                    "request completed",
                    extra={
                        "path": scope.get("path"),
                        "method": scope.get("method"),
                        "status_code": status_code,
                        "latency_ms": latency_ms,
                    },
                )
            await send(message)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_ctx_var.reset(token)


class MaxBodySizeMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        total = 0
        more_body = True
        body_chunks = []
        # Aggregate body manually to enforce limit
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, lambda: message, send)
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                await send({
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"x-request-id", (get_request_id() or '').encode()),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"detail":"Request body too large"}',
                })
                return
            body_chunks.append(chunk)
            more_body = message.get("more_body", False)
        full_body = b"".join(body_chunks)

        async def receive_again():
            return {"type": "http.request", "body": full_body, "more_body": False}

        await self.app(scope, receive_again, send)
