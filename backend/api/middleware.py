"""
全局中间件：统一异常处理 + 请求日志 + CORS。
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse

def register_middleware(app: FastAPI) -> None:

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # 生产环境替换为前端域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 请求日志 + 请求 ID ──
    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start      = time.time()
        response   = await call_next(request)
        elapsed    = round((time.time() - start) * 1000, 1)
        print(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed}ms)"
        )
        response.headers["X-Request-ID"] = request_id
        return response

def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                message=str(exc), code=400
            ).model_dump(),
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                message=f"资源不存在: {exc}", code=404
            ).model_dump(),
        )

    @app.exception_handler(NotImplementedError)
    async def not_impl_handler(request: Request, exc: NotImplementedError):
        return JSONResponse(
            status_code=501,
            content=ErrorResponse(
                message=str(exc), code=501
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message="服务器内部错误", code=500, detail=str(exc)
            ).model_dump(),
        )