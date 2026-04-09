"""
FastAPI 应用主入口。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from api.middleware import register_exception_handlers, register_middleware
from api.routers import data_ir, experiment, model_ir, stream
from api.routers import shape_infer
# ── 应用实例 ──
app = FastAPI(
    title="NoCode PyTorch Platform API",
    description="无代码深度学习平台 REST API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── 中间件 & 异常处理 ──
register_middleware(app)
register_exception_handlers(app)

# ── 路由注册 ──
API_PREFIX = "/api/v1"
app.include_router(model_ir.router,   prefix=API_PREFIX)
app.include_router(data_ir.router,    prefix=API_PREFIX)
app.include_router(experiment.router, prefix=API_PREFIX)
app.include_router(stream.router,     prefix=API_PREFIX)
app.include_router(shape_infer.router, prefix=API_PREFIX)
# ── 健康检查 ──
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# ── 自定义 OpenAPI Schema ──
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi  # type: ignore