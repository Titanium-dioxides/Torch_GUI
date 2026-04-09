"""
SSE（Server-Sent Events）实时进度流。

客户端连接后，服务端每秒轮询 Celery backend，
将最新训练指标推送到前端，直到训练结束或连接断开。
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_store
from core.ir.experiment_ir import ExperimentStatus
from repository.memory_store import MemoryStore

router = APIRouter(prefix="/stream", tags=["Stream"])

POLL_INTERVAL = 1.0   # 轮询间隔（秒）

@router.get("/{exp_id}/progress")
async def stream_progress(
    exp_id: str,
    store: MemoryStore = Depends(get_store),
):
    """
    SSE 接口：实时推送训练进度。

    前端使用方式：
        const es = new EventSource(`/api/v1/stream/${expId}/progress`);
        es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    run = store.get_run(exp_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' 尚未提交")

    async def event_generator():
        from celery.result import AsyncResult
        from tasks.celery_app import celery_app as _celery

        terminal_states = {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        }

        while True:
            # 重新从 store 获取最新 run 状态
            current_run = store.get_run(exp_id)
            if current_run is None:
                break

            payload: dict = {
                "experiment_id": exp_id,
                "status":        current_run.status.value,
            }

            # 从 Celery 获取实时 meta
            if current_run.celery_task_id:
                celery_result = AsyncResult(current_run.celery_task_id, app=_celery)
                if celery_result.info and isinstance(celery_result.info, dict):
                    payload.update(celery_result.info)

            # SSE 格式：data: <json>\n\n
            yield f"data: {json.dumps(payload)}\n\n"

            # 训练结束则发送最终结果并关闭流
            if current_run.status in terminal_states:
                final = {
                    "experiment_id": exp_id,
                    "status":        current_run.status.value,
                    "type":          "final",
                    "best_val_acc":  current_run.result.best_val_acc,
                    "best_epoch":    current_run.result.best_epoch,
                    "total_epochs":  current_run.result.total_epochs,
                }
                yield f"data: {json.dumps(final)}\n\n"
                break

            await asyncio.sleep(POLL_INTERVAL)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # 禁用 Nginx 缓冲
        },
    )