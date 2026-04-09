"""
Experiment IR CRUD + 训练控制（提交 / 查状态 / 取消）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store
from api.schemas import (
    ApiResponse,
    CancelResponse,
    ExperimentStatusResponse,
    ExperimentSummary,
    SubmitExperimentRequest,
)
from core.ir.experiment_ir import ExperimentIR, ExperimentRun, ExperimentStatus
from repository.memory_store import MemoryStore
from tasks.train_task import cancel_training, run_training

router = APIRouter(prefix="/experiments", tags=["Experiment"])

# ─────────────────────────────────────────────
# Experiment IR CRUD
# ─────────────────────────────────────────────

@router.post("", response_model=ApiResponse[ExperimentSummary])
async def create_experiment(
    ir: ExperimentIR,
    store: MemoryStore = Depends(get_store),
):
    """保存一个 Experiment IR"""
    # 校验关联的 Model IR / Data IR 存在
    if not store.get_model_ir(ir.model_ir_id):
        raise HTTPException(
            status_code=404, detail=f"Model IR '{ir.model_ir_id}' 不存在"
        )
    if not store.get_data_ir(ir.data_ir_id):
        raise HTTPException(
            status_code=404, detail=f"Data IR '{ir.data_ir_id}' 不存在"
        )
    store.save_experiment(ir)
    return ApiResponse(data=_to_summary(ir, store), message="Experiment 保存成功")

@router.get("", response_model=ApiResponse[list[ExperimentSummary]])
async def list_experiments(store: MemoryStore = Depends(get_store)):
    return ApiResponse(
        data=[_to_summary(ir, store) for ir in store.list_experiments()]
    )

@router.get("/{exp_id}", response_model=ApiResponse[ExperimentIR])
async def get_experiment(exp_id: str, store: MemoryStore = Depends(get_store)):
    ir = store.get_experiment(exp_id)
    if not ir:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' 不存在")
    return ApiResponse(data=ir)

@router.delete("/{exp_id}", response_model=ApiResponse[None])
async def delete_experiment(exp_id: str, store: MemoryStore = Depends(get_store)):
    if not store.delete_experiment(exp_id):
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' 不存在")
    return ApiResponse(message="删除成功")

# ─────────────────────────────────────────────
# 训练控制
# ─────────────────────────────────────────────

@router.post("/{exp_id}/submit", response_model=ApiResponse[ExperimentStatusResponse])
async def submit_training(
    exp_id: str,
    store: MemoryStore = Depends(get_store),
):
    """
    提交训练任务到 Celery 队列。
    同一实验不允许重复提交（RUNNING 状态时拒绝）。
    """
    ir = store.get_experiment(exp_id)
    if not ir:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' 不存在")

    # 将 IR 注入 Celery task 的内存 Store
    # （生产阶段从数据库加载，此处直接同步内存）
    from tasks.train_task import (
        register_data_ir,
        register_experiment_ir,
        register_model_ir,
    )
    register_model_ir(store.get_model_ir(ir.model_ir_id))
    register_data_ir(store.get_data_ir(ir.data_ir_id))
    register_experiment_ir(ir)

    # 检查是否已在运行
    existing_run = store.get_run(exp_id)
    if existing_run and existing_run.status == ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment '{exp_id}' 已在运行中，请先取消再重新提交",
        )

    # 提交 Celery 任务
    task = run_training.delay(exp_id)

    # 创建并保存 Run 状态
    run = ExperimentRun(
        experiment_id=exp_id,
        celery_task_id=task.id,
        status=ExperimentStatus.PENDING,
    )
    store.save_run(run)

    return ApiResponse(
        data=ExperimentStatusResponse(
            experiment_id=exp_id,
            celery_task_id=task.id,
            status=ExperimentStatus.PENDING.value,
        ),
        message="训练任务已提交",
    )

@router.get("/{exp_id}/status", response_model=ApiResponse[ExperimentStatusResponse])
async def get_training_status(
    exp_id: str,
    store: MemoryStore = Depends(get_store),
):
    """
    查询训练状态。
    优先从 Celery backend 获取实时进度，合并本地 Run 状态。
    """
    run = store.get_run(exp_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' 尚未提交训练")

    response = ExperimentStatusResponse(
        experiment_id=exp_id,
        celery_task_id=run.celery_task_id,
        status=run.status.value,
    )

    # 从 Celery backend 拉取实时 meta
    if run.celery_task_id:
        from celery.result import AsyncResult
        from tasks.celery_app import celery_app as _celery

        celery_result = AsyncResult(run.celery_task_id, app=_celery)
        if celery_result.info and isinstance(celery_result.info, dict):
            info = celery_result.info
            response.status        = info.get("status", response.status)
            response.current_epoch = info.get("current_epoch")
            response.total_epochs  = info.get("total_epochs")
            response.train_loss    = info.get("train_loss")
            response.train_acc     = info.get("train_acc")
            response.val_loss      = info.get("val_loss")
            response.val_acc       = info.get("val_acc")
            response.error         = info.get("error")

    return ApiResponse(data=response)

@router.post("/{exp_id}/cancel", response_model=ApiResponse[CancelResponse])
async def cancel_experiment(
    exp_id: str,
    store: MemoryStore = Depends(get_store),
):
    """取消正在运行的训练任务"""
    run = store.get_run(exp_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' 尚未提交训练")
    if run.status not in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"任务状态为 '{run.status.value}'，无法取消",
        )
    if not run.celery_task_id:
        raise HTTPException(status_code=400, detail="任务 ID 不存在，无法取消")

    cancel_training.delay(run.celery_task_id)
    run.status = ExperimentStatus.CANCELLED
    store.save_run(run)

    return ApiResponse(
        data=CancelResponse(task_id=run.celery_task_id, status="cancel_requested"),
        message="取消请求已发送",
    )

# ── 工具 ──

def _to_summary(ir: ExperimentIR, store: MemoryStore) -> ExperimentSummary:
    run = store.get_run(ir.id)
    return ExperimentSummary(
        id=ir.id,
        name=ir.name,
        model_ir_id=ir.model_ir_id,
        data_ir_id=ir.data_ir_id,
        status=run.status.value if run else ExperimentStatus.PENDING.value,
        best_val_acc=run.result.best_val_acc if run else None,
        total_epochs=run.result.total_epochs if run else 0,
    )