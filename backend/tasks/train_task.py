"""
训练任务：Celery Task 封装。

职责：
1. 从存储层加载 ModelIR / DataIR / ExperimentIR
2. 构建 model / dataloader
3. 启动 Trainer
4. 实时上报进度（通过 task.update_state）
5. 保存结果到存储层
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any

import torch

from celery import Task
from celery.utils.log import get_task_logger

from core.ir.codegen import PyTorchCodeGen
from core.ir.data_builder import DatasetBuilder, build_dataloader
from core.ir.experiment_ir import (
    EpochMetrics,
    ExperimentIR,
    ExperimentResult,
    ExperimentRun,
    ExperimentStatus,
)
from core.ir.model_ir import ModelIR
from core.ir.data_ir import DataIR
from tasks.celery_app import celery_app
from training.callbacks import CheckpointCallback, ProgressReporter
from training.trainer import Trainer

logger = get_task_logger(__name__)

# ─────────────────────────────────────────────
# 简易内存存储（MVP 阶段替代数据库）
# 生产阶段替换为 SQLAlchemy + PostgreSQL
# ─────────────────────────────────────────────

_MODEL_IR_STORE:  dict[str, ModelIR]      = {}
_DATA_IR_STORE:   dict[str, DataIR]       = {}
_EXPERIMENT_STORE: dict[str, ExperimentIR] = {}
_RUN_STORE:        dict[str, ExperimentRun] = {}

def register_model_ir(ir: ModelIR)       -> None: _MODEL_IR_STORE[ir.id]      = ir
def register_data_ir(ir: DataIR)         -> None: _DATA_IR_STORE[ir.id]       = ir
def register_experiment_ir(ir: ExperimentIR) -> None: _EXPERIMENT_STORE[ir.id] = ir
def get_run(experiment_id: str) -> ExperimentRun | None: return _RUN_STORE.get(experiment_id)

# ─────────────────────────────────────────────
# 构建 PyTorch 模型（从 ModelIR 动态生成并实例化）
# ─────────────────────────────────────────────

def _build_model_from_ir(model_ir: ModelIR) -> torch.nn.Module:
    codegen  = PyTorchCodeGen(model_ir)
    code_str = codegen.generate()
    namespace: dict[str, Any] = {}
    exec(code_str, namespace)  # noqa: S102

    # 找到生成的 nn.Module 类
    import torch.nn as nn
    model_cls = next(
        v for v in namespace.values()
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
    )
    return model_cls()

# ─────────────────────────────────────────────
# Celery Task
# ─────────────────────────────────────────────

class TrainTask(Task):
    """自定义 Task 基类，持有 trainer 引用便于取消"""
    _trainer: Trainer | None = None

@celery_app.task(
    bind=True,
    base=TrainTask,
    name="tasks.train_task.run_training",
    max_retries=0,
)
def run_training(self: TrainTask, experiment_id: str) -> dict[str, Any]:
    """
    Celery 训练任务主体。

    参数：
        experiment_id: ExperimentIR 的 ID

    返回：
        实验结果字典（可被 Celery backend 序列化）
    """
    # ── 初始化 Run 状态 ──
    run = ExperimentRun(
        experiment_id=experiment_id,
        celery_task_id=self.request.id,
        status=ExperimentStatus.RUNNING,
        started_at=datetime.utcnow(),
    )
    _RUN_STORE[experiment_id] = run

    self.update_state(
        state="RUNNING",
        meta={"experiment_id": experiment_id, "status": "running"},
    )

    try:
        # ── 加载 IR ──
        exp_ir   = _EXPERIMENT_STORE[experiment_id]
        model_ir = _MODEL_IR_STORE[exp_ir.model_ir_id]
        data_ir  = _DATA_IR_STORE[exp_ir.data_ir_id]

        logger.info(f"[{experiment_id}] 开始加载模型和数据...")

        # ── 构建模型 ──
        model = _build_model_from_ir(model_ir)
        logger.info(f"[{experiment_id}] 模型构建完成: {model.__class__.__name__}")

        # ── 构建 DataLoader ──
        dataset_bundle = DatasetBuilder(data_ir).build()
        loader_bundle  = build_dataloader(dataset_bundle, data_ir)
        logger.info(f"[{experiment_id}] DataLoader 构建完成: {loader_bundle}")

        # ── 构建 Trainer ──
        trainer = Trainer(
            ir=exp_ir,
            model=model,
            train_loader=loader_bundle.train,
            val_loader=loader_bundle.val,
        )
        self._trainer = trainer

        # ── 进度上报回调 ──
        history: list[EpochMetrics] = []

        def on_epoch_end(metrics: EpochMetrics) -> None:
            history.append(metrics)
            # 通过 Celery update_state 上报当前进度
            self.update_state(
                state="RUNNING",
                meta={
                    "experiment_id": experiment_id,
                    "status":        "running",
                    "current_epoch": metrics.epoch,
                    "total_epochs":  exp_ir.hyper_params.epochs,
                    "train_loss":    round(metrics.train_loss, 6),
                    "train_acc":     round(metrics.train_acc, 6),
                    "val_loss":      round(metrics.val_loss, 6),
                    "val_acc":       round(metrics.val_acc, 6),
                    "lr":            metrics.lr,
                },
            )
            logger.info(
                f"[{experiment_id}] Epoch {metrics.epoch}/{exp_ir.hyper_params.epochs} "
                f"| train_loss={metrics.train_loss:.4f} train_acc={metrics.train_acc:.4f} "
                f"| val_loss={metrics.val_loss:.4f} val_acc={metrics.val_acc:.4f} "
                f"| lr={metrics.lr:.2e} | elapsed={metrics.elapsed_s}s"
            )

        trainer.on_epoch_end = on_epoch_end

        # ── Checkpoint 回调 ──
        ckpt_cb = CheckpointCallback(model=model, config=exp_ir.checkpoint)
        original_on_epoch_end = trainer.on_epoch_end

        def combined_on_epoch_end(metrics: EpochMetrics) -> None:
            original_on_epoch_end(metrics)
            ckpt_cb(metrics)

        trainer.on_epoch_end = combined_on_epoch_end

        # ── 执行训练 ──
        trainer.fit()

        # ── 汇总结果 ──
        best_metrics = max(history, key=lambda m: m.val_acc) if history else None
        result = ExperimentResult(
            best_val_acc=best_metrics.val_acc if best_metrics else None,
            best_val_loss=best_metrics.val_loss if best_metrics else None,
            best_epoch=best_metrics.epoch if best_metrics else None,
            best_ckpt_path=ckpt_cb.best_checkpoint_path(),
            total_epochs=len(history),
            history=history,
        )

        # 更新 run 状态
        run.status      = ExperimentStatus.COMPLETED
        run.finished_at = datetime.utcnow()
        run.result      = result

        logger.info(
            f"[{experiment_id}] 训练完成 | "
            f"best_val_acc={result.best_val_acc:.4f} @ epoch {result.best_epoch}"
        )

        return result.model_dump()

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{experiment_id}] 训练异常:\n{tb}")

        run.status        = ExperimentStatus.FAILED
        run.finished_at   = datetime.utcnow()
        run.result.error_message = str(exc)

        self.update_state(
            state="FAILURE",
            meta={
                "experiment_id": experiment_id,
                "status":        "failed",
                "error":         str(exc),
            },
        )
        raise

@celery_app.task(name="tasks.train_task.cancel_training")
def cancel_training(task_id: str) -> dict[str, str]:
    """
    取消正在运行的训练任务。
    通过 Celery revoke + trainer.request_stop() 双重保障。
    """
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True, signal="SIGTERM")
    return {"status": "cancel_requested", "task_id": task_id}