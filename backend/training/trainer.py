"""
Trainer：接收 ExperimentIR + 已构建的 model / dataloader，
执行标准训练循环，通过 callback 上报进度。
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader

from core.ir.experiment_ir import (
    DeviceType,
    EpochMetrics,
    ExperimentIR,
    LossFnType,
    OptimizerType,
    SchedulerType,
)

# ─────────────────────────────────────────────
# 回调类型定义
# ─────────────────────────────────────────────

OnEpochEnd = Callable[[EpochMetrics], None]
OnTrainEnd = Callable[[bool, str], None]   # (success, message)

class Trainer:
    """
    通用训练器。

    使用方式：
        trainer = Trainer(ir, model, train_loader, val_loader)
        trainer.on_epoch_end = my_callback
        trainer.fit()
    """

    def __init__(
        self,
        ir: ExperimentIR,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.ir = ir
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = self._resolve_device()
        self.model.to(self.device)

        self.optimizer  = self._build_optimizer()
        self.scheduler  = self._build_scheduler()
        self.criterion  = self._build_criterion()
        self.scaler     = GradScaler(enabled=ir.hyper_params.use_amp and self.device.type == "cuda")

        # 外部注入回调
        self.on_epoch_end: OnEpochEnd | None = None
        self.on_train_end: OnTrainEnd | None = None

        # 停止标志（Celery 取消时设置）
        self._stop_requested: bool = False

        # 最优指标追踪
        self._best_val_acc:  float = 0.0
        self._best_val_loss: float = float("inf")
        self._best_epoch:    int   = 0

    # ─────────────────────────────────────────
    # 主训练入口
    # ─────────────────────────────────────────

    def fit(self) -> list[EpochMetrics]:
        """执行完整训练，返回所有 epoch 的指标历史"""
        history: list[EpochMetrics] = []
        hp = self.ir.hyper_params

        for epoch in range(1, hp.epochs + 1):
            if self._stop_requested:
                break

            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc     = self._validate_one_epoch()

            # 获取当前 lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 更新调度器
            self._step_scheduler(val_loss)

            elapsed = time.time() - t0
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr,
                elapsed_s=round(elapsed, 2),
            )
            history.append(metrics)

            # 更新最优指标
            if val_acc > self._best_val_acc:
                self._best_val_acc  = val_acc
                self._best_val_loss = val_loss
                self._best_epoch    = epoch

            # 触发回调
            if self.on_epoch_end:
                self.on_epoch_end(metrics)

        if self.on_train_end:
            self.on_train_end(True, "Training completed")

        return history

    def request_stop(self) -> None:
        """外部调用，请求提前停止训练"""
        self._stop_requested = True

    # ─────────────────────────────────────────
    # 单 epoch 训练
    # ─────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        hp         = self.ir.hyper_params

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=hp.use_amp and self.device.type == "cuda"):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            # 梯度裁剪
            if hp.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), hp.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # OneCycleLR 需要每 step 更新
            if self.ir.hyper_params.scheduler.type == SchedulerType.ONE_CYCLE:
                self.scheduler.step()  # type: ignore

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        avg_loss = total_loss / total
        avg_acc  = correct / total
        return avg_loss, avg_acc

    # ─────────────────────────────────────────
    # 验证
    # ─────────────────────────────────────────

    def _validate_one_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs    = self.model(images)
                loss       = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total

    # ─────────────────────────────────────────
    # 调度器 step
    # ─────────────────────────────────────────

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return
        sched_type = self.ir.hyper_params.scheduler.type
        if sched_type == SchedulerType.REDUCE_ON_PLATEAU:
            self.scheduler.step(val_loss)           # type: ignore
        elif sched_type != SchedulerType.ONE_CYCLE:
            self.scheduler.step()

    # ─────────────────────────────────────────
    # 构建工具
    # ─────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        backend = self.ir.backend
        device_type = getattr(backend, "device", DeviceType.AUTO)

        if device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        return torch.device(device_type.value)

    def _build_optimizer(self):
        cfg    = self.ir.hyper_params.optimizer
        params = self.model.parameters()

        match cfg.type:
            case OptimizerType.SGD:
                return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)
            case OptimizerType.ADAM:
                return Adam(params, lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)
            case OptimizerType.ADAMW:
                return AdamW(params, lr=cfg.lr, betas=cfg.betas,
                             weight_decay=cfg.weight_decay)
            case OptimizerType.RMSPROP:
                return RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            case _:
                raise NotImplementedError(f"不支持的优化器: {cfg.type}")

    def _build_scheduler(self):
        cfg  = self.ir.hyper_params.scheduler
        opt  = self.optimizer
        hp   = self.ir.hyper_params

        match cfg.type:
            case SchedulerType.NONE:
                return None
            case SchedulerType.STEP_LR:
                return StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
            case SchedulerType.COSINE_ANNEALING:
                return CosineAnnealingLR(opt, T_max=cfg.t_max, eta_min=cfg.eta_min)
            case SchedulerType.REDUCE_ON_PLATEAU:
                return ReduceLROnPlateau(opt, patience=cfg.patience, factor=cfg.factor)
            case SchedulerType.ONE_CYCLE:
                return OneCycleLR(
                    opt,
                    max_lr=cfg.max_lr,
                    epochs=hp.epochs,
                    steps_per_epoch=len(self.train_loader),
                )
            case _:
                raise NotImplementedError(f"不支持的调度器: {cfg.type}")

    def _build_criterion(self) -> nn.Module:
        cfg = self.ir.hyper_params.loss_fn
        match cfg.type:
            case LossFnType.CROSS_ENTROPY | LossFnType.LABEL_SMOOTHING_CE:
                return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
            case _:
                raise NotImplementedError(f"不支持的损失函数: {cfg.type}")