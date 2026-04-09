"""
训练回调：Checkpoint 保存 + 进度上报。
"""

from __future__ import annotations

import json
import os
from typing import Callable

import torch
import torch.nn as nn

from core.ir.experiment_ir import CheckpointConfig, EpochMetrics

class CheckpointCallback:
    """
    保存最优 checkpoint，维护 top-k 列表。
    """

    def __init__(self, model: nn.Module, config: CheckpointConfig):
        self.model   = model
        self.config  = config
        self.history: list[tuple[float, str]] = []  # (metric_value, path)

        os.makedirs(config.save_dir, exist_ok=True)

    def __call__(self, metrics: EpochMetrics) -> str | None:
        """每个 epoch 结束后调用，返回保存路径（如果保存了）"""
        cfg = self.config
        if not cfg.enabled:
            return None
        if metrics.epoch % cfg.save_every_n_epochs != 0:
            return None

        monitor_val = getattr(metrics, cfg.monitor, None)
        if monitor_val is None:
            return None

        path = os.path.join(cfg.save_dir, f"epoch_{metrics.epoch:04d}.pt")
        torch.save(
            {
                "epoch":       metrics.epoch,
                "model_state": self.model.state_dict(),
                "metrics":     metrics.model_dump(),
            },
            path,
        )

        # 维护 top-k
        self.history.append((monitor_val, path))
        reverse = cfg.mode == "max"
        self.history.sort(key=lambda x: x[0], reverse=reverse)

        # 删除超出 top-k 的 checkpoint
        while len(self.history) > cfg.keep_top_k:
            _, old_path = self.history.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        return path

    def best_checkpoint_path(self) -> str | None:
        if not self.history:
            return None
        return self.history[0][1]

class ProgressReporter:
    """
    进度上报回调，将指标序列化后写入 Redis，
    供 Celery task.update_state 和前端 SSE 消费。
    """

    def __init__(
        self,
        experiment_id: str,
        push_fn: Callable[[str, dict], None],
    ):
        """
        push_fn: (channel, data) -> None
            实际实现中会调用 redis.publish 或 celery update_state
        """
        self.experiment_id = experiment_id
        self.push_fn       = push_fn

    def __call__(self, metrics: EpochMetrics) -> None:
        payload = {
            "experiment_id": self.experiment_id,
            "type":          "epoch_end",
            **metrics.model_dump(),
        }
        self.push_fn(f"experiment:{self.experiment_id}:progress", payload)