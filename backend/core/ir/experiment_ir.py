"""
Experiment IR — 一次完整训练实验的中间表示。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────
# 枚举
# ─────────────────────────────────────────────

class ExperimentStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"

class OptimizerType(str, Enum):
    SGD     = "SGD"
    ADAM    = "Adam"
    ADAMW   = "AdamW"
    RMSPROP = "RMSprop"

class SchedulerType(str, Enum):
    NONE              = "none"
    STEP_LR           = "StepLR"
    COSINE_ANNEALING  = "CosineAnnealingLR"
    REDUCE_ON_PLATEAU = "ReduceLROnPlateau"
    ONE_CYCLE         = "OneCycleLR"

class LossFnType(str, Enum):
    CROSS_ENTROPY       = "CrossEntropyLoss"
    LABEL_SMOOTHING_CE  = "LabelSmoothingCE"

class RuntimeBackend(str, Enum):
    LOCAL  = "local"
    SSH    = "ssh"      # MVP 预留
    DOCKER = "docker"   # 预留

class DeviceType(str, Enum):
    AUTO = "auto"
    CPU  = "cpu"
    CUDA = "cuda"
    MPS  = "mps"

# ─────────────────────────────────────────────
# 优化器 & 调度器 & 损失函数配置
# ─────────────────────────────────────────────

class OptimizerConfig(BaseModel):
    type:         OptimizerType          = OptimizerType.ADAMW
    lr:           float                  = Field(default=1e-3, gt=0)
    weight_decay: float                  = Field(default=1e-4, ge=0)
    momentum:     float                  = Field(default=0.9, ge=0, le=1)
    betas:        tuple[float, float]    = (0.9, 0.999)

class SchedulerConfig(BaseModel):
    type:        SchedulerType      = SchedulerType.COSINE_ANNEALING
    # StepLR
    step_size:   int                = 10
    gamma:       float              = 0.1
    # CosineAnnealingLR
    t_max:       int                = 50
    eta_min:     float              = 1e-6
    # ReduceLROnPlateau
    patience:    int                = 5
    factor:      float              = 0.5
    # OneCycleLR
    max_lr:      float              = 1e-2
    extra_params: dict[str, Any]   = Field(default_factory=dict)

class LossFnConfig(BaseModel):
    type:             LossFnType        = LossFnType.CROSS_ENTROPY
    label_smoothing:  float             = Field(default=0.0, ge=0.0, le=1.0)
    extra_params:     dict[str, Any]    = Field(default_factory=dict)

# ─────────────────────────────────────────────
# 训练超参数
# ─────────────────────────────────────────────

class TrainHyperParams(BaseModel):
    epochs:          int             = Field(default=50, ge=1)
    optimizer:       OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler:       SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss_fn:         LossFnConfig    = Field(default_factory=LossFnConfig)
    use_amp:         bool            = Field(default=True, description="自动混合精度")
    grad_clip_norm:  float | None    = Field(default=1.0, description="梯度裁剪阈值，None 表示不裁剪")

# ─────────────────────────────────────────────
# 运行时后端配置
# ─────────────────────────────────────────────

class LocalBackendConfig(BaseModel):
    type:   Literal[RuntimeBackend.LOCAL] = RuntimeBackend.LOCAL
    device: DeviceType                    = DeviceType.AUTO

class SSHBackendConfig(BaseModel):
    """MVP 预留，暂不实现训练逻辑"""
    type:        Literal[RuntimeBackend.SSH] = RuntimeBackend.SSH
    host:        str = ""
    port:        int = 22
    username:    str = ""
    key_path:    str = ""
    remote_dir:  str = "/tmp/nocode_experiments"

BackendConfig = LocalBackendConfig | SSHBackendConfig

# ─────────────────────────────────────────────
# Checkpoint 配置
# ─────────────────────────────────────────────

class CheckpointConfig(BaseModel):
    enabled:       bool   = True
    save_dir:      str    = "./checkpoints"
    save_every_n_epochs: int = Field(default=5, ge=1)
    keep_top_k:    int    = Field(default=3, ge=1, description="保留最优的 K 个 checkpoint")
    monitor:       str    = "val_acc"
    mode:          Literal["min", "max"] = "max"

# ─────────────────────────────────────────────
# Logging & Tracking 配置
# ─────────────────────────────────────────────

class LoggingConfig(BaseModel):
    log_every_n_steps: int  = Field(default=10, ge=1)
    use_tensorboard:   bool = True
    tensorboard_dir:   str  = "./runs"

class MLflowConfig(BaseModel):
    enabled:         bool = False
    tracking_uri:    str  = "http://localhost:5000"
    experiment_name: str  = "default"
    run_name:        str  = ""
    tags:            dict[str, str] = Field(default_factory=dict)

# ─────────────────────────────────────────────
# 实验运行时状态（动态填充，不属于静态 IR）
# ─────────────────────────────────────────────

class EpochMetrics(BaseModel):
    epoch:      int
    train_loss: float
    train_acc:  float
    val_loss:   float
    val_acc:    float
    lr:         float
    elapsed_s:  float   # 该 epoch 耗时（秒）

class ExperimentResult(BaseModel):
    best_val_acc:   float | None = None
    best_val_loss:  float | None = None
    best_epoch:     int | None   = None
    best_ckpt_path: str | None   = None
    total_epochs:   int          = 0
    history:        list[EpochMetrics] = Field(default_factory=list)
    error_message:  str | None   = None

class ExperimentRun(BaseModel):
    """运行时状态，与静态 IR 分离存储"""
    experiment_id: str
    celery_task_id: str | None  = None
    status:         ExperimentStatus = ExperimentStatus.PENDING
    created_at:     datetime         = Field(default_factory=datetime.utcnow)
    started_at:     datetime | None  = None
    finished_at:    datetime | None  = None
    result:         ExperimentResult = Field(default_factory=ExperimentResult)

# ─────────────────────────────────────────────
# 顶层 Experiment IR
# ─────────────────────────────────────────────

class ExperimentIR(BaseModel):
    """
    一次训练实验的完整静态描述。
    model_ir_id / data_ir_id 通过 ID 引用，不内嵌，
    运行时由 Builder 从存储层加载。
    """
    id:           str  = Field(..., description="实验唯一 ID")
    name:         str  = Field(default="Experiment")
    version:      str  = Field(default="1.0.0")

    model_ir_id:  str  = Field(..., description="关联的 Model IR ID")
    data_ir_id:   str  = Field(..., description="关联的 Data IR ID")

    hyper_params: TrainHyperParams  = Field(default_factory=TrainHyperParams)
    backend:      BackendConfig     = Field(
        default_factory=LocalBackendConfig,
        discriminator="type",
    )
    checkpoint:   CheckpointConfig  = Field(default_factory=CheckpointConfig)
    logging:      LoggingConfig     = Field(default_factory=LoggingConfig)
    mlflow:       MLflowConfig      = Field(default_factory=MLflowConfig)

    description:  str               = ""
    tags:         list[str]         = Field(default_factory=list)