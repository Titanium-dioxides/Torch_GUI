"""
统一 API 请求 / 响应数据结构。
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """所有接口的统一响应包装"""
    success: bool = True
    data:    T | None = None
    message: str = "ok"
    code:    int = 200

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    code:    int
    detail:  Any = None

# ── Model IR ──

class CreateModelIRRequest(BaseModel):
    """前端提交的 Model IR（直接复用 ModelIR Pydantic 模型）"""
    pass   # 直接在 router 中用 ModelIR 作为 Body

class ModelIRSummary(BaseModel):
    """列表接口返回的简要信息"""
    id:      str
    name:    str
    version: str
    node_count: int
    edge_count: int

# ── Data IR ──

class DataIRSummary(BaseModel):
    id:          str
    name:        str
    version:     str
    source_type: str
    num_classes: int

# ── Experiment ──

class SubmitExperimentRequest(BaseModel):
    """提交训练实验"""
    experiment_ir_id: str = Field(..., description="已注册的 ExperimentIR ID")

class ExperimentStatusResponse(BaseModel):
    experiment_id:  str
    celery_task_id: str | None
    status:         str
    current_epoch:  int | None = None
    total_epochs:   int | None = None
    train_loss:     float | None = None
    train_acc:      float | None = None
    val_loss:       float | None = None
    val_acc:        float | None = None
    error:          str | None = None

class ExperimentSummary(BaseModel):
    id:           str
    name:         str
    model_ir_id:  str
    data_ir_id:   str
    status:       str
    best_val_acc: float | None = None
    total_epochs: int = 0

class CancelResponse(BaseModel):
    task_id: str
    status:  str