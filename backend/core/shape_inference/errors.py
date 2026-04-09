"""
Shape Inference 错误类型。
"""

from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ShapeError:
    """单个节点的 shape 推导错误"""
    node_id:   str
    node_name: str
    op_type:   str
    message:   str

class ShapeInferenceError(Exception):
    """Shape Inference 整体失败时抛出"""
    def __init__(self, errors: list[ShapeError]):
        self.errors = errors
        msg = "\n".join(f"[{e.node_id}] {e.op_type}: {e.message}" for e in errors)
        super().__init__(msg)

@dataclass
class ShapeInferenceResult:
    """推导结果"""
    success: bool
    # node_id → output_shape
    shapes:  dict[str, list[int]] = field(default_factory=dict)
    errors:  list[ShapeError]     = field(default_factory=list)