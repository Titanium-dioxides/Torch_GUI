"""
Model IR — 神经网络结构的中间表示。

设计原则：
- 与前端 React Flow 的数据格式对齐（节点/边）
- 完全可序列化为 JSON
- 与具体框架解耦，codegen 层负责翻译
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────
# 枚举：支持的算子类型（MVP 范围）
# ─────────────────────────────────────────────

class OpType(str, Enum):
    # 输入输出
    INPUT = "Input"
    OUTPUT = "Output"

    # 卷积族
    CONV2D = "Conv2d"
    DEPTHWISE_CONV2D = "DepthwiseConv2d"
    CONV_TRANSPOSE2D = "ConvTranspose2d"

    # 归一化
    BATCH_NORM2D = "BatchNorm2d"
    LAYER_NORM = "LayerNorm"

    # 激活
    RELU = "ReLU"
    LEAKY_RELU = "LeakyReLU"
    GELU = "GELU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"

    # 池化
    MAX_POOL2D = "MaxPool2d"
    AVG_POOL2D = "AvgPool2d"
    ADAPTIVE_AVG_POOL2D = "AdaptiveAvgPool2d"

    # 全连接
    LINEAR = "Linear"

    # 结构
    FLATTEN = "Flatten"
    DROPOUT = "Dropout"
    DROPOUT2D = "Dropout2d"

    # 多输入
    ADD = "Add"           # 残差相加
    CONCAT = "Concat"     # 通道拼接

# ─────────────────────────────────────────────
# 节点参数（每种算子独立定义，用 Union 聚合）
# ─────────────────────────────────────────────

class Conv2dParams(BaseModel):
    in_channels: int = 3
    out_channels: int = 64
    kernel_size: int | tuple[int, int] = 3
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 1
    dilation: int = 1
    groups: int = 1
    bias: bool = True

class BatchNorm2dParams(BaseModel):
    num_features: int = 64
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True

class LayerNormParams(BaseModel):
    normalized_shape: list[int] = Field(default_factory=lambda: [64])
    eps: float = 1e-5

class LinearParams(BaseModel):
    in_features: int = 512
    out_features: int = 10
    bias: bool = True

class MaxPool2dParams(BaseModel):
    kernel_size: int | tuple[int, int] = 2
    stride: int | tuple[int, int] | None = None
    padding: int = 0

class AvgPool2dParams(BaseModel):
    kernel_size: int | tuple[int, int] = 2
    stride: int | tuple[int, int] | None = None
    padding: int = 0

class AdaptiveAvgPool2dParams(BaseModel):
    output_size: int | tuple[int, int] = (1, 1)

class FlattenParams(BaseModel):
    start_dim: int = 1
    end_dim: int = -1

class DropoutParams(BaseModel):
    p: float = 0.5
    inplace: bool = False

class LeakyReLUParams(BaseModel):
    negative_slope: float = 0.01
    inplace: bool = False

class ConcatParams(BaseModel):
    dim: int = 1   # 默认在 channel 维度拼接

class InputParams(BaseModel):
    """描述模型输入的形状，不含 batch 维度"""
    shape: list[int] = Field(
        default_factory=lambda: [3, 224, 224],
        description="[C, H, W]"
    )

class OutputParams(BaseModel):
    """输出节点，通常只是一个 pass-through 标记"""
    num_classes: int = 10

# 空参数（用于无参数算子）
class EmptyParams(BaseModel):
    pass

# ─────────────────────────────────────────────
# 核心节点定义
# ─────────────────────────────────────────────

class IRNode(BaseModel):
    """
    图中的单个节点，对应一个神经网络层或操作。
    params 字段为 Any，具体类型由 op_type 决定。
    """
    id: str = Field(..., description="节点唯一 ID，与前端 React Flow 节点 ID 对齐")
    op_type: OpType
    name: str = Field(..., description="用户自定义的层名称，用于生成代码中的变量名")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="算子参数，序列化为 dict，codegen 层负责解析"
    )

    # 可选：前端布局信息（不影响编译）
    position: dict[str, float] | None = Field(
        default=None,
        description="前端画布坐标，仅用于 UI，不参与编译"
    )

    # 可选：shape 推导结果（由后端填充）
    output_shape: list[int] | None = Field(
        default=None,
        description="推导出的输出 shape，不含 batch 维度"
    )

# ─────────────────────────────────────────────
# 边定义
# ─────────────────────────────────────────────

class IREdge(BaseModel):
    """
    有向边，描述两个节点之间的数据流。
    支持多输入（Add/Concat 节点会有多条入边）。
    """
    id: str
    source: str = Field(..., description="源节点 ID")
    target: str = Field(..., description="目标节点 ID")
    source_handle: str | None = None   # 预留：多输出节点使用
    target_handle: str | None = None   # 预留：多输入节点的输入槽

# ─────────────────────────────────────────────
# 顶层 Model IR
# ─────────────────────────────────────────────

class ModelIR(BaseModel):
    """
    完整的模型 IR，前端画布序列化后的顶层结构。
    """
    id: str = Field(..., description="模型唯一 ID")
    name: str = Field(default="MyModel", description="模型名称，用于生成类名")
    version: str = Field(default="1.0.0")

    nodes: list[IRNode] = Field(default_factory=list)
    edges: list[IREdge] = Field(default_factory=list)

    # 元信息
    description: str = ""
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "ModelIR":
        """基础校验：节点 ID 唯一，边引用的节点必须存在"""
        node_ids = {n.id for n in self.nodes}

        # 检查 ID 唯一性
        if len(node_ids) != len(self.nodes):
            raise ValueError("存在重复的节点 ID")

        # 检查边引用合法性
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"边 {edge.id} 的 source '{edge.source}' 不存在")
            if edge.target not in node_ids:
                raise ValueError(f"边 {edge.id} 的 target '{edge.target}' 不存在")

        return self

    def get_node(self, node_id: str) -> IRNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_predecessors(self, node_id: str) -> list[str]:
        """返回某节点所有前驱节点的 ID 列表"""
        return [e.source for e in self.edges if e.target == node_id]

    def get_successors(self, node_id: str) -> list[str]:
        """返回某节点所有后继节点的 ID 列表"""
        return [e.target for e in self.edges if e.source == node_id]