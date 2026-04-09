"""
每种 OpType 的 shape 推导规则。

规则函数签名：
    (input_shapes: list[list[int]], params: dict) -> list[int]

input_shapes: 上游节点的输出 shape 列表（不含 batch 维，如 [C, H, W]）
params:       节点的参数字典
返回值:       当前节点的输出 shape
"""

from __future__ import annotations
import math
from typing import Callable

from core.ir.model_ir import OpType

# 规则注册表
ShapeRule = Callable[[list[list[int]], dict], list[int]]
_RULES: dict[str, ShapeRule] = {}

def register_rule(op_type: OpType):
    def decorator(fn: ShapeRule) -> ShapeRule:
        _RULES[op_type.value] = fn
        return fn
    return decorator

def get_rule(op_type: str) -> ShapeRule | None:
    return _RULES.get(op_type)

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _conv_output_size(size: int, kernel: int, stride: int,
                      padding: int, dilation: int = 1) -> int:
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

def _pool_output_size(size: int, kernel: int, stride: int,
                      padding: int, dilation: int = 1) -> int:
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

def _as_pair(v) -> tuple[int, int]:
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)

# ─────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────

@register_rule(OpType.INPUT)
def _input(inputs: list[list[int]], params: dict) -> list[int]:
    """Input 节点：shape 来自 params.shape"""
    shape = params.get("shape", [3, 224, 224])
    return [int(s) for s in shape]

@register_rule(OpType.OUTPUT)
def _output(inputs: list[list[int]], params: dict) -> list[int]:
    """Output 节点：透传上游 shape"""
    if not inputs:
        raise ValueError("Output 节点缺少上游输入")
    return inputs[0]

# ─────────────────────────────────────────────
# 卷积族
# ─────────────────────────────────────────────

@register_rule(OpType.CONV2D)
def _conv2d(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("Conv2d 缺少输入")
    _, H, W = inputs[0]   # [C, H, W]

    out_channels = int(params.get("out_channels", 64))
    kernel_h, kernel_w = _as_pair(params.get("kernel_size", 3))
    stride_h, stride_w = _as_pair(params.get("stride", 1))
    pad_h,    pad_w    = _as_pair(params.get("padding", 0))
    dil_h,    dil_w    = _as_pair(params.get("dilation", 1))

    out_H = _conv_output_size(H, kernel_h, stride_h, pad_h, dil_h)
    out_W = _conv_output_size(W, kernel_w, stride_w, pad_w, dil_w)
    return [out_channels, out_H, out_W]

@register_rule(OpType.DEPTHWISE_CONV2D)
def _dw_conv2d(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("DepthwiseConv2d 缺少输入")
    C, H, W = inputs[0]

    kernel_h, kernel_w = _as_pair(params.get("kernel_size", 3))
    stride_h, stride_w = _as_pair(params.get("stride", 1))
    pad_h,    pad_w    = _as_pair(params.get("padding", 1))

    out_H = _conv_output_size(H, kernel_h, stride_h, pad_h)
    out_W = _conv_output_size(W, kernel_w, stride_w, pad_w)
    return [C, out_H, out_W]   # 深度卷积不改变通道数

@register_rule(OpType.CONV_TRANSPOSE2D)
def _conv_transpose2d(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("ConvTranspose2d 缺少输入")
    _, H, W = inputs[0]

    out_channels = int(params.get("out_channels", 64))
    kernel_h, kernel_w = _as_pair(params.get("kernel_size", 2))
    stride_h, stride_w = _as_pair(params.get("stride", 2))
    pad_h,    pad_w    = _as_pair(params.get("padding", 0))
    out_pad_h, out_pad_w = _as_pair(params.get("output_padding", 0))

    out_H = (H - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
    out_W = (W - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w
    return [out_channels, out_H, out_W]

# ─────────────────────────────────────────────
# 归一化（不改变 shape）
# ─────────────────────────────────────────────

def _passthrough(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("缺少上游输入")
    return inputs[0]

for _op in (OpType.BATCH_NORM2D, OpType.LAYER_NORM,
            OpType.RELU, OpType.LEAKY_RELU, OpType.GELU,
            OpType.SIGMOID, OpType.TANH,
            OpType.DROPOUT, OpType.DROPOUT2D):
    _RULES[_op.value] = _passthrough

# ─────────────────────────────────────────────
# 池化
# ─────────────────────────────────────────────

@register_rule(OpType.MAX_POOL2D)
def _maxpool2d(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("MaxPool2d 缺少输入")
    C, H, W = inputs[0]

    kernel_h, kernel_w = _as_pair(params.get("kernel_size", 2))
    stride_h, stride_w = _as_pair(params.get("stride", params.get("kernel_size", 2)))
    pad_h,    pad_w    = _as_pair(params.get("padding", 0))
    dil_h,    dil_w    = _as_pair(params.get("dilation", 1))

    out_H = _pool_output_size(H, kernel_h, stride_h, pad_h, dil_h)
    out_W = _pool_output_size(W, kernel_w, stride_w, pad_w, dil_w)
    return [C, out_H, out_W]

@register_rule(OpType.AVG_POOL2D)
def _avgpool2d(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("AvgPool2d 缺少输入")
    C, H, W = inputs[0]

    kernel_h, kernel_w = _as_pair(params.get("kernel_size", 2))
    stride_h, stride_w = _as_pair(params.get("stride", params.get("kernel_size", 2)))
    pad_h,    pad_w    = _as_pair(params.get("padding", 0))

    out_H = _pool_output_size(H, kernel_h, stride_h, pad_h)
    out_W = _pool_output_size(W, kernel_w, stride_w, pad_w)
    return [C, out_H, out_W]

@register_rule(OpType.ADAPTIVE_AVG_POOL2D)
def _adaptive_avgpool2d(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("AdaptiveAvgPool2d 缺少输入")
    C = inputs[0][0]
    output_size = params.get("output_size", [1, 1])
    if isinstance(output_size, (int, float)):
        h = w = int(output_size)
    else:
        h, w = int(output_size[0]), int(output_size[1])
    return [C, h, w]

# ─────────────────────────────────────────────
# 线性
# ─────────────────────────────────────────────

@register_rule(OpType.LINEAR)
def _linear(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("Linear 缺少输入")
    upstream = inputs[0]

    # 校验 in_features
    declared_in = int(params.get("in_features", 0))
    actual_in   = upstream[-1]   # 最后一维
    if declared_in and declared_in != actual_in:
        raise ValueError(
            f"in_features={declared_in} 与上游输出 {actual_in} 不匹配"
        )
    out_features = int(params.get("out_features", 10))
    # 保留除最后一维外的所有维度
    return [*upstream[:-1], out_features]

# ─────────────────────────────────────────────
# 结构变换
# ─────────────────────────────────────────────

@register_rule(OpType.FLATTEN)
def _flatten(inputs: list[list[int]], params: dict) -> list[int]:
    if not inputs:
        raise ValueError("Flatten 缺少输入")
    shape = inputs[0]
    start_dim = int(params.get("start_dim", 1))

    # 去掉 batch 维，start_dim 对应 shape 中的索引（batch=0 已省略）
    # 实际 start_dim=1 → 从 shape[0] 开始展平
    actual_start = start_dim - 1  # 因为我们省略了 batch 维
    if actual_start < 0:
        actual_start = 0
    prefix = shape[:actual_start]
    flat   = math.prod(shape[actual_start:])
    return [*prefix, flat]

# ─────────────────────────────────────────────
# 多输入融合
# ─────────────────────────────────────────────

@register_rule(OpType.ADD)
def _add(inputs: list[list[int]], params: dict) -> list[int]:
    if len(inputs) < 2:
        raise ValueError("Add 需要至少两个输入")
    base = inputs[0]
    for i, inp in enumerate(inputs[1:], 1):
        if inp != base:
            raise ValueError(
                f"Add 输入 shape 不一致：inputs[0]={base} vs inputs[{i}]={inp}"
            )
    return base

@register_rule(OpType.CONCAT)
def _concat(inputs: list[list[int]], params: dict) -> list[int]:
    if len(inputs) < 2:
        raise ValueError("Concat 需要至少两个输入")
    dim = int(params.get("dim", 1))

    # 不含 batch 维，dim=1 → 拼接 shape 中的第 0 维（通道）
    cat_dim = dim - 1   # 省略 batch 维后的索引

    base = list(inputs[0])
    for i, inp in enumerate(inputs[1:], 1):
        if len(inp) != len(base):
            raise ValueError(
                f"Concat 输入维度数不一致：inputs[0]={base} vs inputs[{i}]={inp}"
            )
        for d, (a, b) in enumerate(zip(base, inp)):
            if d == cat_dim:
                base[d] += b
            elif a != b:
                raise ValueError(
                    f"Concat 非拼接维度 {d+1} 不一致：{a} vs {b}"
                )
    return base