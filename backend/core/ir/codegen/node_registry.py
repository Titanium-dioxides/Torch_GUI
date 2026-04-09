"""
算子注册表：维护每种 OpType 对应的 PyTorch 代码片段生成逻辑。

每个算子注册一个 builder 函数：
    (node: IRNode) -> (init_code: str, forward_code: str)

- init_code:    在 __init__ 中的初始化语句，例如 "nn.Conv2d(3, 64, 3)"
- forward_code: 在 forward 中的调用语句，例如 "x = self.conv1(x)"
"""

from typing import Callable
from core.ir.model_ir import IRNode, OpType

# 类型别名
NodeBuilder = Callable[[IRNode], tuple[str, str]]

_REGISTRY: dict[OpType, NodeBuilder] = {}

def register(op_type: OpType):
    """装饰器：注册某 OpType 对应的代码生成函数"""
    def decorator(fn: NodeBuilder) -> NodeBuilder:
        _REGISTRY[op_type] = fn
        return fn
    return decorator

def get_builder(op_type: OpType) -> NodeBuilder:
    if op_type not in _REGISTRY:
        raise NotImplementedError(f"OpType '{op_type}' 尚未注册代码生成器")
    return _REGISTRY[op_type]

# ─────────────────────────────────────────────
# 各算子的代码生成注册
# ─────────────────────────────────────────────

@register(OpType.INPUT)
def _input_builder(node: IRNode) -> tuple[str, str]:
    # Input 节点不生成 nn.Module，只是 forward 的入口
    return "", ""   # init 无需生成，forward 会单独处理

@register(OpType.OUTPUT)
def _output_builder(node: IRNode) -> tuple[str, str]:
    return "", ""

@register(OpType.CONV2D)
def _conv2d_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    in_ch = p.get("in_channels", 3)
    out_ch = p.get("out_channels", 64)
    ks = p.get("kernel_size", 3)
    stride = p.get("stride", 1)
    padding = p.get("padding", 1)
    groups = p.get("groups", 1)
    bias = p.get("bias", True)

    init = (
        f"nn.Conv2d({in_ch}, {out_ch}, kernel_size={ks}, "
        f"stride={stride}, padding={padding}, groups={groups}, bias={bias})"
    )
    return init, ""

@register(OpType.DEPTHWISE_CONV2D)
def _dw_conv2d_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    channels = p.get("in_channels", 64)
    ks = p.get("kernel_size", 3)
    stride = p.get("stride", 1)
    padding = p.get("padding", 1)

    init = (
        f"nn.Conv2d({channels}, {channels}, kernel_size={ks}, "
        f"stride={stride}, padding={padding}, groups={channels}, bias=False)"
    )
    return init, ""

@register(OpType.BATCH_NORM2D)
def _bn2d_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    num_features = p.get("num_features", 64)
    eps = p.get("eps", 1e-5)
    momentum = p.get("momentum", 0.1)
    init = f"nn.BatchNorm2d({num_features}, eps={eps}, momentum={momentum})"
    return init, ""

@register(OpType.LAYER_NORM)
def _ln_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    normalized_shape = p.get("normalized_shape", [64])
    init = f"nn.LayerNorm({normalized_shape})"
    return init, ""

@register(OpType.RELU)
def _relu_builder(node: IRNode) -> tuple[str, str]:
    return "nn.ReLU(inplace=True)", ""

@register(OpType.LEAKY_RELU)
def _leaky_relu_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    slope = p.get("negative_slope", 0.01)
    return f"nn.LeakyReLU(negative_slope={slope}, inplace=True)", ""

@register(OpType.GELU)
def _gelu_builder(node: IRNode) -> tuple[str, str]:
    return "nn.GELU()", ""

@register(OpType.SIGMOID)
def _sigmoid_builder(node: IRNode) -> tuple[str, str]:
    return "nn.Sigmoid()", ""

@register(OpType.TANH)
def _tanh_builder(node: IRNode) -> tuple[str, str]:
    return "nn.Tanh()", ""

@register(OpType.MAX_POOL2D)
def _maxpool_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    ks = p.get("kernel_size", 2)
    stride = p.get("stride", None)
    padding = p.get("padding", 0)
    stride_str = f", stride={stride}" if stride is not None else ""
    init = f"nn.MaxPool2d(kernel_size={ks}{stride_str}, padding={padding})"
    return init, ""

@register(OpType.AVG_POOL2D)
def _avgpool_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    ks = p.get("kernel_size", 2)
    stride = p.get("stride", None)
    stride_str = f", stride={stride}" if stride is not None else ""
    init = f"nn.AvgPool2d(kernel_size={ks}{stride_str})"
    return init, ""

@register(OpType.ADAPTIVE_AVG_POOL2D)
def _adaptive_avgpool_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    output_size = p.get("output_size", (1, 1))
    init = f"nn.AdaptiveAvgPool2d(output_size={output_size})"
    return init, ""

@register(OpType.LINEAR)
def _linear_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    in_f = p.get("in_features", 512)
    out_f = p.get("out_features", 10)
    bias = p.get("bias", True)
    init = f"nn.Linear({in_f}, {out_f}, bias={bias})"
    return init, ""

@register(OpType.FLATTEN)
def _flatten_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    start_dim = p.get("start_dim", 1)
    init = f"nn.Flatten(start_dim={start_dim})"
    return init, ""

@register(OpType.DROPOUT)
def _dropout_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    prob = p.get("p", 0.5)
    init = f"nn.Dropout(p={prob})"
    return init, ""

@register(OpType.DROPOUT2D)
def _dropout2d_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    prob = p.get("p", 0.5)
    init = f"nn.Dropout2d(p={prob})"
    return init, ""

@register(OpType.ADD)
def _add_builder(node: IRNode) -> tuple[str, str]:
    # Add 不需要 nn.Module，在 forward 里直接做张量加法
    return "", ""

@register(OpType.CONCAT)
def _concat_builder(node: IRNode) -> tuple[str, str]:
    # Concat 也不需要 nn.Module
    return "", ""