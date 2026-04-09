"""
Transform 注册表：将 TransformNode 映射为 torchvision.transforms 对象。

每个 op_type 注册一个 builder：
    (node: TransformNode) -> torchvision.transforms.*
"""

from typing import Callable
from torchvision import transforms

from core.ir.data_ir import TransformNode, TransformOpType

TransformBuilder = Callable[[TransformNode], object]

_REGISTRY: dict[TransformOpType, TransformBuilder] = {}

def register(op_type: TransformOpType):
    def decorator(fn: TransformBuilder) -> TransformBuilder:
        _REGISTRY[op_type] = fn
        return fn
    return decorator

def build_transform(node: TransformNode) -> object:
    if node.op_type not in _REGISTRY:
        raise NotImplementedError(f"TransformOpType '{node.op_type}' 尚未注册")
    return _REGISTRY[node.op_type](node)

def build_pipeline(nodes: list[TransformNode]) -> transforms.Compose:
    """将 TransformNode 列表构建为 torchvision Compose"""
    transform_list = [build_transform(n) for n in nodes]
    return transforms.Compose(transform_list)

# ─────────────────────────────────────────────
# 注册各 Transform
# ─────────────────────────────────────────────

@register(TransformOpType.RESIZE)
def _resize(node: TransformNode):
    size = node.params.get("size", 224)
    return transforms.Resize(size)

@register(TransformOpType.CENTER_CROP)
def _center_crop(node: TransformNode):
    size = node.params.get("size", 224)
    return transforms.CenterCrop(size)

@register(TransformOpType.RANDOM_CROP)
def _random_crop(node: TransformNode):
    size = node.params.get("size", 224)
    padding = node.params.get("padding", None)
    return transforms.RandomCrop(size, padding=padding)

@register(TransformOpType.RANDOM_HORIZONTAL_FLIP)
def _random_hflip(node: TransformNode):
    p = node.params.get("p", 0.5)
    return transforms.RandomHorizontalFlip(p=p)

@register(TransformOpType.RANDOM_VERTICAL_FLIP)
def _random_vflip(node: TransformNode):
    p = node.params.get("p", 0.5)
    return transforms.RandomVerticalFlip(p=p)

@register(TransformOpType.RANDOM_ROTATION)
def _random_rotation(node: TransformNode):
    degrees = node.params.get("degrees", 15)
    return transforms.RandomRotation(degrees=degrees)

@register(TransformOpType.RANDOM_RESIZED_CROP)
def _random_resized_crop(node: TransformNode):
    size = node.params.get("size", 224)
    scale = tuple(node.params.get("scale", [0.08, 1.0]))
    ratio = tuple(node.params.get("ratio", [0.75, 1.333]))
    return transforms.RandomResizedCrop(size, scale=scale, ratio=ratio)

@register(TransformOpType.COLOR_JITTER)
def _color_jitter(node: TransformNode):
    return transforms.ColorJitter(
        brightness=node.params.get("brightness", 0.2),
        contrast=node.params.get("contrast", 0.2),
        saturation=node.params.get("saturation", 0.2),
        hue=node.params.get("hue", 0.0),
    )

@register(TransformOpType.GRAYSCALE)
def _grayscale(node: TransformNode):
    num_output_channels = node.params.get("num_output_channels", 1)
    return transforms.Grayscale(num_output_channels=num_output_channels)

@register(TransformOpType.RANDOM_GRAYSCALE)
def _random_grayscale(node: TransformNode):
    p = node.params.get("p", 0.1)
    return transforms.RandomGrayscale(p=p)

@register(TransformOpType.TO_TENSOR)
def _to_tensor(node: TransformNode):
    return transforms.ToTensor()

@register(TransformOpType.NORMALIZE)
def _normalize(node: TransformNode):
    # ImageNet 默认均值和方差
    mean = node.params.get("mean", [0.485, 0.456, 0.406])
    std = node.params.get("std", [0.229, 0.224, 0.225])
    return transforms.Normalize(mean=mean, std=std)

@register(TransformOpType.RANDOM_ERASING)
def _random_erasing(node: TransformNode):
    p = node.params.get("p", 0.5)
    scale = tuple(node.params.get("scale", [0.02, 0.33]))
    return transforms.RandomErasing(p=p, scale=scale)

@register(TransformOpType.AUTO_AUGMENT)
def _auto_augment(node: TransformNode):
    policy_name = node.params.get("policy", "IMAGENET")
    policy = transforms.AutoAugmentPolicy[policy_name]
    return transforms.AutoAugment(policy=policy)

@register(TransformOpType.TRIVIAL_AUGMENT)
def _trivial_augment(node: TransformNode):
    return transforms.TrivialAugmentWide()