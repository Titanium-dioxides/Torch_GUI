"""
端到端测试：Data IR → DataLoader
使用 CIFAR10（会自动下载）验证完整链路。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.ir.data_ir import (
    DataIR,
    DataSchema,
    RatioSplitConfig,
    TaskType,
    TorchvisionSource,
    TransformNode,
    TransformOpType,
    TransformPipeline,
    DataLoaderConfig,
)
from core.ir.data_builder import DatasetBuilder, build_dataloader

def build_cifar10_data_ir() -> DataIR:
    """构造一个 CIFAR10 的标准 Data IR"""

    # 数据源
    source = TorchvisionSource(dataset_name="CIFAR10", download_root="./data")

    # 数据描述
    schema = DataSchema(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        num_classes=10,
        class_names=[
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        input_channels=3,
    )

    # 切分策略（从 train 中切出 15% 作为 val）
    split = RatioSplitConfig(
        train_ratio=0.85,
        val_ratio=0.15,
        test_ratio=0.0,
        random_seed=42,
    )

    # Train Pipeline：含数据增强
    train_pipeline = TransformPipeline(transforms=[
        TransformNode(id="t1", op_type=TransformOpType.RANDOM_HORIZONTAL_FLIP, params={"p": 0.5}),
        TransformNode(id="t2", op_type=TransformOpType.RANDOM_CROP, params={"size": 32, "padding": 4}),
        TransformNode(id="t3", op_type=TransformOpType.TO_TENSOR),
        TransformNode(id="t4", op_type=TransformOpType.NORMALIZE, params={
            "mean": [0.4914, 0.4822, 0.4465],
            "std":  [0.2470, 0.2435, 0.2616],
        }),
    ])

    # Val Pipeline：只做基础归一化
    val_pipeline = TransformPipeline(transforms=[
        TransformNode(id="v1", op_type=TransformOpType.TO_TENSOR),
        TransformNode(id="v2", op_type=TransformOpType.NORMALIZE, params={
            "mean": [0.4914, 0.4822, 0.4465],
            "std":  [0.2470, 0.2435, 0.2616],
        }),
    ])

    return DataIR(
        id="data-cifar10-001",
        name="CIFAR10Pipeline",
        source=source,
        schema=schema,
        split=split,
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        train_loader_config=DataLoaderConfig(batch_size=64, num_workers=2),
        val_loader_config=DataLoaderConfig(batch_size=128, num_workers=2, drop_last=False),
    )

def test_data_ir():
    ir = build_cifar10_data_ir()

    # 序列化 / 反序列化
    ir_dict = ir.model_dump()
    ir_restored = DataIR.model_validate(ir_dict)
    assert ir_restored.id == ir.id
    print("✅ Data IR 序列化/反序列化通过")

    # 构建 Dataset
    builder = DatasetBuilder(ir_restored)
    bundle = builder.build()
    print(f"✅ Dataset 构建完成: {bundle}")

    # 构建 DataLoader
    loader_bundle = build_dataloader(bundle, ir_restored)
    print(f"✅ DataLoader 构建完成: {loader_bundle}")

    # 取一个 batch 验证 shape
    images, labels = next(iter(loader_bundle.train))
    assert images.shape == (64, 3, 32, 32), f"Batch shape 异常: {images.shape}"
    assert labels.shape == (64,)
    print(f"✅ Train batch shape: images={images.shape}, labels={labels.shape}")

    val_images, val_labels = next(iter(loader_bundle.val))
    print(f"✅ Val batch shape: images={val_images.shape}, labels={val_labels.shape}")

if __name__ == "__main__":
    test_data_ir()