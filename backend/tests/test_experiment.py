"""
端到端测试：Experiment IR → Trainer（不启动 Celery，直接本地运行）
使用 CIFAR10 跑 2 个 epoch 验证完整链路。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ir.model_ir import IREdge, IRNode, ModelIR, OpType
from core.ir.data_ir import (
    DataIR, DataSchema, RatioSplitConfig, TaskType,
    TorchvisionSource, TransformNode, TransformOpType,
    TransformPipeline, DataLoaderConfig,
)
from core.ir.experiment_ir import (
    ExperimentIR, TrainHyperParams, OptimizerConfig, OptimizerType,
    SchedulerConfig, SchedulerType, CheckpointConfig, LoggingConfig,
    LocalBackendConfig,
)
from core.ir.data_builder import DatasetBuilder, build_dataloader
from core.ir.codegen import PyTorchCodeGen
from training.trainer import Trainer
import torch.nn as nn, torch

def build_small_cnn_ir() -> ModelIR:
    nodes = [
        IRNode(id="n0", op_type=OpType.INPUT,   name="input",   params={"shape": [3, 32, 32]}),
        IRNode(id="n1", op_type=OpType.CONV2D,  name="conv1",   params={"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}),
        IRNode(id="n2", op_type=OpType.BATCH_NORM2D, name="bn1", params={"num_features": 32}),
        IRNode(id="n3", op_type=OpType.RELU,    name="relu1",   params={}),
        IRNode(id="n4", op_type=OpType.MAX_POOL2D, name="pool1", params={"kernel_size": 2, "stride": 2}),
        IRNode(id="n5", op_type=OpType.ADAPTIVE_AVG_POOL2D, name="gap", params={"output_size": (1, 1)}),
        IRNode(id="n6", op_type=OpType.FLATTEN, name="flatten", params={}),
        IRNode(id="n7", op_type=OpType.LINEAR,  name="fc1",     params={"in_features": 32, "out_features": 10}),
        IRNode(id="n8", op_type=OpType.OUTPUT,  name="output",  params={}),
    ]
    edges = [IREdge(id=f"e{i}{i+1}", source=f"n{i}", target=f"n{i+1}") for i in range(8)]
    return ModelIR(id="model-test-001", name="SmallCNN", nodes=nodes, edges=edges)

def build_cifar10_data_ir() -> DataIR:
    return DataIR(
        id="data-test-001",
        name="CIFAR10Test",
        source=TorchvisionSource(dataset_name="CIFAR10", download_root="./data"),
        schema=DataSchema(task_type=TaskType.IMAGE_CLASSIFICATION, num_classes=10),
        split=RatioSplitConfig(train_ratio=0.85, val_ratio=0.15, test_ratio=0.0),
        train_pipeline=TransformPipeline(transforms=[
            TransformNode(id="t1", op_type=TransformOpType.TO_TENSOR),
            TransformNode(id="t2", op_type=TransformOpType.NORMALIZE,
                          params={"mean": [0.4914, 0.4822, 0.4465],
                                  "std":  [0.2470, 0.2435, 0.2616]}),
        ]),
        val_pipeline=TransformPipeline(transforms=[
            TransformNode(id="v1", op_type=TransformOpType.TO_TENSOR),
            TransformNode(id="v2", op_type=TransformOpType.NORMALIZE,
                          params={"mean": [0.4914, 0.4822, 0.4465],
                                  "std":  [0.2470, 0.2435, 0.2616]}),
        ]),
        train_loader_config=DataLoaderConfig(batch_size=128, num_workers=0),
        val_loader_config=DataLoaderConfig(batch_size=256, num_workers=0),
    )

def build_experiment_ir(model_ir_id: str, data_ir_id: str) -> ExperimentIR:
    return ExperimentIR(
        id="exp-test-001",
        name="SmallCNN-CIFAR10-Test",
        model_ir_id=model_ir_id,
        data_ir_id=data_ir_id,
        hyper_params=TrainHyperParams(
            epochs=2,
            optimizer=OptimizerConfig(type=OptimizerType.ADAMW, lr=1e-3),
            scheduler=SchedulerConfig(type=SchedulerType.COSINE_ANNEALING, t_max=2),
            use_amp=False,   # 测试环境关闭 AMP
        ),
        backend=LocalBackendConfig(),
        checkpoint=CheckpointConfig(enabled=False),   # 测试不保存 checkpoint
    )

def test_full_pipeline():
    model_ir = build_small_cnn_ir()
    data_ir  = build_cifar10_data_ir()
    exp_ir   = build_experiment_ir(model_ir.id, data_ir.id)

    # 序列化验证
    assert ExperimentIR.model_validate(exp_ir.model_dump()).id == exp_ir.id
    print("✅ Experiment IR 序列化/反序列化通过")

    # 构建模型
    code = PyTorchCodeGen(model_ir).generate()
    ns: dict = {}
    exec(code, ns)
    model_cls = next(v for v in ns.values()
                     if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module)
    model = model_cls()
    print(f"✅ 模型构建完成: {model.__class__.__name__}")

    # 构建 DataLoader
    bundle  = DatasetBuilder(data_ir).build()
    loaders = build_dataloader(bundle, data_ir)
    print(f"✅ DataLoader 构建完成: {loaders}")

    # 训练
    logs: list = []
    trainer = Trainer(exp_ir, model, loaders.train, loaders.val)
    trainer.on_epoch_end = lambda m: logs.append(m) or print(
        f"  Epoch {m.epoch} | train_loss={m.train_loss:.4f} "
        f"val_acc={m.val_acc:.4f} lr={m.lr:.2e}"
    )
    history = trainer.fit()

    assert len(history) == 2
    print(f"✅ 训练完成，共 {len(history)} 个 epoch")
    print(f"   最终 val_acc = {history[-1].val_acc:.4f}")

if __name__ == "__main__":
    test_full_pipeline()