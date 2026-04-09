"""
Shape Inference 端到端测试。
覆盖：顺序网络、残差连接（Add）、拼接（Concat）、shape 不匹配错误。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ir.model_ir import ModelIR, IRNode, IREdge, OpType
from core.shape_inference import ShapeInferenceEngine

def make_node(id, op, name, params=None):
    return IRNode(id=id, op_type=op, name=name, params=params or {})

def make_edge(src, tgt):
    return IREdge(id=f"e-{src}-{tgt}", source=src, target=tgt)

# ── 测试 1：顺序 CNN ──

def test_sequential_cnn():
    """Input → Conv2d → BN → ReLU → MaxPool → Flatten → Linear → Output"""
    ir = ModelIR(
        id="test-seq", name="SeqCNN", version="1.0.0",
        nodes=[
            make_node("n0", OpType.INPUT,         "input",   {"shape": [3, 32, 32]}),
            make_node("n1", OpType.CONV2D,         "conv1",
                      {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}),
            make_node("n2", OpType.BATCH_NORM2D,   "bn1",    {"num_features": 16}),
            make_node("n3", OpType.RELU,           "relu1",  {}),
            make_node("n4", OpType.MAX_POOL2D,     "pool1",
                      {"kernel_size": 2, "stride": 2}),
            make_node("n5", OpType.FLATTEN,        "flatten",{"start_dim": 1}),
            make_node("n6", OpType.LINEAR,         "fc1",
                      {"in_features": 16 * 16 * 16, "out_features": 10}),
            make_node("n7", OpType.OUTPUT,         "output", {}),
        ],
        edges=[
            make_edge("n0","n1"), make_edge("n1","n2"),
            make_edge("n2","n3"), make_edge("n3","n4"),
            make_edge("n4","n5"), make_edge("n5","n6"),
            make_edge("n6","n7"),
        ],
    )

    engine = ShapeInferenceEngine(ir)
    result = engine.infer_and_annotate()

    assert result.success, f"推导失败: {result.errors}"
    assert result.shapes["n1"] == [16, 32, 32],  f"Conv2d shape 错误: {result.shapes['n1']}"
    assert result.shapes["n4"] == [16, 16, 16],  f"MaxPool shape 错误: {result.shapes['n4']}"
    assert result.shapes["n5"] == [16 * 16 * 16],f"Flatten shape 错误: {result.shapes['n5']}"
    assert result.shapes["n6"] == [10],           f"Linear shape 错误: {result.shapes['n6']}"

    # 验证写回 IR
    conv_node = next(n for n in ir.nodes if n.id == "n1")
    assert conv_node.output_shape == [16, 32, 32]

    print("✅ 顺序 CNN Shape Inference 通过")
    print(f"   各节点 shape: { {nid: s for nid, s in result.shapes.items()} }")

# ── 测试 2：残差连接（Add）──

def test_residual_add():
    """
    n0(Input) → n1(Conv2d) → n2(BN) → n3(ReLU) ──┐
    n0(Input) ──────────────────────────────────── n4(Add) → n5(Output)
    """
    ir = ModelIR(
        id="test-res", name="ResBlock", version="1.0.0",
        nodes=[
            make_node("n0", OpType.INPUT,  "input",  {"shape": [64, 32, 32]}),
            make_node("n1", OpType.CONV2D, "conv1",
                      {"in_channels": 64, "out_channels": 64,
                       "kernel_size": 3, "padding": 1}),
            make_node("n2", OpType.BATCH_NORM2D, "bn1", {}),
            make_node("n3", OpType.RELU,         "relu1", {}),
            make_node("n4", OpType.ADD,          "add",   {}),
            make_node("n5", OpType.OUTPUT,       "output", {}),
        ],
        edges=[
            make_edge("n0","n1"), make_edge("n1","n2"),
            make_edge("n2","n3"), make_edge("n3","n4"),
            make_edge("n0","n4"),   # 残差支路
            make_edge("n4","n5"),
        ],
    )

    result = ShapeInferenceEngine(ir).infer()
    assert result.success, f"残差推导失败: {result.errors}"
    assert result.shapes["n4"] == [64, 32, 32]
    print("✅ 残差连接（Add）Shape Inference 通过")

# ── 测试 3：Concat ──

def test_concat():
    """两路不同通道数的特征拼接"""
    ir = ModelIR(
        id="test-cat", name="ConcatNet", version="1.0.0",
        nodes=[
            make_node("n0", OpType.INPUT,  "input", {"shape": [3, 32, 32]}),
            make_node("n1", OpType.CONV2D, "branch_a",
                      {"in_channels": 3, "out_channels": 32, "kernel_size": 1}),
            make_node("n2", OpType.CONV2D, "branch_b",
                      {"in_channels": 3, "out_channels": 64, "kernel_size": 1}),
            make_node("n3", OpType.CONCAT, "concat", {"dim": 1}),
            make_node("n4", OpType.OUTPUT, "output", {}),
        ],
        edges=[
            make_edge("n0","n1"), make_edge("n0","n2"),
            make_edge("n1","n3"), make_edge("n2","n3"),
            make_edge("n3","n4"),
        ],
    )

    result = ShapeInferenceEngine(ir).infer()
    assert result.success, f"Concat 推导失败: {result.errors}"
    assert result.shapes["n3"] == [96, 32, 32]   # 32 + 64 = 96
    print("✅ Concat Shape Inference 通过")

# ── 测试 4：Linear in_features 不匹配错误 ──

def test_linear_mismatch_error():
    ir = ModelIR(
        id="test-err", name="ErrNet", version="1.0.0",
        nodes=[
            make_node("n0", OpType.INPUT,   "input",   {"shape": [3, 32, 32]}),
            make_node("n1", OpType.FLATTEN, "flatten", {"start_dim": 1}),
            make_node("n2", OpType.LINEAR,  "fc",
                      {"in_features": 999, "out_features": 10}),  # 故意写错
            make_node("n3", OpType.OUTPUT,  "output",  {}),
        ],
        edges=[
            make_edge("n0","n1"), make_edge("n1","n2"), make_edge("n2","n3"),
        ],
    )

    result = ShapeInferenceEngine(ir).infer()
    assert not result.success
    assert any("in_features" in e.message for e in result.errors)
    print(f"✅ Linear 不匹配错误捕获通过: {result.errors[0].message}")

# ── 测试 5：AdaptiveAvgPool + ResNet 尾部 ──

def test_adaptive_pool_to_linear():
    """Conv → AdaptiveAvgPool(1,1) → Flatten → Linear"""
    ir = ModelIR(
        id="test-gap", name="GAPNet", version="1.0.0",
        nodes=[
            make_node("n0", OpType.INPUT,             "input",  {"shape": [3, 224, 224]}),
            make_node("n1", OpType.CONV2D,            "conv1",
                      {"in_channels": 3, "out_channels": 512,
                       "kernel_size": 3, "padding": 1}),
            make_node("n2", OpType.ADAPTIVE_AVG_POOL2D,"gap",   {"output_size": [1, 1]}),
            make_node("n3", OpType.FLATTEN,           "flatten",{"start_dim": 1}),
            make_node("n4", OpType.LINEAR,            "fc",
                      {"in_features": 512, "out_features": 1000}),
            make_node("n5", OpType.OUTPUT,            "output", {}),
        ],
        edges=[
            make_edge("n0","n1"), make_edge("n1","n2"),
            make_edge("n2","n3"), make_edge("n3","n4"),
            make_edge("n4","n5"),
        ],
    )

    result = ShapeInferenceEngine(ir).infer()
    assert result.success, f"GAP 推导失败: {result.errors}"
    assert result.shapes["n2"] == [512, 1, 1]
    assert result.shapes["n3"] == [512]
    assert result.shapes["n4"] == [1000]
    print("✅ AdaptiveAvgPool → Flatten → Linear 通过")

if __name__ == "__main__":
    test_sequential_cnn()
    test_residual_add()
    test_concat()
    test_linear_mismatch_error()
    test_adaptive_pool_to_linear()
    print("\n🎉 所有 Shape Inference 测试通过")