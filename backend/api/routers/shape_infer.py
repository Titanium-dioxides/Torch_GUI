"""
Shape Inference REST 接口。

POST /api/v1/model-irs/{ir_id}/shape-infer
    → 对已存储的 ModelIR 做 Shape Inference，返回每个节点的 shape

POST /api/v1/shape-infer/preview
    → 对请求 Body 中的 ModelIR（未保存）做 Shape Inference（前端实时预览用）
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_store
from api.schemas import ApiResponse
from core.ir.model_ir import ModelIR
from core.shape_inference import ShapeInferenceEngine, ShapeInferenceResult
from repository.memory_store import MemoryStore

router = APIRouter(tags=["Shape Inference"])

# ── 响应 Schema ──

class NodeShapeInfo(BaseModel):
    node_id:      str
    node_name:    str
    op_type:      str
    output_shape: list[int]

class ShapeInferResponse(BaseModel):
    success:     bool
    node_shapes: list[NodeShapeInfo]
    errors:      list[dict]

# ── 对已存储的 ModelIR 推导 ──

@router.post(
    "/model-irs/{ir_id}/shape-infer",
    response_model=ApiResponse[ShapeInferResponse],
)
async def infer_shapes_for_stored(
    ir_id: str,
    store: MemoryStore = Depends(get_store),
):
    """
    对 store 中已保存的 ModelIR 做 Shape Inference，
    并将结果写回节点的 output_shape 字段。
    """
    ir = store.get_model_ir(ir_id)
    if not ir:
        raise HTTPException(status_code=404, detail=f"Model IR '{ir_id}' 不存在")

    result = _run_inference(ir)

    # 写回 store
    store.save_model_ir(ir)

    return ApiResponse(
        data=_to_response(ir, result),
        message="Shape Inference 完成" if result.success else "Shape Inference 存在错误",
    )

# ── 实时预览（不保存）──

@router.post(
    "/shape-infer/preview",
    response_model=ApiResponse[ShapeInferResponse],
)
async def preview_shapes(ir: ModelIR):
    """
    前端实时调用：传入当前画布的 ModelIR，
    返回每个节点的推导 shape（不修改 store）。
    """
    result = _run_inference(ir)
    return ApiResponse(
        data=_to_response(ir, result),
        message="Shape Inference 预览完成",
    )

# ── 工具 ──

def _run_inference(ir: ModelIR) -> ShapeInferenceResult:
    engine = ShapeInferenceEngine(ir)
    return engine.infer_and_annotate()

def _to_response(ir: ModelIR, result: ShapeInferenceResult) -> ShapeInferResponse:
    node_shapes = [
        NodeShapeInfo(
            node_id=node.id,
            node_name=node.name,
            op_type=node.op_type.value,
            output_shape=result.shapes.get(node.id, []),
        )
        for node in ir.nodes
    ]
    errors = [
        {
            "node_id":   e.node_id,
            "node_name": e.node_name,
            "op_type":   e.op_type,
            "message":   e.message,
        }
        for e in result.errors
    ]
    return ShapeInferResponse(
        success=result.success,
        node_shapes=node_shapes,
        errors=errors,
    )