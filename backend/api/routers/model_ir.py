"""
Model IR CRUD + 代码生成接口。
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from api.dependencies import get_store
from api.schemas import ApiResponse, ModelIRSummary
from core.ir.model_ir import ModelIR
from core.ir.codegen import PyTorchCodeGen
from repository.memory_store import MemoryStore

router = APIRouter(prefix="/model-irs", tags=["Model IR"])

@router.post("", response_model=ApiResponse[ModelIRSummary])
async def create_model_ir(
    ir: ModelIR,
    store: MemoryStore = Depends(get_store),
):
    """创建 / 保存一个 Model IR"""
    store.save_model_ir(ir)
    return ApiResponse(data=_to_summary(ir), message="Model IR 保存成功")

@router.get("", response_model=ApiResponse[list[ModelIRSummary]])
async def list_model_irs(store: MemoryStore = Depends(get_store)):
    """列出所有 Model IR"""
    summaries = [_to_summary(ir) for ir in store.list_model_irs()]
    return ApiResponse(data=summaries)

@router.get("/{ir_id}", response_model=ApiResponse[ModelIR])
async def get_model_ir(ir_id: str, store: MemoryStore = Depends(get_store)):
    """获取完整 Model IR"""
    ir = store.get_model_ir(ir_id)
    if not ir:
        raise HTTPException(status_code=404, detail=f"Model IR '{ir_id}' 不存在")
    return ApiResponse(data=ir)

@router.put("/{ir_id}", response_model=ApiResponse[ModelIRSummary])
async def update_model_ir(
    ir_id: str,
    ir: ModelIR,
    store: MemoryStore = Depends(get_store),
):
    """更新 Model IR（全量替换）"""
    if not store.get_model_ir(ir_id):
        raise HTTPException(status_code=404, detail=f"Model IR '{ir_id}' 不存在")
    if ir.id != ir_id:
        raise HTTPException(status_code=400, detail="路径 ID 与 Body ID 不一致")
    store.save_model_ir(ir)
    return ApiResponse(data=_to_summary(ir), message="Model IR 更新成功")

@router.delete("/{ir_id}", response_model=ApiResponse[None])
async def delete_model_ir(ir_id: str, store: MemoryStore = Depends(get_store)):
    """删除 Model IR"""
    if not store.delete_model_ir(ir_id):
        raise HTTPException(status_code=404, detail=f"Model IR '{ir_id}' 不存在")
    return ApiResponse(message="删除成功")

@router.get("/{ir_id}/codegen", response_class=PlainTextResponse)
async def generate_pytorch_code(
    ir_id: str,
    store: MemoryStore = Depends(get_store),
):
    """
    从 Model IR 生成 PyTorch 代码（纯文本返回）。
    前端可直接展示生成的代码。
    """
    ir = store.get_model_ir(ir_id)
    if not ir:
        raise HTTPException(status_code=404, detail=f"Model IR '{ir_id}' 不存在")
    code = PyTorchCodeGen(ir).generate()
    return code

# ── 工具 ──

def _to_summary(ir: ModelIR) -> ModelIRSummary:
    return ModelIRSummary(
        id=ir.id,
        name=ir.name,
        version=ir.version,
        node_count=len(ir.nodes),
        edge_count=len(ir.edges),
    )