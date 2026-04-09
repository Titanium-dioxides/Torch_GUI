"""
Data IR CRUD 接口。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_store
from api.schemas import ApiResponse, DataIRSummary
from core.ir.data_ir import DataIR, RatioSplitConfig
from repository.memory_store import MemoryStore

router = APIRouter(prefix="/data-irs", tags=["Data IR"])

# ── 预览响应 Schema ──

class DataPreviewInfo(BaseModel):
    """数据集预览信息"""
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    num_classes: int
    class_names: list[str]
    sample_shape: list[int]

# ── CRUD 接口 ──

@router.post("/preview", response_model=ApiResponse[DataPreviewInfo])
async def preview_data_ir(ir: DataIR):
    """
    预览 Data IR 的数据集信息（不保存到 store）。
    前端实时预览用。
    """
    # 计算样本数量
    total_samples = 10000  # 默认值，实际应该从数据源获取
    
    # 根据 split 类型计算样本数量
    if ir.split.strategy == "ratio":
        # RatioSplitConfig
        split_config = ir.split
        train_ratio = split_config.train_ratio  # type: ignore
        val_ratio = split_config.val_ratio  # type: ignore
        test_ratio = split_config.test_ratio  # type: ignore
    else:
        # PredefinedSplitConfig - 使用默认比例
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
    
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples
    
    # 获取类别信息
    num_classes = ir.schema.num_classes
    class_names = ir.schema.class_names if ir.schema.class_names else [f"class_{i}" for i in range(num_classes)]
    
    # 获取样本形状
    sample_shape = ir.schema.input_shape if hasattr(ir.schema, 'input_shape') else [3, 224, 224]
    
    return ApiResponse(
        data=DataPreviewInfo(
            total_samples=total_samples,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            num_classes=num_classes,
            class_names=class_names,
            sample_shape=sample_shape,
        ),
        message="数据预览生成成功",
    )

@router.post("", response_model=ApiResponse[DataIRSummary])
async def create_data_ir(ir: DataIR, store: MemoryStore = Depends(get_store)):
    store.save_data_ir(ir)
    return ApiResponse(data=_to_summary(ir), message="Data IR 保存成功")

@router.get("", response_model=ApiResponse[list[DataIRSummary]])
async def list_data_irs(store: MemoryStore = Depends(get_store)):
    return ApiResponse(data=[_to_summary(ir) for ir in store.list_data_irs()])

@router.get("/{ir_id}", response_model=ApiResponse[DataIR])
async def get_data_ir(ir_id: str, store: MemoryStore = Depends(get_store)):
    ir = store.get_data_ir(ir_id)
    if not ir:
        raise HTTPException(status_code=404, detail=f"Data IR '{ir_id}' 不存在")
    return ApiResponse(data=ir)

@router.put("/{ir_id}", response_model=ApiResponse[DataIRSummary])
async def update_data_ir(
    ir_id: str,
    ir: DataIR,
    store: MemoryStore = Depends(get_store),
):
    if not store.get_data_ir(ir_id):
        raise HTTPException(status_code=404, detail=f"Data IR '{ir_id}' 不存在")
    if ir.id != ir_id:
        raise HTTPException(status_code=400, detail="路径 ID 与 Body ID 不一致")
    store.save_data_ir(ir)
    return ApiResponse(data=_to_summary(ir), message="Data IR 更新成功")

@router.delete("/{ir_id}", response_model=ApiResponse[None])
async def delete_data_ir(ir_id: str, store: MemoryStore = Depends(get_store)):
    if not store.delete_data_ir(ir_id):
        raise HTTPException(status_code=404, detail=f"Data IR '{ir_id}' 不存在")
    return ApiResponse(message="删除成功")

def _to_summary(ir: DataIR) -> DataIRSummary:
    return DataIRSummary(
        id=ir.id,
        name=ir.name,
        version=ir.version,
        source_type=ir.source.type.value,
        num_classes=ir.schema.num_classes,
    )