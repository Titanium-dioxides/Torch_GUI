"""
Data IR — 数据处理流程的中间表示。

覆盖范围：
- 数据源描述（本地路径 / torchvision 内置数据集）
- 数据 Schema（任务类型、类别）
- 切分策略
- Transform Pipeline（train / val / test 各自独立配置）
- DataLoader 配置
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────
# 枚举定义
# ─────────────────────────────────────────────

class TaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    # 预留后续扩展
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"

class DataSourceType(str, Enum):
    LOCAL_FOLDER = "local_folder"       # ImageFolder 格式的本地目录
    LOCAL_CSV = "local_csv"             # CSV 文件（路径 + 标签）
    TORCHVISION = "torchvision"         # torchvision 内置数据集
    S3 = "s3"                           # 预留：S3/MinIO

class SplitStrategy(str, Enum):
    RATIO = "ratio"             # 按比例随机切分
    PREDEFINED = "predefined"   # 数据源目录中已有 train/val/test 子目录

class TransformOpType(str, Enum):
    # 几何变换
    RESIZE = "Resize"
    CENTER_CROP = "CenterCrop"
    RANDOM_CROP = "RandomCrop"
    RANDOM_HORIZONTAL_FLIP = "RandomHorizontalFlip"
    RANDOM_VERTICAL_FLIP = "RandomVerticalFlip"
    RANDOM_ROTATION = "RandomRotation"
    RANDOM_RESIZED_CROP = "RandomResizedCrop"

    # 色彩变换
    COLOR_JITTER = "ColorJitter"
    GRAYSCALE = "Grayscale"
    RANDOM_GRAYSCALE = "RandomGrayscale"

    # 张量化 & 归一化
    TO_TENSOR = "ToTensor"
    NORMALIZE = "Normalize"

    # 其他
    RANDOM_ERASING = "RandomErasing"
    AUTO_AUGMENT = "AutoAugment"
    TRIVIAL_AUGMENT = "TrivialAugmentWide"

# ─────────────────────────────────────────────
# 数据源定义
# ─────────────────────────────────────────────

class LocalFolderSource(BaseModel):
    """
    本地 ImageFolder 格式：
    root/
      class_a/img1.jpg
      class_b/img2.jpg
    """
    type: Literal[DataSourceType.LOCAL_FOLDER] = DataSourceType.LOCAL_FOLDER
    root_path: str = Field(..., description="数据集根目录绝对路径")

class LocalCSVSource(BaseModel):
    """
    CSV 格式：每行包含 image_path 和 label 两列
    """
    type: Literal[DataSourceType.LOCAL_CSV] = DataSourceType.LOCAL_CSV
    csv_path: str = Field(..., description="CSV 文件路径")
    image_col: str = Field(default="image_path", description="图片路径列名")
    label_col: str = Field(default="label", description="标签列名")
    image_root: str = Field(default="", description="图片路径的根目录前缀")

class TorchvisionSource(BaseModel):
    """
    torchvision 内置数据集，如 CIFAR10、MNIST 等
    """
    type: Literal[DataSourceType.TORCHVISION] = DataSourceType.TORCHVISION
    dataset_name: str = Field(
        ...,
        description="torchvision.datasets 中的类名，如 'CIFAR10'、'MNIST'"
    )
    download_root: str = Field(default="./data", description="数据集下载/缓存路径")
    download: bool = Field(default=True)

class S3Source(BaseModel):
    """预留：S3/MinIO 数据源"""
    type: Literal[DataSourceType.S3] = DataSourceType.S3
    bucket: str
    prefix: str
    endpoint_url: str = ""

# 联合类型，Pydantic v2 用 Annotated + discriminator 处理
DataSource = LocalFolderSource | LocalCSVSource | TorchvisionSource | S3Source

# ─────────────────────────────────────────────
# 数据 Schema
# ─────────────────────────────────────────────

class DataSchema(BaseModel):
    """描述数据集的任务类型和类别信息"""
    task_type: TaskType = TaskType.IMAGE_CLASSIFICATION
    num_classes: int = Field(..., ge=2, description="类别数量")
    class_names: list[str] = Field(
        default_factory=list,
        description="类别名称列表，顺序对应 label index"
    )
    input_channels: int = Field(default=3, description="输入图片通道数，RGB=3，灰度=1")
    input_shape: list[int] = Field(default_factory=lambda: [3, 224, 224], description="输入图片形状 [C, H, W]")

    @model_validator(mode="after")
    def validate_class_names(self) -> "DataSchema":
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names 长度 ({len(self.class_names)}) "
                f"与 num_classes ({self.num_classes}) 不一致"
            )
        return self

# ─────────────────────────────────────────────
# 切分策略
# ─────────────────────────────────────────────

class RatioSplitConfig(BaseModel):
    """按比例随机切分"""
    strategy: Literal[SplitStrategy.RATIO] = SplitStrategy.RATIO
    train_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    random_seed: int = 42

    @model_validator(mode="after")
    def check_ratio_sum(self) -> "RatioSplitConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {total:.4f}")
        return self

class PredefinedSplitConfig(BaseModel):
    """数据源已包含 train/val/test 子目录"""
    strategy: Literal[SplitStrategy.PREDEFINED] = SplitStrategy.PREDEFINED
    train_dir: str = Field(default="train")
    val_dir: str = Field(default="val")
    test_dir: str = Field(default="test", description="可为空，表示不使用测试集")

SplitConfig = RatioSplitConfig | PredefinedSplitConfig

# ─────────────────────────────────────────────
# Transform 节点
# ─────────────────────────────────────────────

class TransformNode(BaseModel):
    """
    单个数据变换操作。
    params 字段存储该 transform 的参数，由 transform_registry 解析。
    """
    id: str = Field(..., description="节点唯一 ID")
    op_type: TransformOpType
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True, description="可在前端快速开关某个 transform")

class TransformPipeline(BaseModel):
    """
    一个有序的 Transform 列表，对应 torchvision.transforms.Compose。
    train/val/test 各自独立配置，通常 val/test 不做随机增强。
    """
    transforms: list[TransformNode] = Field(default_factory=list)

    def get_enabled(self) -> list[TransformNode]:
        """只返回 enabled=True 的 transform"""
        return [t for t in self.transforms if t.enabled]

# ─────────────────────────────────────────────
# DataLoader 配置
# ─────────────────────────────────────────────

class DataLoaderConfig(BaseModel):
    """对应 torch.utils.data.DataLoader 的核心参数"""
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True, description="GPU 训练时建议开启")
    drop_last: bool = Field(default=False, description="丢弃最后不满一个 batch 的数据")
    prefetch_factor: int | None = Field(
        default=2,
        description="每个 worker 预取的 batch 数，num_workers=0 时必须为 None"
    )

    @model_validator(mode="after")
    def fix_prefetch(self) -> "DataLoaderConfig":
        if self.num_workers == 0:
            self.prefetch_factor = None
        return self

# ─────────────────────────────────────────────
# 顶层 Data IR
# ─────────────────────────────────────────────

class DataIR(BaseModel):
    """
    完整的数据处理流程 IR。
    前端画布中的数据配置面板序列化后得到此结构。
    """
    id: str = Field(..., description="Data pipeline 唯一 ID")
    name: str = Field(default="MyDataset")
    version: str = Field(default="1.0.0")

    # 数据来源
    source: DataSource = Field(..., discriminator="type")

    # 数据描述
    schema: DataSchema

    # 切分策略
    split: SplitConfig = Field(
        default_factory=lambda: RatioSplitConfig(),
        discriminator="strategy"
    )

    # Transform Pipelines（train/val/test 独立配置）
    train_pipeline: TransformPipeline = Field(default_factory=TransformPipeline)
    val_pipeline: TransformPipeline = Field(default_factory=TransformPipeline)
    test_pipeline: TransformPipeline = Field(default_factory=TransformPipeline)

    # DataLoader 配置（train/val 可分别指定）
    train_loader_config: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    val_loader_config: DataLoaderConfig = Field(
        default_factory=lambda: DataLoaderConfig(batch_size=32, drop_last=False)
    )

    description: str = ""
    tags: list[str] = Field(default_factory=list)