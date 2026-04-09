"""
Dataset 构建器：根据 DataIR 的 source 和 split 配置，
构建 train / val / test 三个 torch.utils.data.Dataset。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

from core.ir.data_ir import (
    DataIR,
    DataSourceType,
    LocalFolderSource,
    PredefinedSplitConfig,
    RatioSplitConfig,
    SplitStrategy,
    TorchvisionSource,
    TransformPipeline,
)
from core.ir.data_builder.transform_registry import build_pipeline

class DatasetBundle:
    """包含 train/val/test 三个 Dataset 的容器"""

    def __init__(
        self,
        train: Dataset,
        val: Dataset,
        test: Dataset | None = None,
    ):
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self) -> str:
        test_len = len(self.test) if self.test else 0  # type: ignore
        return (
            f"DatasetBundle("
            f"train={len(self.train)}, "  # type: ignore
            f"val={len(self.val)}, "      # type: ignore
            f"test={test_len})"
        )

class DatasetBuilder:
    """根据 DataIR 构建 DatasetBundle"""

    def __init__(self, ir: DataIR):
        self.ir = ir

    def build(self) -> DatasetBundle:
        source = self.ir.source

        if source.type == DataSourceType.TORCHVISION:
            return self._build_torchvision(source)  # type: ignore
        elif source.type == DataSourceType.LOCAL_FOLDER:
            return self._build_local_folder(source)  # type: ignore
        else:
            raise NotImplementedError(
                f"数据源类型 '{source.type}' 的 Dataset 构建尚未实现"
            )

    # ─────────────────────────────────────────
    # torchvision 内置数据集
    # ─────────────────────────────────────────

    def _build_torchvision(self, source: TorchvisionSource) -> DatasetBundle:
        """
        支持 CIFAR10、CIFAR100、MNIST、FashionMNIST 等。
        torchvision 数据集通常自带 train/test 划分，
        我们从 train 中再切出 val。
        """
        dataset_cls = getattr(datasets, source.dataset_name, None)
        if dataset_cls is None:
            raise ValueError(
                f"torchvision.datasets 中不存在 '{source.dataset_name}'"
            )

        train_transform = build_pipeline(self.ir.train_pipeline.get_enabled())
        val_transform = build_pipeline(self.ir.val_pipeline.get_enabled())

        # 获取原始 train 集（无 transform，用于后续切分）
        full_train = dataset_cls(
            root=source.download_root,
            train=True,
            download=source.download,
            transform=None,
        )

        test_dataset = dataset_cls(
            root=source.download_root,
            train=False,
            download=source.download,
            transform=val_transform,
        )

        # 从 train 中切出 val
        split_cfg = self.ir.split
        if isinstance(split_cfg, RatioSplitConfig):
            val_ratio = split_cfg.val_ratio / (split_cfg.train_ratio + split_cfg.val_ratio)
            val_size = int(len(full_train) * val_ratio)
            train_size = len(full_train) - val_size

            generator = torch.Generator().manual_seed(split_cfg.random_seed)
            train_subset, val_subset = random_split(
                full_train, [train_size, val_size], generator=generator
            )

            # 分别包装 transform
            train_dataset = _TransformSubset(train_subset, train_transform)
            val_dataset = _TransformSubset(val_subset, val_transform)
        else:
            # torchvision 数据集不适合 predefined split，直接使用全量 train
            full_train.transform = train_transform
            train_dataset = full_train
            val_dataset = test_dataset

        return DatasetBundle(train=train_dataset, val=val_dataset, test=test_dataset)

    # ─────────────────────────────────────────
    # 本地 ImageFolder 格式
    # ─────────────────────────────────────────

    def _build_local_folder(self, source: LocalFolderSource) -> DatasetBundle:
        split_cfg = self.ir.split

        train_transform = build_pipeline(self.ir.train_pipeline.get_enabled())
        val_transform = build_pipeline(self.ir.val_pipeline.get_enabled())
        test_transform = build_pipeline(self.ir.test_pipeline.get_enabled())

        if isinstance(split_cfg, PredefinedSplitConfig):
            # 目录中已有 train/val/test 子目录
            train_path = os.path.join(source.root_path, split_cfg.train_dir)
            val_path = os.path.join(source.root_path, split_cfg.val_dir)

            train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
            val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

            test_dataset = None
            if split_cfg.test_dir:
                test_path = os.path.join(source.root_path, split_cfg.test_dir)
                if os.path.isdir(test_path):
                    test_dataset = datasets.ImageFolder(
                        test_path, transform=test_transform
                    )

            return DatasetBundle(
                train=train_dataset, val=val_dataset, test=test_dataset
            )

        elif isinstance(split_cfg, RatioSplitConfig):
            # 从根目录按比例切分
            full_dataset = datasets.ImageFolder(source.root_path, transform=None)
            total = len(full_dataset)

            train_size = int(total * split_cfg.train_ratio)
            val_size = int(total * split_cfg.val_ratio)
            test_size = total - train_size - val_size

            generator = torch.Generator().manual_seed(split_cfg.random_seed)
            train_sub, val_sub, test_sub = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator,
            )

            return DatasetBundle(
                train=_TransformSubset(train_sub, train_transform),
                val=_TransformSubset(val_sub, val_transform),
                test=_TransformSubset(test_sub, test_transform) if test_size > 0 else None,
            )

        else:
            raise NotImplementedError(f"未知的切分策略: {type(split_cfg)}")

# ─────────────────────────────────────────────
# 辅助：为 Subset 动态附加 transform
# ─────────────────────────────────────────────

class _TransformSubset(Dataset):
    """
    包装 Subset，在取数据时动态应用 transform。
    解决 random_split 后无法独立设置 train/val transform 的问题。
    """

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label