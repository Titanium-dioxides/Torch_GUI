"""
DataLoader 构建器：将 DatasetBundle + DataIR 配置
组装为可直接用于训练的 DataLoader 字典。
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from core.ir.data_ir import DataIR, DataLoaderConfig
from core.ir.data_builder.dataset_builder import DatasetBundle

class DataLoaderBundle:
    """包含 train/val/test DataLoader 的容器"""

    def __init__(
        self,
        train: DataLoader,
        val: DataLoader,
        test: DataLoader | None = None,
    ):
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self) -> str:
        return (
            f"DataLoaderBundle("
            f"train_batches={len(self.train)}, "
            f"val_batches={len(self.val)}, "
            f"test_batches={len(self.test) if self.test else 0})"
        )

def build_dataloader(
    dataset_bundle: DatasetBundle,
    ir: DataIR,
) -> DataLoaderBundle:
    """
    根据 DataIR 中的 DataLoaderConfig 构建三组 DataLoader。
    """
    train_loader = _make_loader(
        dataset_bundle.train,
        ir.train_loader_config,
        shuffle=True,
    )
    val_loader = _make_loader(
        dataset_bundle.val,
        ir.val_loader_config,
        shuffle=False,
    )
    test_loader = None
    if dataset_bundle.test is not None:
        test_loader = _make_loader(
            dataset_bundle.test,
            ir.val_loader_config,   # test 复用 val 的 loader 配置
            shuffle=False,
        )

    return DataLoaderBundle(train=train_loader, val=val_loader, test=test_loader)

def _make_loader(
    dataset,
    config: DataLoaderConfig,
    shuffle: bool,
) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    if config.prefetch_factor is not None and config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor

    return DataLoader(**kwargs)