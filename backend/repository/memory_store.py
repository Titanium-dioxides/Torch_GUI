"""
内存存储层（MVP）。
生产阶段替换为 SQLAlchemy Repository，接口保持一致。
"""

from __future__ import annotations

from core.ir.model_ir import ModelIR
from core.ir.data_ir import DataIR
from core.ir.experiment_ir import ExperimentIR, ExperimentRun

class MemoryStore:
    """单例内存存储，整个应用共享同一实例"""

    _instance: "MemoryStore | None" = None

    def __new__(cls) -> "MemoryStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_stores()
        return cls._instance

    def _init_stores(self) -> None:
        self.model_irs:    dict[str, ModelIR]       = {}
        self.data_irs:     dict[str, DataIR]        = {}
        self.experiments:  dict[str, ExperimentIR]  = {}
        self.runs:         dict[str, ExperimentRun] = {}

    # ── Model IR ──
    def save_model_ir(self, ir: ModelIR) -> None:
        self.model_irs[ir.id] = ir

    def get_model_ir(self, ir_id: str) -> ModelIR | None:
        return self.model_irs.get(ir_id)

    def list_model_irs(self) -> list[ModelIR]:
        return list(self.model_irs.values())

    def delete_model_ir(self, ir_id: str) -> bool:
        return self.model_irs.pop(ir_id, None) is not None

    # ── Data IR ──
    def save_data_ir(self, ir: DataIR) -> None:
        self.data_irs[ir.id] = ir

    def get_data_ir(self, ir_id: str) -> DataIR | None:
        return self.data_irs.get(ir_id)

    def list_data_irs(self) -> list[DataIR]:
        return list(self.data_irs.values())

    def delete_data_ir(self, ir_id: str) -> bool:
        return self.data_irs.pop(ir_id, None) is not None

    # ── Experiment IR ──
    def save_experiment(self, ir: ExperimentIR) -> None:
        self.experiments[ir.id] = ir

    def get_experiment(self, exp_id: str) -> ExperimentIR | None:
        return self.experiments.get(exp_id)

    def list_experiments(self) -> list[ExperimentIR]:
        return list(self.experiments.values())

    def delete_experiment(self, exp_id: str) -> bool:
        return self.experiments.pop(exp_id, None) is not None

    # ── Run 状态 ──
    def save_run(self, run: ExperimentRun) -> None:
        self.runs[run.experiment_id] = run

    def get_run(self, experiment_id: str) -> ExperimentRun | None:
        return self.runs.get(experiment_id)