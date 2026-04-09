"""
Shape Inference 引擎。

流程：
  1. 将 ModelIR 的 nodes / edges 构建为有向图
  2. 拓扑排序（Kahn 算法）
  3. 按拓扑顺序，对每个节点调用对应的 shape rule
  4. 收集结果 / 错误，返回 ShapeInferenceResult
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from core.ir.model_ir import ModelIR
from core.shape_inference.errors import (
    ShapeError,
    ShapeInferenceResult,
)
from core.shape_inference.op_rules import get_rule

class ShapeInferenceEngine:

    def __init__(self, ir: ModelIR):
        self.ir = ir
        # node_id → IRNode
        self.nodes = {n.id: n for n in ir.nodes}
        # node_id → list[predecessor node_id]（注意：有序，保持 edge 声明顺序）
        self.predecessors: dict[str, list[str]] = defaultdict(list)
        # node_id → list[successor node_id]
        self.successors:   dict[str, list[str]] = defaultdict(list)
        # 入度表
        self.in_degree:    dict[str, int]       = {n.id: 0 for n in ir.nodes}

        for edge in ir.edges:
            self.predecessors[edge.target].append(edge.source)
            self.successors[edge.source].append(edge.target)
            self.in_degree[edge.target] += 1

    # ─────────────────────────────────────────
    # 拓扑排序（Kahn 算法）
    # ─────────────────────────────────────────

    def _topological_sort(self) -> list[str]:
        """返回拓扑排序后的 node_id 列表；若有环则抛出 ValueError"""
        queue  = deque(nid for nid, deg in self.in_degree.items() if deg == 0)
        result = []
        degree = dict(self.in_degree)

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for succ in self.successors[nid]:
                degree[succ] -= 1
                if degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self.nodes):
            raise ValueError("ModelIR 中存在环形连接，无法进行 Shape Inference")
        return result

    # ─────────────────────────────────────────
    # 主推导逻辑
    # ─────────────────────────────────────────

    def infer(self) -> ShapeInferenceResult:
        errors:  list[ShapeError]     = []
        shapes:  dict[str, list[int]] = {}   # node_id → output_shape

        try:
            topo_order = self._topological_sort()
        except ValueError as e:
            return ShapeInferenceResult(
                success=False,
                errors=[ShapeError(
                    node_id="*", node_name="*",
                    op_type="*", message=str(e)
                )],
            )

        for node_id in topo_order:
            node = self.nodes[node_id]
            rule = get_rule(node.op_type.value)

            if rule is None:
                errors.append(ShapeError(
                    node_id=node_id,
                    node_name=node.name,
                    op_type=node.op_type.value,
                    message=f"OpType '{node.op_type.value}' 没有注册 Shape Rule",
                ))
                # 继续推导，但此节点 shape 未知，用空列表占位
                shapes[node_id] = []
                continue

            # 收集所有前驱节点的输出 shape（保持 edge 声明顺序）
            input_shapes: list[list[int]] = []
            missing = False
            for pred_id in self.predecessors[node_id]:
                pred_shape = shapes.get(pred_id, [])
                if not pred_shape and pred_id in shapes:
                    # 前驱推导失败，跳过本节点
                    missing = True
                    break
                input_shapes.append(pred_shape)

            if missing:
                shapes[node_id] = []
                continue

            try:
                output_shape = rule(input_shapes, node.params)
                shapes[node_id] = output_shape
            except (ValueError, IndexError, KeyError) as e:
                errors.append(ShapeError(
                    node_id=node_id,
                    node_name=node.name,
                    op_type=node.op_type.value,
                    message=str(e),
                ))
                shapes[node_id] = []

        return ShapeInferenceResult(
            success=len(errors) == 0,
            shapes=shapes,
            errors=errors,
        )

    def infer_and_annotate(self) -> ShapeInferenceResult:
        """
        推导后将 output_shape 写回 ModelIR 的节点，
        返回推导结果（含错误）。
        """
        result = self.infer()
        for node in self.ir.nodes:
            shape = result.shapes.get(node.id)
            if shape:
                node.output_shape = shape
        return result