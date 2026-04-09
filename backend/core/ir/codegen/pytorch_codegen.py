"""
PyTorch 代码生成器。

输入：ModelIR
输出：合法的 PyTorch Python 代码字符串（包含完整的 nn.Module 类定义）

生成策略：
1. 拓扑排序节点
2. 为每个有参数的节点在 __init__ 生成 self.xxx = nn.XXX(...)
3. 在 forward 中按拓扑顺序生成 x = self.xxx(x)
4. Add / Concat 等多输入节点特殊处理
"""

from __future__ import annotations

from collections import deque
from textwrap import indent

from core.ir.model_ir import IRNode, ModelIR, OpType
from core.ir.codegen.node_registry import get_builder

class PyTorchCodeGen:
    """将 ModelIR 编译为 PyTorch nn.Module 代码"""

    INDENT = "    "  # 4 空格缩进

    def __init__(self, ir: ModelIR):
        self.ir = ir
        self._node_map: dict[str, IRNode] = {n.id: n for n in ir.nodes}
        self._topo_order: list[str] = []   # 拓扑排序后的节点 ID 列表

    # ─────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────

    def generate(self) -> str:
        """生成完整的 Python 代码字符串"""
        self._topo_order = self._topological_sort()
        self._validate_single_input_output()

        class_name = self._to_class_name(self.ir.name)
        init_lines = self._build_init_body()
        forward_lines = self._build_forward_body()

        return self._render_module(class_name, init_lines, forward_lines)

    # ─────────────────────────────────────────
    # 拓扑排序（Kahn 算法）
    # ─────────────────────────────────────────

    def _topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {n.id: 0 for n in self.ir.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.ir.nodes}

        for edge in self.ir.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for successor in adj[nid]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(self.ir.nodes):
            raise ValueError("模型图中存在环，无法进行拓扑排序")

        return result

    # ─────────────────────────────────────────
    # 校验
    # ─────────────────────────────────────────

    def _validate_single_input_output(self) -> None:
        inputs = [n for n in self.ir.nodes if n.op_type == OpType.INPUT]
        outputs = [n for n in self.ir.nodes if n.op_type == OpType.OUTPUT]
        if len(inputs) != 1:
            raise ValueError(f"MVP 阶段只支持单输入节点，当前有 {len(inputs)} 个")
        if len(outputs) != 1:
            raise ValueError(f"MVP 阶段只支持单输出节点，当前有 {len(outputs)} 个")

    # ─────────────────────────────────────────
    # 生成 __init__ 方法体
    # ─────────────────────────────────────────

    def _build_init_body(self) -> list[str]:
        lines: list[str] = ["super().__init__()"]

        for node_id in self._topo_order:
            node = self._node_map[node_id]

            # Input/Output/Add/Concat 不需要 nn.Module 成员
            if node.op_type in (OpType.INPUT, OpType.OUTPUT, OpType.ADD, OpType.CONCAT):
                continue

            builder = get_builder(node.op_type)
            init_code, _ = builder(node)

            if init_code:
                var_name = self._node_to_var(node)
                lines.append(f"self.{var_name} = {init_code}")

        return lines

    # ─────────────────────────────────────────
    # 生成 forward 方法体
    # ─────────────────────────────────────────

    def _build_forward_body(self) -> list[str]:
        lines: list[str] = []

        # 记录每个节点输出对应的变量名（用于多输入节点引用）
        node_output_var: dict[str, str] = {}

        for node_id in self._topo_order:
            node = self._node_map[node_id]
            predecessors = self.ir.get_predecessors(node_id)

            if node.op_type == OpType.INPUT:
                # 输入节点：声明 forward 的参数
                var = "x"
                node_output_var[node_id] = var
                # 不生成代码行，forward 的参数声明在渲染层处理
                continue

            if node.op_type == OpType.OUTPUT:
                # 输出节点：return 前驱节点的输出
                assert len(predecessors) == 1
                prev_var = node_output_var[predecessors[0]]
                lines.append(f"return {prev_var}")
                continue

            if node.op_type == OpType.ADD:
                # 残差相加：所有前驱变量求和
                input_vars = [node_output_var[p] for p in predecessors]
                result_var = f"out_{self._node_to_var(node)}"
                lines.append(f"{result_var} = {' + '.join(input_vars)}")
                node_output_var[node_id] = result_var
                continue

            if node.op_type == OpType.CONCAT:
                # Concat：torch.cat
                input_vars = [node_output_var[p] for p in predecessors]
                dim = node.params.get("dim", 1)
                result_var = f"out_{self._node_to_var(node)}"
                vars_str = ", ".join(input_vars)
                lines.append(f"{result_var} = torch.cat([{vars_str}], dim={dim})")
                node_output_var[node_id] = result_var
                continue

            # 普通单输入节点
            assert len(predecessors) >= 1, f"节点 {node_id} 没有前驱节点"
            input_var = node_output_var[predecessors[0]]
            var_name = self._node_to_var(node)
            result_var = f"out_{var_name}" if self._has_multiple_successors(node_id) else "x"
            lines.append(f"{result_var} = self.{var_name}({input_var})")
            node_output_var[node_id] = result_var

        return lines

    # ─────────────────────────────────────────
    # 渲染最终代码
    # ─────────────────────────────────────────

    def _render_module(
        self,
        class_name: str,
        init_lines: list[str],
        forward_lines: list[str],
    ) -> str:
        ind = self.INDENT

        init_body = indent("\n".join(init_lines), ind * 2)
        forward_body = indent("\n".join(forward_lines), ind * 2)

        code = f"""\
\"\"\"
Auto-generated by NoCode PyTorch Platform
Model: {self.ir.name}  Version: {self.ir.version}
\"\"\"

import torch
import torch.nn as nn

class {class_name}(nn.Module):

{ind}def __init__(self):
{init_body}

{ind}def forward(self, x: torch.Tensor) -> torch.Tensor:
{forward_body}
"""
        return code

    # ─────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────

    def _node_to_var(self, node: IRNode) -> str:
        """将节点 name 转换为合法的 Python 变量名"""
        return node.name.lower().replace(" ", "_").replace("-", "_")

    def _to_class_name(self, name: str) -> str:
        """将模型名转换为 PascalCase 类名"""
        return "".join(word.capitalize() for word in name.replace("-", " ").split())

    def _has_multiple_successors(self, node_id: str) -> bool:
        return len(self.ir.get_successors(node_id)) > 1