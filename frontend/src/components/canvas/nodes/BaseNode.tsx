import { memo } from "react";
import { Handle, NodeProps, Position } from "reactflow";
import { useCanvasStore, NodeData } from "@/store/canvasStore";
import { getNodeMeta } from "../nodeRegistry";

export const BaseNode = memo(({ id, data }: NodeProps<NodeData>) => {
  const selectNode = useCanvasStore((s) => s.selectNode);
  const deleteNode = useCanvasStore((s) => s.deleteNode);
  const meta = getNodeMeta(data.irNode.op_type);

  const borderColor = data.selected ? "#f59e0b" : "transparent";
  const bgColor = meta?.color ?? "#64748b";
  const isInput  = data.irNode.op_type === "Input";
  const isOutput = data.irNode.op_type === "Output";

  return (
    <div
      onClick={() => selectNode(id)}
      style={{
        border: `2px solid ${borderColor}`,
        borderRadius: 8,
        background: "#1e293b",
        minWidth: 140,
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      {/* 顶部色条 */}
      <div
        style={{
          background: bgColor,
          borderRadius: "6px 6px 0 0",
          padding: "4px 8px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{ color: "#fff", fontSize: 11, fontWeight: 600 }}>
          {meta?.label ?? data.irNode.op_type}
        </span>
        <button
          onClick={(e) => { e.stopPropagation(); deleteNode(id); }}
          style={{
            background: "transparent",
            border: "none",
            color: "rgba(255,255,255,0.7)",
            cursor: "pointer",
            fontSize: 12,
            padding: "0 2px",
          }}
        >
          ✕
        </button>
      </div>

      {/* 节点主体 */}
      <div style={{ padding: "6px 10px" }}>
        <div style={{ color: "#94a3b8", fontSize: 11 }}>
          {data.irNode.name}
        </div>
        {/* 显示 output_shape */}
        {data.irNode.output_shape && (
          <div style={{ color: "#64748b", fontSize: 10, marginTop: 2 }}>
            [{data.irNode.output_shape.join(", ")}]
          </div>
        )}
      </div>

      {/* 连接点 */}
      {!isInput && (
        <Handle
          type="target"
          position={Position.Top}
          style={{ background: bgColor, width: 8, height: 8 }}
        />
      )}
      {!isOutput && (
        <Handle
          type="source"
          position={Position.Bottom}
          style={{ background: bgColor, width: 8, height: 8 }}
        />
      )}
    </div>
  );
});