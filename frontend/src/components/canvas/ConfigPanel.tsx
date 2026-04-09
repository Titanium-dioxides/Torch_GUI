import { useCanvasStore } from "@/store/canvasStore";
import { getNodeMeta } from "./nodeRegistry";

export function ConfigPanel() {
  const { nodes, selectedNodeId, updateNodeParams, updateNodeName } =
    useCanvasStore();
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  if (!selectedNode) {
    return (
      <div
        style={{
          width: 240,
          background: "#0f172a",
          borderLeft: "1px solid #1e293b",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#475569",
          fontSize: 13,
        }}
      >
        点击节点编辑参数
      </div>
    );
  }

  const { irNode } = selectedNode.data;
  const meta = getNodeMeta(irNode.op_type);

  const handleParamChange = (key: string, rawValue: string) => {
    const current = irNode.params[key];
    // 类型推导：跟随当前参数类型
    let value: unknown = rawValue;
    if (typeof current === "number") {
      value = rawValue === "" ? 0 : Number(rawValue);
    } else if (typeof current === "boolean") {
      value = rawValue === "true";
    } else if (Array.isArray(current)) {
      try { value = JSON.parse(rawValue); } catch { value = rawValue; }
    }
    updateNodeParams(irNode.id, { ...irNode.params, [key]: value });
  };

  return (
    <div
      style={{
        width: 240,
        height: "100%",
        background: "#0f172a",
        borderLeft: "1px solid #1e293b",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 14px",
          borderBottom: "1px solid #1e293b",
          background: meta?.color ?? "#334155",
        }}
      >
        <div style={{ color: "#fff", fontWeight: 700, fontSize: 14 }}>
          {meta?.label}
        </div>
        <div style={{ color: "rgba(255,255,255,0.7)", fontSize: 11, marginTop: 2 }}>
          {meta?.description}
        </div>
      </div>

      {/* 层名称 */}
      <div style={{ padding: "12px 14px", borderBottom: "1px solid #1e293b" }}>
        <label style={labelStyle}>层名称</label>
        <input
          value={irNode.name}
          onChange={(e) => updateNodeName(irNode.id, e.target.value)}
          style={inputStyle}
        />
      </div>

      {/* 参数列表 */}
      <div style={{ padding: "12px 14px", flex: 1 }}>
        <div style={{ color: "#475569", fontSize: 10, fontWeight: 700,
          textTransform: "uppercase", marginBottom: 10 }}>
          参数配置
        </div>
        {Object.entries(irNode.params).map(([key, value]) => (
          <div key={key} style={{ marginBottom: 12 }}>
            <label style={labelStyle}>{key}</label>
            {typeof value === "boolean" ? (
              <select
                value={String(value)}
                onChange={(e) => handleParamChange(key, e.target.value)}
                style={inputStyle}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                value={Array.isArray(value) ? JSON.stringify(value) : String(value)}
                onChange={(e) => handleParamChange(key, e.target.value)}
                style={inputStyle}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block",
  color: "#94a3b8",
  fontSize: 11,
  marginBottom: 4,
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 6,
  color: "#e2e8f0",
  padding: "6px 8px",
  fontSize: 12,
  outline: "none",
  boxSizing: "border-box",
};