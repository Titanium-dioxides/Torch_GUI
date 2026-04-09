
import { useState } from "react";
import { useCanvasStore } from "@/store/canvasStore";
import { modelIrApi } from "@/api/modelIr";
import { experimentApi } from "@/api/experiment";
import { FlowCanvas } from "@/components/canvas/FlowCanvas";
import { NodePanel } from "@/components/canvas/NodePanel";
import { ConfigPanel } from "@/components/canvas/ConfigPanel";
import { TrainingMonitor } from "@/components/monitor/TrainingMonitor";
import { useShapeInference } from "@/hooks/useShapeInference";
import { DataPanel } from "@/components/data/DataPanel";

export default function App() {
  const [dataOpen, setDataOpen] = useState(false);
  useShapeInference(); 
  const { toModelIR, modelName, setModelName } = useCanvasStore();
  const [saving,   setSaving]   = useState(false);
  const [training, setTraining] = useState(false);
  const [activeExpId, setActiveExpId] = useState<string | null>(null);
  const [showMonitor, setShowMonitor] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
    
  // ── 保存 Model IR ──
  const handleSave = async () => {
    setSaving(true);
    setStatusMsg("");
    try {
      const ir = toModelIR();
      await modelIrApi.save(ir);
      setStatusMsg("✅ 模型已保存");
    } catch (e: any) {
      setStatusMsg(`❌ ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  // ── 提交训练（简化：直接内联创建 Experiment IR）──
  const handleTrain = async () => {
    setTraining(true);
    setStatusMsg("");
    try {
      const modelIR = toModelIR();

      // 1. 保存 Model IR
      await modelIrApi.save(modelIR);

      // 2. 创建 Experiment IR（使用简单默认配置）
      const expId = `exp-${modelIR.id}`;
      const expIR = {
        id: expId,
        name: `${modelName} Training`,
        model_ir_id: modelIR.id,
        data_ir_id: "data-default",   // 前端另有 Data 配置面板（预留）
        hyper_params: { epochs: 20 },
        backend: { type: "local", device: "auto" },
        checkpoint: { enabled: true, save_dir: "./checkpoints" },
      };
      await experimentApi.create(expIR);

      // 3. 提交训练
      await experimentApi.submit(expId);
      setActiveExpId(expId);
      setShowMonitor(true);
      setStatusMsg("🚀 训练已提交");
    } catch (e: any) {
      setStatusMsg(`❌ ${e.message}`);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh",
      background: "#0d1117", color: "#e2e8f0", fontFamily: "sans-serif" }}>

      {/* ── 顶部工具栏 ── */}
      <div style={{
        height: 48, background: "#0f172a", borderBottom: "1px solid #1e293b",
        display: "flex", alignItems: "center", padding: "0 16px", gap: 12,
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{ fontWeight: 800, fontSize: 15, color: "#6366f1",
          letterSpacing: "-0.02em" }}>
          ⚡ NoCode PyTorch
        </div>

        <div style={{ width: 1, height: 20, background: "#1e293b" }} />
        
        {/* 数据配置按钮 */}
        <button
          onClick={() => setDataOpen(true)}
          style={toolbarBtn("#0ea5e9")}
        >
          🗄️ 数据配置
        </button>

        <div style={{ width: 1, height: 20, background: "#1e293b" }} />
        
        {/* 模型名称 */}
        <input
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          style={{
            background: "transparent", border: "none", color: "#e2e8f0",
            fontSize: 14, fontWeight: 600, outline: "none", width: 160,
          }}
        />

        <div style={{ flex: 1 }} />

        {/* 状态提示 */}
        {statusMsg && (
          <span style={{ fontSize: 12, color: "#94a3b8" }}>{statusMsg}</span>
        )}

        {/* 操作按钮 */}
        <button onClick={handleSave} disabled={saving}
          style={toolbarBtn("#334155")}>
          {saving ? "保存中..." : "💾 保存"}
        </button>

        {showMonitor ? (
          <button onClick={() => setShowMonitor(true)}
            style={toolbarBtn("#6366f1")}>
            📊 监控
          </button>
        ) : (
          <button onClick={handleTrain} disabled={training}
            style={toolbarBtn("#10b981")}>
            {training ? "提交中..." : "▶ 开始训练"}
          </button>
        )}
      </div>

      {/* ── 主内容区 ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <NodePanel />
        <FlowCanvas />
        <ConfigPanel />
      </div>

      {/* ── 训练监控浮层 ── */}
      {showMonitor && (
        <TrainingMonitor
          experimentId={activeExpId}
          onClose={() => setShowMonitor(false)}
        />
      )}

      {/* ── 数据配置抽屉 ── */}
      <DataPanel
        open={dataOpen}
        onClose={() => setDataOpen(false)}
      />
    </div>
  );
}

const toolbarBtn = (bg: string): React.CSSProperties => ({
  background: bg, border: "none", borderRadius: 7,
  color: "#fff", fontSize: 12, fontWeight: 600,
  padding: "6px 14px", cursor: "pointer",
  transition: "opacity 0.15s",
});