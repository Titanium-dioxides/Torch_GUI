import { useState } from "react";
import { useSSE } from "@/hooks/useSSE";
import { LossChart } from "./LossChart";
import { experimentApi } from "@/api/experiment";

interface Props {
  experimentId: string | null;
  onClose: () => void;
}

export function TrainingMonitor({ experimentId, onClose }: Props) {
  const { progress, history, reset } = useSSE(experimentId);
  const [cancelling, setCancelling] = useState(false);

  if (!experimentId) return null;

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await experimentApi.cancel(experimentId);
    } finally {
      setCancelling(false);
    }
  };

  const statusColor =
    progress?.status === "running"   ? "#10b981"
    : progress?.status === "completed" ? "#6366f1"
    : progress?.status === "failed"    ? "#ef4444"
    : progress?.status === "cancelled" ? "#f59e0b"
    : "#64748b";

  return (
    <div
      style={{
        position: "fixed", bottom: 20, right: 20,
        width: 440, background: "#0f172a",
        border: "1px solid #1e293b", borderRadius: 12,
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        zIndex: 1000, overflow: "hidden",
      }}
    >
      {/* 标题栏 */}
      <div
        style={{
          padding: "10px 14px",
          background: "#1e293b",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 8, height: 8, borderRadius: "50%",
              background: statusColor,
              boxShadow: progress?.status === "running"
                ? `0 0 6px ${statusColor}` : "none",
            }}
          />
          <span style={{ color: "#e2e8f0", fontSize: 13, fontWeight: 600 }}>
            训练监控
          </span>
          <span style={{ color: "#475569", fontSize: 11 }}>
            {progress?.status ?? "—"}
          </span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {progress?.status === "running" && (
            <button onClick={handleCancel} disabled={cancelling}
              style={btnStyle("#ef4444")}>
              {cancelling ? "取消中..." : "停止"}
            </button>
          )}
          <button onClick={() => { reset(); onClose(); }}
            style={btnStyle("#334155")}>
            关闭
          </button>
        </div>
      </div>

      {/* 实时指标 */}
      {progress && (
        <div
          style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr",
            gap: 1, background: "#1e293b", margin: "1px 0",
          }}
        >
          {[
            ["Epoch", progress.current_epoch !== undefined
              ? `${progress.current_epoch} / ${progress.total_epochs}` : "—"],
            ["Train Loss", progress.train_loss?.toFixed(4) ?? "—"],
            ["Train Acc",  progress.train_acc !== undefined
              ? `${(progress.train_acc * 100).toFixed(1)}%` : "—"],
            ["Val Acc",    progress.val_acc !== undefined
              ? `${(progress.val_acc * 100).toFixed(1)}%` : "—"],
          ].map(([label, value]) => (
            <div key={label}
              style={{ background: "#0f172a", padding: "10px 12px", textAlign: "center" }}>
              <div style={{ color: "#475569", fontSize: 10 }}>{label}</div>
              <div style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 700, marginTop: 2 }}>
                {value}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Loss/Acc 曲线 */}
      <div style={{ padding: 14 }}>
        {history.length > 0 ? (
          <LossChart history={history} />
        ) : (
          <div style={{ color: "#334155", fontSize: 12, textAlign: "center", padding: 20 }}>
            等待训练数据...
          </div>
        )}
      </div>
    </div>
  );
}

const btnStyle = (bg: string): React.CSSProperties => ({
  background: bg, border: "none", borderRadius: 6,
  color: "#fff", fontSize: 11, padding: "4px 10px",
  cursor: "pointer",
});