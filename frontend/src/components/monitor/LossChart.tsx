import {
  CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { TrainingProgress } from "@/types/ir";

interface Props { history: TrainingProgress[] }

export function LossChart({ history }: Props) {
  const data = history.map((p) => ({
    epoch:      p.current_epoch,
    train_loss: p.train_loss ? +p.train_loss.toFixed(4) : undefined,
    val_acc:    p.val_acc    ? +(p.val_acc * 100).toFixed(2) : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="epoch" tick={{ fill: "#475569", fontSize: 10 }} />
        <YAxis yAxisId="loss" tick={{ fill: "#475569", fontSize: 10 }} />
        <YAxis yAxisId="acc" orientation="right" tick={{ fill: "#475569", fontSize: 10 }} />
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "none", borderRadius: 6, fontSize: 11 }}
          labelStyle={{ color: "#94a3b8" }}
        />
        <Legend wrapperStyle={{ fontSize: 10, color: "#64748b" }} />
        <Line
          yAxisId="loss" type="monotone" dataKey="train_loss"
          stroke="#0ea5e9" strokeWidth={2} dot={false} name="Train Loss"
        />
        <Line
          yAxisId="acc" type="monotone" dataKey="val_acc"
          stroke="#10b981" strokeWidth={2} dot={false} name="Val Acc %"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}