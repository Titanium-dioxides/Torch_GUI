import { useEffect, useRef, useState } from "react";
import { TrainingProgress } from "@/types/ir";

const SSE_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export function useSSE(experimentId: string | null) {
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<TrainingProgress[]>([]);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    // 关闭旧连接
    esRef.current?.close();

    const es = new EventSource(
      `${SSE_BASE}/stream/${experimentId}/progress`
    );

    es.onmessage = (e) => {
      const data: TrainingProgress = JSON.parse(e.data);
      setProgress(data);

      // 只记录 running 状态的逐 epoch 数据
      if (data.status === "running" && data.current_epoch !== undefined) {
        setHistory((prev) => {
          // 去重：同一 epoch 只保留最新
          const filtered = prev.filter(
            (p) => p.current_epoch !== data.current_epoch
          );
          return [...filtered, data];
        });
      }

      // 训练结束，关闭 SSE
      if (["completed", "failed", "cancelled"].includes(data.status)) {
        es.close();
      }
    };

    es.onerror = () => es.close();
    esRef.current = es;

    return () => es.close();
  }, [experimentId]);

  const reset = () => {
    setProgress(null);
    setHistory([]);
  };

  return { progress, history, reset };
}