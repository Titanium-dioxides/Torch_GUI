import { useCallback, useEffect, useRef } from "react";
import { useCanvasStore } from "@/store/canvasStore";
import { apiClient } from "@/api/client";

interface NodeShapeInfo {
  node_id:      string;
  node_name:    string;
  op_type:      string;
  output_shape: number[];
}

interface ShapeInferResponse {
  success:     boolean;
  node_shapes: NodeShapeInfo[];
  errors:      Array<{ node_id: string; message: string }>;
}

/**
 * 画布变化时自动触发 Shape Inference。
 * 将推导到的 output_shape 写回各节点的 irNode.output_shape，
 * 触发 React Flow 重渲染，节点下方实时显示维度信息。
 */
export function useShapeInference(debounceMs = 600) {
  const { nodes, edges, toModelIR } = useCanvasStore();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 将 shape 结果写回 store
  const applyShapes = useCallback(
    (nodeShapes: NodeShapeInfo[]) => {
      const store = useCanvasStore.getState();
      const shapeMap = new Map(nodeShapes.map((s) => [s.node_id, s.output_shape]));

      // 直接操作 Zustand store
      useCanvasStore.setState((state) => ({
        nodes: state.nodes.map((n) => {
          const shape = shapeMap.get(n.id);
          if (!shape || shape.length === 0) return n;
          return {
            ...n,
            data: {
              ...n.data,
              irNode: { ...n.data.irNode, output_shape: shape },
            },
          };
        }),
      }));
    },
    []
  );

  useEffect(() => {
    // 画布为空时不触发
    if (nodes.length === 0) return;

    // 防抖：用户停止操作 600ms 后触发
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      try {
        const ir = toModelIR();
        const res = await apiClient.post<{ data: ShapeInferResponse }>(
          "/shape-infer/preview",
          ir
        );
        applyShapes(res.data.data.node_shapes);
      } catch {
        // Shape Inference 失败时静默处理（不影响用户操作）
      }
    }, debounceMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [nodes, edges]);   // nodes / edges 变化时触发
}