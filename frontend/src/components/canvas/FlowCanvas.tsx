import { useCallback, useRef } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlowInstance,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";

import { useCanvasStore } from "@/store/canvasStore";
import { NodeMeta } from "./nodeRegistry";
import { BaseNode } from "./nodes/BaseNode";

const nodeTypes = { baseNode: BaseNode };

function FlowCanvasInner() {
  const {
    nodes, edges,
    onNodesChange, onEdgesChange, onConnect,
    selectNode, addNode,
  } = useCanvasStore();

  const rfInstanceRef = useRef<ReactFlowInstance | null>(null);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData("application/node-meta");
      if (!raw || !rfInstanceRef.current) return;

      const meta: NodeMeta = JSON.parse(raw);
      const position = rfInstanceRef.current.screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });
      addNode(meta, position);
    },
    [addNode]
  );

  return (
    <div style={{ flex: 1, height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={(inst) => { rfInstanceRef.current = inst; }}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onPaneClick={() => selectNode(null)}
        fitView
        style={{ background: "#0d1117" }}
        defaultEdgeOptions={{ animated: true, style: { stroke: "#334155" } }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          color="#1e293b"
          gap={20}
        />
        <Controls
          style={{ background: "#1e293b", border: "1px solid #334155" }}
        />
        <MiniMap
          nodeColor={(n) => {
            const data = (n as any).data;
            return data?.irNode?.op_type === "Input"  ? "#6366f1"
                 : data?.irNode?.op_type === "Output" ? "#6366f1"
                 : "#334155";
          }}
          style={{ background: "#0f172a", border: "1px solid #1e293b" }}
        />
      </ReactFlow>
    </div>
  );
}

export function FlowCanvas() {
  return (
    <ReactFlowProvider>
      <FlowCanvasInner />
    </ReactFlowProvider>
  );
}