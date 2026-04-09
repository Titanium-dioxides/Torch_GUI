import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Connection,
  Edge,
  EdgeChange,
  Node,
  NodeChange,
} from "reactflow";
import { nanoid } from "nanoid";
import { IRNode, ModelIR, NodeMeta, OpType } from "@/types/ir";

// React Flow 节点的 data 字段
export interface NodeData {
  irNode: IRNode;
  selected: boolean;
}

export type FlowNode = Node<NodeData>;
export type FlowEdge = Edge;

interface CanvasState {
  // React Flow 状态
  nodes: FlowNode[];
  edges: FlowEdge[];

  // 当前选中节点（用于右侧配置面板）
  selectedNodeId: string | null;

  // 模型元信息
  modelId: string;
  modelName: string;

  // Actions
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (meta: NodeMeta, position: { x: number; y: number }) => void;
  selectNode: (nodeId: string | null) => void;
  updateNodeParams: (nodeId: string, params: Record<string, unknown>) => void;
  updateNodeName: (nodeId: string, name: string) => void;
  deleteNode: (nodeId: string) => void;
  setModelName: (name: string) => void;

  // IR 序列化
  toModelIR: () => ModelIR;
  loadFromModelIR: (ir: ModelIR) => void;
}

export const useCanvasStore = create<CanvasState>()(
  immer((set, get) => ({
    nodes: [],
    edges: [],
    selectedNodeId: null,
    modelId: nanoid(),
    modelName: "MyModel",

    // ── React Flow 事件处理 ──

    onNodesChange: (changes) =>
      set((state) => {
        state.nodes = applyNodeChanges(changes, state.nodes) as FlowNode[];
      }),

    onEdgesChange: (changes) =>
      set((state) => {
        state.edges = applyEdgeChanges(changes, state.edges);
      }),

    onConnect: (connection) =>
      set((state) => {
        state.edges = addEdge(
          { ...connection, id: `e-${nanoid(6)}`, animated: true },
          state.edges
        );
      }),

    // ── 节点操作 ──

    addNode: (meta, position) =>
      set((state) => {
        const id = `node-${nanoid(6)}`;
        const irNode: IRNode = {
          id,
          op_type: meta.op_type,
          name: `${meta.op_type.toLowerCase()}_${state.nodes.length + 1}`,
          params: { ...meta.defaultParams },
          position,
        };
        const flowNode: FlowNode = {
          id,
          type: "baseNode",
          position,
          data: { irNode, selected: false },
        };
        state.nodes.push(flowNode);
      }),

    selectNode: (nodeId) =>
      set((state) => {
        state.selectedNodeId = nodeId;
        state.nodes.forEach((n) => {
          n.data.selected = n.id === nodeId;
        });
      }),

    updateNodeParams: (nodeId, params) =>
      set((state) => {
        const node = state.nodes.find((n) => n.id === nodeId);
        if (node) {
          node.data.irNode.params = params;
        }
      }),

    updateNodeName: (nodeId, name) =>
      set((state) => {
        const node = state.nodes.find((n) => n.id === nodeId);
        if (node) node.data.irNode.name = name;
      }),

    deleteNode: (nodeId) =>
      set((state) => {
        state.nodes = state.nodes.filter((n) => n.id !== nodeId);
        state.edges = state.edges.filter(
          (e) => e.source !== nodeId && e.target !== nodeId
        );
        if (state.selectedNodeId === nodeId) state.selectedNodeId = null;
      }),

    setModelName: (name) =>
      set((state) => {
        state.modelName = name;
      }),

    // ── IR 序列化 ──

    toModelIR: (): ModelIR => {
      const { nodes, edges, modelId, modelName } = get();
      return {
        id: modelId,
        name: modelName,
        version: "1.0.0",
        nodes: nodes.map((n) => ({
          ...n.data.irNode,
          position: { x: n.position.x, y: n.position.y },
        })),
        edges: edges.map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          source_handle: e.sourceHandle ?? undefined,
          target_handle: e.targetHandle ?? undefined,
        })),
      };
    },

    loadFromModelIR: (ir: ModelIR) =>
      set((state) => {
        state.modelId = ir.id;
        state.modelName = ir.name;
        state.nodes = ir.nodes.map((irNode) => ({
          id: irNode.id,
          type: "baseNode",
          position: irNode.position ?? { x: 0, y: 0 },
          data: { irNode, selected: false },
        }));
        state.edges = ir.edges.map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.source_handle,
          targetHandle: e.target_handle,
          animated: true,
        }));
        state.selectedNodeId = null;
      }),
  }))
);