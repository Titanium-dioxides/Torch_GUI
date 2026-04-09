// ── Model IR ──

export type OpType =
  | "Input" | "Output"
  | "Conv2d" | "DepthwiseConv2d" | "ConvTranspose2d"
  | "BatchNorm2d" | "LayerNorm"
  | "ReLU" | "LeakyReLU" | "GELU" | "Sigmoid" | "Tanh"
  | "MaxPool2d" | "AvgPool2d" | "AdaptiveAvgPool2d"
  | "Linear" | "Flatten" | "Dropout" | "Dropout2d"
  | "Add" | "Concat";

export interface IRNode {
  id: string;
  op_type: OpType;
  name: string;
  params: Record<string, unknown>;
  position?: { x: number; y: number };
  output_shape?: number[];
}

export interface IREdge {
  id: string;
  source: string;
  target: string;
  source_handle?: string;
  target_handle?: string;
}

export interface ModelIR {
  id: string;
  name: string;
  version: string;
  nodes: IRNode[];
  edges: IREdge[];
  description?: string;
  tags?: string[];
}

// ── 节点元信息（用于左侧面板展示）──

export interface NodeMeta {
  op_type: OpType;
  label: string;
  category: string;
  color: string;
  defaultParams: Record<string, unknown>;
  description: string;
}

// ── 训练进度 ──

export interface TrainingProgress {
  experiment_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  current_epoch?: number;
  total_epochs?: number;
  train_loss?: number;
  train_acc?: number;
  val_loss?: number;
  val_acc?: number;
  best_val_acc?: number;
  error?: string;
  type?: "final";
}