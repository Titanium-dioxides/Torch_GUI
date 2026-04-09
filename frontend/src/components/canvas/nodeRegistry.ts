export interface NodeMeta {
  op_type: string;
  label: string;
  category: string;
  defaultParams: Record<string, any>;
  inputs?: number;
  outputs?: number;
  color?: string;
  description?: string;
}

export const NODE_CATEGORIES = [
  { id: "io", label: "输入输出" },
  { id: "conv", label: "卷积层" },
  { id: "norm", label: "归一化" },
  { id: "activation", label: "激活函数" },
  { id: "pool", label: "池化层" },
  { id: "linear", label: "全连接层" },
  { id: "structure", label: "结构层" },
  { id: "multi", label: "多输入" },
];

const NODE_REGISTRY: Record<string, NodeMeta> = {
  "Input": {
    op_type: "Input",
    label: "Input",
    category: "io",
    defaultParams: { shape: [3, 224, 224] },
    outputs: 1,
    color: "#6366f1",
    description: "输入层，定义输入张量的形状 [C, H, W]",
  },
  "Output": {
    op_type: "Output",
    label: "Output",
    category: "io",
    defaultParams: {},
    inputs: 1,
    color: "#6366f1",
    description: "输出层，标记模型的最终输出",
  },
  "Conv2d": {
    op_type: "Conv2d",
    label: "Conv2d",
    category: "conv",
    defaultParams: {
      in_channels: 3,
      out_channels: 64,
      kernel_size: 3,
      stride: 1,
      padding: 1,
      dilation: 1,
      groups: 1,
      bias: true,
    },
    inputs: 1,
    outputs: 1,
    color: "#10b981",
    description: "2D 卷积层，用于提取图像特征",
  },
  "DepthwiseConv2d": {
    op_type: "DepthwiseConv2d",
    label: "DepthwiseConv2d",
    category: "conv",
    defaultParams: {
      in_channels: 64,
      kernel_size: 3,
      stride: 1,
      padding: 1,
    },
    inputs: 1,
    outputs: 1,
    color: "#10b981",
    description: "深度可分离卷积，参数量更少",
  },
  "ConvTranspose2d": {
    op_type: "ConvTranspose2d",
    label: "ConvTranspose2d",
    category: "conv",
    defaultParams: {
      in_channels: 64,
      out_channels: 3,
      kernel_size: 3,
      stride: 1,
      padding: 1,
      output_padding: 0,
    },
    inputs: 1,
    outputs: 1,
    color: "#10b981",
    description: "转置卷积，用于上采样和生成任务",
  },
  "BatchNorm2d": {
    op_type: "BatchNorm2d",
    label: "BatchNorm2d",
    category: "norm",
    defaultParams: {
      num_features: 64,
      eps: 1e-5,
      momentum: 0.1,
      affine: true,
    },
    inputs: 1,
    outputs: 1,
    color: "#f59e0b",
    description: "批归一化层，加速训练并提高稳定性",
  },
  "LayerNorm": {
    op_type: "LayerNorm",
    label: "LayerNorm",
    category: "norm",
    defaultParams: {
      normalized_shape: [64],
      eps: 1e-5,
    },
    inputs: 1,
    outputs: 1,
    color: "#f59e0b",
    description: "层归一化，常用于 Transformer",
  },
  "ReLU": {
    op_type: "ReLU",
    label: "ReLU",
    category: "activation",
    defaultParams: { inplace: true },
    inputs: 1,
    outputs: 1,
    color: "#ec4899",
    description: "ReLU 激活函数，引入非线性",
  },
  "LeakyReLU": {
    op_type: "LeakyReLU",
    label: "LeakyReLU",
    category: "activation",
    defaultParams: { negative_slope: 0.01, inplace: true },
    inputs: 1,
    outputs: 1,
    color: "#ec4899",
    description: "LeakyReLU，解决 ReLU 的神经元死亡问题",
  },
  "GELU": {
    op_type: "GELU",
    label: "GELU",
    category: "activation",
    defaultParams: {},
    inputs: 1,
    outputs: 1,
    color: "#ec4899",
    description: "GELU 激活函数，用于 Transformer",
  },
  "Sigmoid": {
    op_type: "Sigmoid",
    label: "Sigmoid",
    category: "activation",
    defaultParams: {},
    inputs: 1,
    outputs: 1,
    color: "#ec4899",
    description: "Sigmoid 激活函数，输出范围 (0, 1)",
  },
  "Tanh": {
    op_type: "Tanh",
    label: "Tanh",
    category: "activation",
    defaultParams: {},
    inputs: 1,
    outputs: 1,
    color: "#ec4899",
    description: "Tanh 激活函数，输出范围 (-1, 1)",
  },
  "MaxPool2d": {
    op_type: "MaxPool2d",
    label: "MaxPool2d",
    category: "pool",
    defaultParams: {
      kernel_size: 2,
      stride: 2,
      padding: 0,
    },
    inputs: 1,
    outputs: 1,
    color: "#8b5cf6",
    description: "最大池化，降采样并保留最强特征",
  },
  "AvgPool2d": {
    op_type: "AvgPool2d",
    label: "AvgPool2d",
    category: "pool",
    defaultParams: {
      kernel_size: 2,
      stride: 2,
      padding: 0,
    },
    inputs: 1,
    outputs: 1,
    color: "#8b5cf6",
    description: "平均池化，平滑特征",
  },
  "AdaptiveAvgPool2d": {
    op_type: "AdaptiveAvgPool2d",
    label: "AdaptiveAvgPool2d",
    category: "pool",
    defaultParams: {
      output_size: [1, 1],
    },
    inputs: 1,
    outputs: 1,
    color: "#8b5cf6",
    description: "自适应平均池化，输出固定尺寸",
  },
  "Linear": {
    op_type: "Linear",
    label: "Linear",
    category: "linear",
    defaultParams: {
      in_features: 512,
      out_features: 10,
      bias: true,
    },
    inputs: 1,
    outputs: 1,
    color: "#06b6d4",
    description: "全连接层，用于特征映射",
  },
  "Flatten": {
    op_type: "Flatten",
    label: "Flatten",
    category: "structure",
    defaultParams: {
      start_dim: 1,
      end_dim: -1,
    },
    inputs: 1,
    outputs: 1,
    color: "#64748b",
    description: "展平层，将多维张量展平为一维",
  },
  "Dropout": {
    op_type: "Dropout",
    label: "Dropout",
    category: "structure",
    defaultParams: {
      p: 0.5,
      inplace: false,
    },
    inputs: 1,
    outputs: 1,
    color: "#64748b",
    description: "Dropout，防止过拟合",
  },
  "Dropout2d": {
    op_type: "Dropout2d",
    label: "Dropout2d",
    category: "structure",
    defaultParams: {
      p: 0.5,
      inplace: false,
    },
    inputs: 1,
    outputs: 1,
    color: "#64748b",
    description: "2D Dropout，用于卷积网络",
  },
  "Add": {
    op_type: "Add",
    label: "Add",
    category: "multi",
    defaultParams: {},
    inputs: 2,
    outputs: 1,
    color: "#ef4444",
    description: "逐元素相加，用于残差连接",
  },
  "Concat": {
    op_type: "Concat",
    label: "Concat",
    category: "multi",
    defaultParams: {
      dim: 1,
    },
    inputs: 2,
    outputs: 1,
    color: "#ef4444",
    description: "拼接张量，沿指定维度连接",
  },
};

export function getNodeMeta(op_type: string): NodeMeta {
  return NODE_REGISTRY[op_type];
}

export function getAllNodes(): NodeMeta[] {
  return Object.values(NODE_REGISTRY);
}