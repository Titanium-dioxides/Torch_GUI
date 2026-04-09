// ── 数据源类型 ──

export type DataSourceType = "torchvision" | "local_folder" | "local_csv";

export type TaskType =
  | "image_classification"
  | "object_detection"
  | "segmentation"
  | "regression";

export interface TorchvisionSource {
  type:         "torchvision";
  dataset_name: string;
  download_root: string;
  download?:      boolean;
}

export interface ImageFolderSource {
  type:      "local_folder";
  root_path:  string;
}

export interface CsvSource {
  type:          "local_csv";
  csv_path:      string;
  image_col:     string;
  label_col:     string;
  image_root:    string;
}

export type DataSource = TorchvisionSource | ImageFolderSource | CsvSource;

// ── Transform ──

export type TransformOpType =
  | "ToTensor"
  | "Normalize"
  | "RandomHorizontalFlip"
  | "RandomVerticalFlip"
  | "RandomCrop"
  | "CenterCrop"
  | "Resize"
  | "RandomResizedCrop"
  | "ColorJitter"
  | "RandomRotation"
  | "Grayscale"
  | "RandomGrayscale"
  | "RandomErasing"
  | "AutoAugment"
  | "TrivialAugmentWide";

export interface TransformNode {
  id:       string;
  op_type:  TransformOpType;
  params:   Record<string, unknown>;
  enabled?:  boolean;
}

export interface TransformPipeline {
  transforms: TransformNode[];
}

// ── DataLoader 配置 ──

export interface DataLoaderConfig {
  batch_size:      number;
  num_workers:     number;
  shuffle:         boolean;
  pin_memory:      boolean;
  drop_last:       boolean;
  prefetch_factor?: number | null;
}

// ── 划分配置 ──

export interface RatioSplitConfig {
  type:        "ratio";
  strategy:     "ratio";
  train_ratio: number;
  val_ratio:   number;
  test_ratio:  number;
  random_seed: number;
}

// ── 顶层 Data IR ──

export interface DataSchema {
  task_type:      TaskType;
  num_classes:    number;
  class_names:    string[];
  input_channels: number;
  input_shape:    [number, number, number];  // [C, H, W]
}

export interface DataIR {
  id:                 string;
  name:               string;
  version?:            string;
  source:             DataSource;
  schema:             DataSchema;
  split:              RatioSplitConfig;
  train_pipeline:     TransformPipeline;
  val_pipeline:       TransformPipeline;
  test_pipeline?:     TransformPipeline;
  train_loader_config: DataLoaderConfig;
  val_loader_config:  DataLoaderConfig;
  description?:        string;
  tags?:               string[];
}

// ── 预览响应 ──

export interface DataPreviewInfo {
  total_samples:  number;
  train_samples:  number;
  val_samples:    number;
  test_samples:   number;
  num_classes:    number;
  class_names:    string[];
  sample_shape:   number[];
}