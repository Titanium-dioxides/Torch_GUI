import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { nanoid } from "nanoid";
import {
  DataIR, DataSource, DataLoaderConfig,
  RatioSplitConfig, TransformNode,
} from "@/types/dataIr";
import { createTransformNode, TransformMeta } from "@/components/data/TransformRegistry";

interface DataState {
  // 核心 IR 字段
  dataId:   string;
  dataName: string;
  source:   DataSource;
  schema: {
    task_type:      string;
    num_classes:    number;
    class_names:    string[];
    input_channels: number;
    input_shape:    [number, number, number];
  };
  split: RatioSplitConfig;
  trainPipeline: TransformNode[];
  valPipeline:   TransformNode[];
  trainLoaderConfig: DataLoaderConfig;
  valLoaderConfig:   DataLoaderConfig;

  // Actions
  setDataName:    (name: string) => void;
  setSource:      (source: DataSource) => void;
  setSchema:      (patch: Partial<DataState["schema"]>) => void;
  setSplit:       (patch: Partial<RatioSplitConfig>) => void;

  // Transform
  addTrainTransform:    (meta: TransformMeta) => void;
  addValTransform:      (meta: TransformMeta) => void;
  removeTrainTransform: (id: string) => void;
  removeValTransform:   (id: string) => void;
  updateTrainTransform: (id: string, params: Record<string, unknown>) => void;
  updateValTransform:   (id: string, params: Record<string, unknown>) => void;
  reorderTrainTransform:(fromIdx: number, toIdx: number) => void;
  reorderValTransform:  (fromIdx: number, toIdx: number) => void;

  // Loader
  setTrainLoaderConfig: (patch: Partial<DataLoaderConfig>) => void;
  setValLoaderConfig:   (patch: Partial<DataLoaderConfig>) => void;

  // 序列化
  toDataIR: () => DataIR;
}

const defaultLoader = (): DataLoaderConfig => ({
  batch_size:      64,
  num_workers:     2,
  shuffle:         true,
  pin_memory:      true,
  drop_last:       false,
  prefetch_factor: 2,
});

export const useDataStore = create<DataState>()(
  immer((set, get) => ({
    dataId:   nanoid(),
    dataName: "MyDataset",
    source: {
      type:          "torchvision",
      dataset_name:  "CIFAR10",
      download_root: "./data",
      download:       true,
    } as DataSource,
    schema: {
      task_type:      "image_classification",
      num_classes:    10,
      class_names:    [],
      input_channels:  3,
      input_shape:    [3, 32, 32],
    },
    split: {
      type:        "ratio",
      strategy:     "ratio",
      train_ratio: 0.8,
      val_ratio:   0.1,
      test_ratio:  0.1,
      random_seed: 42,
    },
    trainPipeline: [],
    valPipeline:   [],
    trainLoaderConfig: defaultLoader(),
    valLoaderConfig:   { ...defaultLoader(), shuffle: false },

    setDataName: (name) => set((s) => { s.dataName = name; }),
    setSource:   (source) => set((s) => { s.source = source; }),
    setSchema:   (patch)  => set((s) => { Object.assign(s.schema, patch); }),
    setSplit:    (patch)  => set((s) => { Object.assign(s.split, patch); }),

    // ── Transform 操作 ──

    addTrainTransform: (meta) => set((s) => {
      s.trainPipeline.push(createTransformNode(meta));
    }),
    addValTransform: (meta) => set((s) => {
      s.valPipeline.push(createTransformNode(meta));
    }),

    removeTrainTransform: (id) => set((s) => {
      s.trainPipeline = s.trainPipeline.filter((t) => t.id !== id);
    }),
    removeValTransform: (id) => set((s) => {
      s.valPipeline = s.valPipeline.filter((t) => t.id !== id);
    }),

    updateTrainTransform: (id, params) => set((s) => {
      const t = s.trainPipeline.find((t) => t.id === id);
      if (t) t.params = params;
    }),
    updateValTransform: (id, params) => set((s) => {
      const t = s.valPipeline.find((t) => t.id === id);
      if (t) t.params = params;
    }),

    reorderTrainTransform: (fromIdx, toIdx) => set((s) => {
      const [item] = s.trainPipeline.splice(fromIdx, 1);
      s.trainPipeline.splice(toIdx, 0, item);
    }),
    reorderValTransform: (fromIdx, toIdx) => set((s) => {
      const [item] = s.valPipeline.splice(fromIdx, 1);
      s.valPipeline.splice(toIdx, 0, item);
    }),

    setTrainLoaderConfig: (patch) => set((s) => { Object.assign(s.trainLoaderConfig, patch); }),
    setValLoaderConfig:   (patch) => set((s) => { Object.assign(s.valLoaderConfig, patch); }),

    // ── 序列化为 DataIR ──

    toDataIR: (): DataIR => {
      const s = get();
      return {
        id:   s.dataId,
        name: s.dataName,
        version: "1.0.0",
        source: s.source,
        schema: {
          task_type:      s.schema.task_type as any,
          num_classes:    s.schema.num_classes,
          class_names:    s.schema.class_names,
          input_channels: s.schema.input_channels,
          input_shape:    s.schema.input_shape,
        },
        split: s.split,
        train_pipeline:      { transforms: s.trainPipeline },
        val_pipeline:        { transforms: s.valPipeline },
        test_pipeline:       { transforms: [] },
        train_loader_config: s.trainLoaderConfig,
        val_loader_config:   s.valLoaderConfig,
        description: "",
        tags: [],
      };
    },
  })
))