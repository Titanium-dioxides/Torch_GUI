# NoCode PyTorch Platform - 数据流文档

## 📊 目录

- [整体数据流](#整体数据流)
- [模型设计数据流](#模型设计数据流)
- [训练执行数据流](#训练执行数据流)
- [数据处理数据流](#数据处理数据流)
- [形状推导数据流](#形状推导数据流)
- [实时监控数据流](#实时监控数据流)

---

## 整体数据流

### 系统级数据流图

```mermaid
graph TB
    subgraph "前端层 Frontend"
        A1[React Flow Canvas]
        A2[Zustand Store]
        A3[API Client]
        A4[UI Components]
    end
    
    subgraph "API 层 FastAPI"
        B1[REST API Router]
        B2[Request Validator]
        B3[Response Formatter]
    end
    
    subgraph "业务逻辑层 Business Logic"
        C1[Model IR Manager]
        C2[Data IR Manager]
        C3[Experiment Manager]
        C4[Shape Inference Engine]
        C5[PyTorch CodeGen]
    end
    
    subgraph "任务队列层 Task Queue"
        D1[Celery Broker]
        D2[Celery Worker]
        D3[Task Executor]
    end
    
    subgraph "训练层 Training"
        E1[Trainer]
        E2[PyTorch Model]
        E3[DataLoader]
    end
    
    subgraph "存储层 Storage"
        F1[Memory Store]
        F2[File System]
        F3[Redis Cache]
    end
    
    A4 --> A1
    A1 --> A2
    A2 --> A3
    A3 -->|HTTP Request| B1
    B1 --> B2
    B2 --> C1
    C1 --> F1
    C1 --> C4
    C4 --> F1
    C1 --> C5
    C5 --> B3
    B3 -->|HTTP Response| A3
    
    A3 -->|Submit Training| B1
    B1 --> C3
    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> C1
    D3 --> C2
    D3 --> E1
    E1 --> E2
    E1 --> E3
    E1 -->|Progress Update| D1
    D1 -->|SSE Push| A3
    
    E1 -->|Checkpoint| F2
    E1 -->|Metrics| F3
```

### 数据流说明

1. **前端交互流**：用户在 React Flow Canvas 中操作，状态存储在 Zustand Store，通过 API Client 发送 HTTP 请求
2. **API 处理流**：FastAPI 接收请求，验证数据，调用业务逻辑层
3. **IR 管理流**：Model IR、Data IR、Experiment IR 在内存存储中管理
4. **形状推导流**：Shape Inference Engine 推导每层的输入输出形状
5. **代码生成流**：PyTorch CodeGen 将 Model IR 转换为可执行代码
6. **训练任务流**：Celery 接收训练任务，Worker 执行训练
7. **训练执行流**：Trainer 使用 PyTorch Model 和 DataLoader 执行训练
8. **进度推送流**：训练进度通过 Celery Broker 推送到前端
9. **结果存储流**：检查点和指标存储到文件系统和 Redis

---

## 模型设计数据流

### 详细流程图

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 用户
    participant C as 🎨 Canvas
    participant S as 📦 Store
    participant A as 🌐 API
    participant I as 🧠 Model IR
    participant G as ⚙️ CodeGen
    participant P as 📄 Preview

    U->>C: 1. 拖拽节点到画布
    C->>S: 2. 添加节点到状态
    S->>S: 3. 触发形状推导
    S->>A: 4. POST /shape-inference
    A->>I: 5. 调用形状推导引擎
    I->>I: 6. 拓扑排序节点
    I->>I: 7. 应用形状规则
    I-->>A: 8. 返回推导结果
    A-->>S: 9. 返回形状信息
    S->>S: 10. 更新节点形状
    S->>C: 11. 重新渲染画布
    C->>U: 12. 显示节点形状

    U->>C: 13. 配置节点参数
    C->>S: 14. 更新节点参数
    S->>S: 15. 重新触发形状推导
    S->>A: 16. POST /shape-inference
    A->>I: 17. 重新推导形状
    I-->>A: 18. 返回新形状
    A-->>S: 19. 返回形状信息
    S->>C: 20. 更新画布显示

    U->>C: 21. 连接两个节点
    C->>S: 22. 添加边到状态
    S->>S: 23. 检查图结构
    S->>S: 24. 重新触发形状推导
    S->>A: 25. POST /shape-inference
    A->>I: 26. 重新推导形状
    I-->>A: 27. 返回形状
    A-->>S: 28. 返回形状信息
    S->>C: 29. 更新画布显示

    U->>C: 30. 点击"保存模型"
    C->>S: 31. 调用 toModelIR()
    S->>A: 32. POST /model-irs
    A->>I: 33. 序列化 Model IR
    I->>I: 34. 验证图结构
    I->>I: 35. 验证节点参数
    I-->>A: 36. 返回验证结果
    A-->>S: 37. 返回保存结果
    S->>C: 38. 显示保存成功
    C->>U: 39. 提示用户

    U->>C: 40. 点击"生成代码"
    C->>S: 41. 调用代码生成
    S->>A: 42. GET /model-irs/{id}/codegen
    A->>I: 43. 获取 Model IR
    I->>G: 44. 调用代码生成器
    G->>G: 45. 拓扑排序节点
    G->>G: 46. 生成 __init__ 方法
    G->>G: 47. 生成 forward 方法
    G->>G: 48. 渲染完整代码
    G-->>I: 49. 返回代码字符串
    I-->>A: 50. 返回代码
    A-->>S: 51. 返回代码
    S->>P: 52. 显示生成的代码
    P->>U: 53. 展示 PyTorch 代码
```

### 数据结构转换

#### 1. Canvas Node → IRNode
```typescript
// 前端 Canvas Node
interface CanvasNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    irNode: IRNode;
    label: string;
  };
}

// 转换为后端 IRNode
interface IRNode {
  id: string;
  op_type: OpType;
  name: string;
  params: Record<string, any>;
  position?: { x: number; y: number };
  output_shape?: number[];
}
```

#### 2. Canvas Edge → IREdge
```typescript
// 前端 Canvas Edge
interface CanvasEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

// 转换为后端 IREdge
interface IREdge {
  id: string;
  source: string;
  target: string;
  source_handle?: string;
  target_handle?: string;
}
```

#### 3. Model IR → PyTorch Code
```python
# Model IR 结构
ModelIR(
    id="model-001",
    name="Simple Cnn",
    nodes=[
        IRNode(id="n0", op_type=OpType.INPUT, name="input", params={"shape": [3, 32, 32]}),
        IRNode(id="n1", op_type=OpType.CONV2D, name="conv1", params={"in_channels": 3, "out_channels": 16, "kernel_size": 3}),
        IRNode(id="n2", op_type=OpType.RELU, name="relu1", params={}),
        IRNode(id="n3", op_type=OpType.OUTPUT, name="output", params={}),
    ],
    edges=[
        IREdge(id="e01", source="n0", target="n1"),
        IREdge(id="e12", source="n1", target="n2"),
        IREdge(id="e23", source="n2", target="n3"),
    ]
)

# 生成的 PyTorch 代码
class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        return x
```

---

## 训练执行数据流

### 详细流程图

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 用户
    participant F as 🖥️ 前端
    participant A as 🌐 API
    participant C as ⚡ Celery
    participant T as 📋 Train Task
    participant S as 💾 Storage
    participant M as 🧠 Model IR
    participant D as 📊 Data IR
    participant E as 🧪 Experiment IR
    participant G as ⚙️ CodeGen
    participant B as 🏋️ Builder
    participant R as 🏋️ Trainer
    participant P as 🔥 PyTorch

    U->>F: 1. 点击"开始训练"
    F->>F: 2. 创建 Experiment IR
    F->>A: 3. POST /experiments
    A->>S: 4. 保存 Experiment IR
    A-->>F: 5. 返回实验ID
    
    F->>A: 6. POST /experiments/{id}/submit
    A->>C: 7. 提交训练任务
    C->>T: 8. 分配任务给 Worker
    A-->>F: 9. 返回任务ID
    
    T->>S: 10. 加载 Experiment IR
    T->>S: 11. 加载 Model IR
    T->>S: 12. 加载 Data IR
    S-->>T: 13. 返回 Experiment IR
    S-->>T: 14. 返回 Model IR
    S-->>T: 15. 返回 Data IR
    
    T->>G: 16. 调用代码生成器
    G->>G: 17. 生成 PyTorch 代码
    G->>P: 18. 执行生成的代码
    P-->>G: 19. 返回模型类
    G-->>T: 20. 返回模型实例
    
    T->>B: 21. 构建 Dataset
    B->>D: 22. 解析数据源配置
    B->>B: 23. 应用切分策略
    B->>B: 24. 构建 Transform Pipeline
    B-->>T: 25. 返回 Dataset Bundle
    
    T->>B: 26. 构建 DataLoader
    B->>B: 27. 配置 batch_size
    B->>B: 28. 配置 num_workers
    B-->>T: 29. 返回 DataLoader Bundle
    
    T->>R: 30. 初始化 Trainer
    R->>P: 31. 初始化优化器
    R->>P: 32. 初始化学习率调度器
    R->>P: 33. 初始化损失函数
    R->>P: 34. 将模型移到设备
    R-->>T: 35. 返回 Trainer 实例
    
    loop 每个 Epoch
        R->>P: 36. 训练一个 Epoch
        P->>P: 37. 前向传播
        P->>P: 38. 计算损失
        P->>P: 39. 反向传播
        P->>P: 40. 更新参数
        P-->>R: 41. 返回训练指标
        
        R->>P: 42. 验证一个 Epoch
        P->>P: 43. 前向传播
        P->>P: 44. 计算指标
        P-->>R: 45. 返回验证指标
        
        R->>T: 46. 回调：Epoch 结束
        T->>T: 47. 计算 Epoch 指标
        T->>C: 48. 更新任务状态
        C->>F: 49. SSE 推送进度
        F->>U: 50. 更新训练监控
    end
    
    R->>T: 51. 训练完成
    T->>S: 52. 保存训练结果
    T->>C: 53. 更新任务状态为 COMPLETED
    C->>F: 54. SSE 推送完成通知
    F->>U: 55. 显示训练完成
```

### 训练数据流

#### 1. 数据加载流
```
Data IR → DatasetBuilder → DatasetBundle
    ↓
    ├─ Train Dataset
    │   └─ Train Transform Pipeline
    │       └─ Train DataLoader
    │           └─ Train Batches
    │
    ├─ Val Dataset
    │   └─ Val Transform Pipeline
    │       └─ Val DataLoader
    │           └─ Val Batches
    │
    └─ Test Dataset
        └─ Test Transform Pipeline
            └─ Test DataLoader
                └─ Test Batches
```

#### 2. 训练循环流
```
for epoch in range(epochs):
    # 训练阶段
    for batch in train_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 计算指标
        accuracy = calculate_accuracy(outputs, labels)
    
    # 学习率调度
    scheduler.step()
    
    # 回调
    on_epoch_end(EpochMetrics)
```

#### 3. 进度上报流
```
EpochMetrics → Celery Task.update_state()
    ↓
    Redis Backend
    ↓
    SSE Stream
    ↓
    Frontend useSSE Hook
    ↓
    TrainingMonitor Component
    ↓
    LossChart Component
```

---

## 数据处理数据流

### 详细流程图

```mermaid
flowchart TD
    A[Data IR] --> B{数据源类型}
    
    B -->|Torchvision| C[torchvision 数据集]
    B -->|Local Folder| D[ImageFolder]
    B -->|CSV| E[CSV Dataset]
    
    C --> F[下载/加载数据集]
    D --> F
    E --> F
    
    F --> G{切分策略}
    
    G -->|Ratio| H[随机切分]
    G -->|Predefined| I[使用已有切分]
    
    H --> J[生成随机索引]
    J --> K[按比例切分]
    K --> L[创建 Train Subset]
    K --> M[创建 Val Subset]
    K --> N[创建 Test Subset]
    
    I --> L
    I --> M
    I --> N
    
    L --> O[Train Transform Pipeline]
    M --> P[Val Transform Pipeline]
    N --> Q[Test Transform Pipeline]
    
    O --> R{Transform 类型}
    P --> R
    Q --> R
    
    R -->|Resize| S[调整图像大小]
    R -->|RandomCrop| T[随机裁剪]
    R -->|ToTensor| U[转换为张量]
    R -->|Normalize| V[归一化]
    
    S --> W[应用 Transform]
    T --> W
    U --> W
    V --> W
    
    W --> X[Transformed Dataset]
    
    X --> Y[DataLoader 配置]
    
    Y --> Z[Train DataLoader]
    Y --> AA[Val DataLoader]
    Y --> AB[Test DataLoader]
    
    Z --> AC[训练批次]
    AA --> AD[验证批次]
    AB --> AE[测试批次]
```

### Transform Pipeline 数据流

#### 1. Train Pipeline
```
原始图像
    ↓
RandomHorizontalFlip(p=0.5)
    ↓
RandomCrop(size=32, padding=4)
    ↓
ToTensor()
    ↓
Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ↓
训练张量 (C, H, W)
```

#### 2. Val Pipeline
```
原始图像
    ↓
ToTensor()
    ↓
Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ↓
验证张量 (C, H, W)
```

### DataLoader 数据流

#### 1. 批次处理
```
Dataset → DataLoader
    ↓
    ├─ batch_size=64
    ├─ shuffle=True (train) / False (val)
    ├─ num_workers=2
    └─ pin_memory=True
    ↓
Batch Iterator
    ↓
每个 Batch:
    ├─ images: (64, 3, 32, 32) torch.Tensor
    ├─ labels: (64,) torch.Tensor
    └─ 移到设备: images.to(device), labels.to(device)
```

---

## 形状推导数据流

### 详细流程图

```mermaid
flowchart LR
    A[Model IR] --> B[形状推导引擎]
    
    B --> C[解析节点和边]
    C --> D[构建邻接表]
    D --> E[计算入度]
    
    E --> F[拓扑排序]
    
    F --> G[初始化队列]
    G --> H{队列非空?}
    
    H -->|是| I[弹出节点]
    I --> J{节点类型}
    
    J -->|Input| K[获取输入形状]
    J -->|Conv2d| L[应用卷积规则]
    J -->|MaxPool2d| M[应用池化规则]
    J -->|Linear| N[应用全连接规则]
    J -->|Flatten| O[应用展平规则]
    J -->|Add/Concat| P[处理多输入]
    J -->|Output| Q[传递前驱形状]
    
    K --> R[计算输出形状]
    L --> R
    M --> R
    N --> R
    O --> R
    P --> R
    Q --> R
    
    R --> S[更新节点形状]
    S --> T[更新后继节点入度]
    T --> U{后继节点入度为0?}
    
    U -->|是| V[加入队列]
    U -->|否| W[等待其他前驱]
    
    V --> H
    W --> H
    
    H -->|否| X[推导完成]
    
    X --> Y[验证所有节点形状]
    Y --> Z{验证通过?}
    
    Z -->|是| AA[返回形状结果]
    Z -->|否| AB[抛出形状错误]
```

### 形状规则示例

#### 1. Conv2d 形状规则
```python
def conv2d_shape_rule(input_shape: list[int], params: dict) -> list[int]:
    """
    输入: [C_in, H_in, W_in]
    参数: {
        out_channels: C_out,
        kernel_size: K,
        stride: S,
        padding: P,
        dilation: D
    }
    输出: [C_out, H_out, W_out]
    """
    C_in, H_in, W_in = input_shape
    C_out = params['out_channels']
    K = params['kernel_size']
    S = params['stride']
    P = params['padding']
    D = params['dilation']
    
    # 计算输出高度和宽度
    H_out = (H_in + 2 * P - D * (K - 1) - 1) // S + 1
    W_out = (W_in + 2 * P - D * (K - 1) - 1) // S + 1
    
    return [C_out, H_out, W_out]
```

#### 2. MaxPool2d 形状规则
```python
def maxpool2d_shape_rule(input_shape: list[int], params: dict) -> list[int]:
    """
    输入: [C, H_in, W_in]
    参数: {
        kernel_size: K,
        stride: S,
        padding: P
    }
    输出: [C, H_out, W_out]
    """
    C, H_in, W_in = input_shape
    K = params['kernel_size']
    S = params['stride'] if params['stride'] else K
    P = params['padding']
    
    H_out = (H_in + 2 * P - K) // S + 1
    W_out = (W_in + 2 * P - K) // S + 1
    
    return [C, H_out, W_out]
```

#### 3. Linear 形状规则
```python
def linear_shape_rule(input_shape: list[int], params: dict) -> list[int]:
    """
    输入: [in_features] 或 [C, H, W]
    参数: {
        in_features: in_f,
        out_features: out_f
    }
    输出: [out_f]
    """
    in_f = params['in_features']
    out_f = params['out_features']
    
    # 如果输入是 3D，展平为 1D
    if len(input_shape) == 3:
        in_f = input_shape[0] * input_shape[1] * input_shape[2]
    
    return [out_f]
```

#### 4. Flatten 形状规则
```python
def flatten_shape_rule(input_shape: list[int], params: dict) -> list[int]:
    """
    输入: [C, H, W]
    参数: {
        start_dim: start,
        end_dim: end
    }
    输出: [C * H * W]
    """
    start = params.get('start_dim', 1)
    end = params.get('end_dim', -1)
    
    # 计算展平后的维度
    flattened_dim = 1
    for i in range(start, len(input_shape) if end == -1 else end + 1):
        flattened_dim *= input_shape[i]
    
    return [flattened_dim]
```

#### 5. Add 形状规则
```python
def add_shape_rule(input_shapes: list[list[int]], params: dict) -> list[int]:
    """
    输入: [[C, H, W], [C, H, W], ...]
    输出: [C, H, W]
    """
    # 所有输入形状必须相同
    first_shape = input_shapes[0]
    for shape in input_shapes[1:]:
        if shape != first_shape:
            raise ShapeError("Add 操作要求所有输入形状相同")
    
    return first_shape
```

#### 6. Concat 形状规则
```python
def concat_shape_rule(input_shapes: list[list[int]], params: dict) -> list[int]:
    """
    输入: [[C1, H, W], [C2, H, W], ...]
    参数: {
        dim: d  # 默认为 1 (通道维度)
    }
    输出: [C1 + C2 + ..., H, W]
    """
    dim = params.get('dim', 1)
    
    # 检查除拼接维度外其他维度是否相同
    first_shape = input_shapes[0]
    for shape in input_shapes[1:]:
        for i, (s1, s2) in enumerate(zip(first_shape, shape)):
            if i != dim and s1 != s2:
                raise ShapeError(f"Concat 操作要求除维度 {dim} 外其他维度相同")
    
    # 计算拼接后的形状
    output_shape = list(first_shape)
    output_shape[dim] = sum(shape[dim] for shape in input_shapes)
    
    return output_shape
```

---

## 实时监控数据流

### 详细流程图

```mermaid
sequenceDiagram
    autonumber
    participant T as 🏋️ Trainer
    participant C as ⚡ Celery
    participant R as 💾 Redis
    participant S as 🌐 SSE Stream
    participant H as 🎣 useSSE Hook
    participant M as 📊 TrainingMonitor
    participant C1 as 📈 LossChart
    participant U as 👤 用户

    T->>C: 1. Epoch 结束回调
    C->>C: 2. 计算 Epoch 指标
    C->>C: 3. 更新任务状态
    C->>R: 4. task.update_state()
    
    R->>R: 5. 存储任务状态
    R->>S: 6. 触发 SSE 事件
    
    S->>H: 7. 发送 SSE 消息
    Note over H: Event: task-update
    
    H->>H: 8. 解析消息
    H->>M: 9. 更新训练状态
    M->>C1: 10. 更新图表数据
    C1->>C1: 11. 重新渲染曲线
    M->>U: 12. 显示当前指标
    
    Note over T,U: 重复每个 Epoch
    
    T->>C: 13. 训练完成
    C->>R: 14. 更新状态为 COMPLETED
    R->>S: 15. 触发 SSE 事件
    S->>H: 16. 发送完成通知
    H->>M: 17. 更新状态为完成
    M->>U: 18. 显示完成消息
```

### SSE 消息格式

#### 1. 训练进度消息
```json
{
  "event": "task-update",
  "data": {
    "experiment_id": "exp-001",
    "status": "running",
    "current_epoch": 5,
    "total_epochs": 10,
    "train_loss": 0.5234,
    "train_acc": 0.8123,
    "val_loss": 0.6789,
    "val_acc": 0.7856,
    "lr": 0.0005
  }
}
```

#### 2. 训练完成消息
```json
{
  "event": "task-completed",
  "data": {
    "experiment_id": "exp-001",
    "status": "completed",
    "result": {
      "best_val_acc": 0.8234,
      "best_val_loss": 0.5123,
      "best_epoch": 8,
      "total_epochs": 10,
      "best_ckpt_path": "./checkpoints/exp-001/best.pt"
    }
  }
}
```

#### 3. 训练失败消息
```json
{
  "event": "task-failed",
  "data": {
    "experiment_id": "exp-001",
    "status": "failed",
    "error": "CUDA out of memory"
  }
}
```

### 图表数据流

#### 1. LossChart 数据结构
```typescript
interface ChartData {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  lr: number;
}

interface TrainingState {
  currentEpoch: number;
  totalEpochs: number;
  history: ChartData[];
  bestValAcc: number;
  bestEpoch: number;
}
```

#### 2. 图表更新流程
```
新 Epoch 指标
    ↓
添加到 history 数组
    ↓
更新 bestValAcc 和 bestEpoch
    ↓
触发 Recharts 重新渲染
    ↓
显示训练曲线
```

---

## 数据流优化

### 1. 缓存策略

#### 形状推导缓存
```python
# 缓存键：节点ID + 输入形状
cache_key = f"{node_id}:{hash(tuple(input_shape))}"

# 检查缓存
if cache_key in shape_cache:
    return shape_cache[cache_key]

# 计算形状
output_shape = compute_shape(input_shape, params)

# 存入缓存
shape_cache[cache_key] = output_shape
```

#### 代码生成缓存
```python
# 缓存键：Model IR 的 hash
cache_key = f"codegen:{hash(model_ir.model_dump_json())}"

# 检查缓存
if cache_key in codegen_cache:
    return codegen_cache[cache_key]

# 生成代码
code = PyTorchCodeGen(model_ir).generate()

# 存入缓存
codegen_cache[cache_key] = code
```

### 2. 批量处理

#### 批量形状推导
```python
# 一次推导所有节点形状
def batch_infer_shapes(model_ir: ModelIR) -> dict[str, list[int]]:
    shape_map = {}
    for node in model_ir.nodes:
        shape_map[node.id] = infer_node_shape(node, shape_map)
    return shape_map
```

#### 批量代码生成
```python
# 一次生成多个模型的代码
def batch_codegen(model_irs: list[ModelIR]) -> dict[str, str]:
    code_map = {}
    for ir in model_irs:
        code_map[ir.id] = PyTorchCodeGen(ir).generate()
    return code_map
```

### 3. 异步处理

#### 异步形状推导
```python
@celery_app.task(name="tasks.shape_inference.batch_infer")
def batch_infer_shapes_async(model_ir_ids: list[str]) -> dict[str, dict]:
    results = {}
    for ir_id in model_ir_ids:
        model_ir = get_model_ir(ir_id)
        shapes = infer_shapes(model_ir)
        results[ir_id] = shapes
    return results
```

#### 异步代码生成
```python
@celery_app.task(name="tasks.codegen.batch_generate")
def batch_codegen_async(model_ir_ids: list[str]) -> dict[str, str]:
    results = {}
    for ir_id in model_ir_ids:
        model_ir = get_model_ir(ir_id)
        code = PyTorchCodeGen(model_ir).generate()
        results[ir_id] = code
    return results
```

---

## 错误处理数据流

### 1. 形状推导错误
```
形状推导失败
    ↓
捕获 ShapeError
    ↓
返回错误信息
    ↓
前端显示错误提示
    ↓
用户修正模型结构
```

### 2. 训练错误
```
训练过程出错
    ↓
捕获异常
    ↓
记录错误日志
    ↓
更新任务状态为 FAILED
    ↓
SSE 推送错误通知
    ↓
前端显示错误消息
```

### 3. 数据加载错误
```
数据集加载失败
    ↓
捕获 DatasetError
    ↓
返回错误信息
    ↓
前端显示错误提示
    ↓
用户修正数据配置
```

---

## 性能监控数据流

### 1. 训练性能指标
```
每个 Epoch 记录:
    ├─ 训练时间
    ├─ 验证时间
    ├─ 数据加载时间
    ├─ 前向传播时间
    ├─ 反向传播时间
    └─ 内存使用量
```

### 2. 系统性能指标
```
实时监控:
    ├─ CPU 使用率
    ├─ GPU 使用率
    ├─ 内存使用量
    ├─ 磁盘 I/O
    └─ 网络流量
```

---

**最后更新**：2025-04-09