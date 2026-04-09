# NoCode PyTorch Platform - 快速开始指南

## 🚀 5 分钟快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- Redis 6+
- (可选) CUDA 11.8+ (用于 GPU 训练)

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/Torch_GUI.git
cd Torch_GUI
```

### 2. 安装依赖

#### 后端
```bash
cd backend
pip install -r requirements.txt
```

#### 前端
```bash
cd frontend
npm install
```

### 3. 启动服务

#### 启动 Redis
```bash
redis-server
```

#### 启动后端
```bash
# 终端 1: FastAPI
cd backend
uvicorn api.main:app --reload --port 8000

# 终端 2: Celery Worker
cd backend
celery -A tasks.celery_app worker --loglevel=info
```

#### 启动前端
```bash
# 终端 3: Vite Dev Server
cd frontend
npm run dev
```

### 4. 访问应用

- **前端**：http://localhost:5173
- **API 文档**：http://localhost:8000/api/docs
- **健康检查**：http://localhost:8000/health

---

## 📚 核心概念

### IR (Intermediate Representation)

IR 是系统的核心数据结构，分为三种类型：

#### 1. Model IR
描述神经网络结构，包含：
- **节点**：表示神经网络层（Conv2d, Linear, ReLU 等）
- **边**：表示数据流连接
- **参数**：每层的配置参数

#### 2. Data IR
描述数据处理流程，包含：
- **数据源**：torchvision 数据集、本地文件夹等
- **切分策略**：train/val/test 划分
- **Transform Pipeline**：数据增强和预处理
- **DataLoader 配置**：batch_size, num_workers 等

#### 3. Experiment IR
描述训练实验，包含：
- **超参数**：epochs, learning_rate, optimizer 等
- **后端配置**：device (cpu/cuda), 并行设置
- **检查点配置**：保存路径、保存策略

---

## 🎨 创建第一个模型

### 步骤 1：打开画布

访问 http://localhost:5173，你会看到：

```
┌─────────────────────────────────────────────┐
│  节点面板    │      画布区域      │
│              │                     │
│  [Input]     │                     │
│  [Conv2d]    │                     │
│  [ReLU]      │                     │
│  [Linear]    │                     │
│  [Output]    │                     │
└─────────────────────────────────────────────┘
```

### 步骤 2：拖拽节点

1. 从左侧节点面板拖拽 **Input** 节点到画布
2. 拖拽 **Conv2d** 节点
3. 拖拽 **ReLU** 节点
4. 拖拽 **Linear** 节点
5. 拖拽 **Output** 节点

### 步骤 3：连接节点

1. 点击 **Input** 节点的右侧连接点
2. 拖动到 **Conv2d** 节点的左侧连接点
3. 依次连接：Conv2d → ReLU → Linear → Output

### 步骤 4：配置参数

1. 点击 **Conv2d** 节点
2. 在右侧配置面板中设置：
   - `in_channels`: 3
   - `out_channels`: 16
   - `kernel_size`: 3
   - `padding`: 1

3. 点击 **Linear** 节点
4. 配置：
   - `in_features`: 4096 (根据形状推导自动计算)
   - `out_features`: 10

### 步骤 5：查看形状

系统会自动推导每层的形状：

```
Input:     [3, 32, 32]
Conv2d:    [16, 32, 32]
ReLU:       [16, 32, 32]
Flatten:    [16384]
Linear:     [10]
Output:     [10]
```

### 步骤 6：生成代码

点击"生成代码"按钮，查看生成的 PyTorch 代码：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(16384, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
```

### 步骤 7：保存模型

点击"保存模型"按钮，模型 IR 会被保存到后端存储。

---

## 📊 配置数据集

### 步骤 1：选择数据源

在数据配置页面选择数据源类型：

- **Torchvision**：内置数据集（CIFAR10, MNIST, FashionMNIST）
- **Local Folder**：本地 ImageFolder 格式数据
- **CSV**：CSV 文件格式的数据

### 步骤 2：配置 CIFAR10

```json
{
  "source": {
    "type": "torchvision",
    "dataset_name": "CIFAR10",
    "download_root": "./data"
  },
  "schema": {
    "task_type": "image_classification",
    "num_classes": 10
  },
  "split": {
    "strategy": "ratio",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1
  }
}
```

### 步骤 3：配置 Transform Pipeline

#### Train Pipeline
```json
{
  "transforms": [
    {"op_type": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"op_type": "RandomCrop", "params": {"size": 32, "padding": 4}},
    {"op_type": "ToTensor"},
    {"op_type": "Normalize", "params": {
      "mean": [0.4914, 0.4822, 0.4465],
      "std": [0.2470, 0.2435, 0.2616]
    }}
  ]
}
```

#### Val Pipeline
```json
{
  "transforms": [
    {"op_type": "ToTensor"},
    {"op_type": "Normalize", "params": {
      "mean": [0.4914, 0.4822, 0.4465],
      "std": [0.2470, 0.2435, 0.2616]
    }}
  ]
}
```

---

## 🏋️ 启动训练

### 步骤 1：创建实验

```json
{
  "id": "exp-001",
  "name": "CIFAR10 Training",
  "model_ir_id": "model-001",
  "data_ir_id": "data-001",
  "hyper_params": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_fn": "CrossEntropyLoss",
    "scheduler": "StepLR",
    "scheduler_params": {
      "step_size": 5,
      "gamma": 0.1
    }
  },
  "backend": {
    "type": "local",
    "device": "cuda"
  },
  "checkpoint": {
    "enabled": true,
    "save_dir": "./checkpoints",
    "save_best_only": true
  }
}
```

### 步骤 2：提交训练

点击"开始训练"按钮，系统会：

1. 保存 Experiment IR
2. 提交 Celery 训练任务
3. 返回任务 ID

### 步骤 3：监控训练

训练监控页面会实时显示：

- **当前 Epoch**：1/10
- **训练损失**：2.1234
- **验证准确率**：0.3456
- **学习率**：0.001
- **损失曲线**：实时更新的图表

### 步骤 4：查看结果

训练完成后，会显示：

```
✅ 训练完成
最佳验证准确率：0.8234 @ Epoch 8
最佳检查点：./checkpoints/exp-001/best.pt
总训练时间：15.2 分钟
```

---

## 🔧 常见任务

### 添加自定义算子

#### 后端

1. 在 `model_ir.py` 中添加 OpType
```python
class OpType(str, Enum):
    # ... 现有算子
    CUSTOM_OP = "CustomOp"
```

2. 在 `node_registry.py` 中注册
```python
@register(OpType.CUSTOM_OP)
def _custom_op_builder(node: IRNode) -> tuple[str, str]:
    p = node.params
    init = f"nn.CustomOp({p['param1']}, {p['param2']})"
    return init, ""
```

3. 在 `op_rules.py` 中添加形状规则
```python
def custom_op_shape_rule(input_shape: list[int], params: dict) -> list[int]:
    # 实现形状推导逻辑
    return output_shape
```

#### 前端

1. 在 `nodeRegistry.ts` 中添加元数据
```typescript
[OpType.CUSTOM_OP]: {
  op_type: OpType.CUSTOM_OP,
  label: "CustomOp",
  category: "custom",
  defaultParams: { param1: 1, param2: 2 },
  inputs: 1,
  outputs: 1,
}
```

### 调试形状推导

如果形状推导失败：

1. 检查节点连接是否正确
2. 查看浏览器控制台的错误信息
3. 检查 `backend/core/shape_inference/op_rules.py` 中的形状规则
4. 使用 API 文档中的 `/api/v1/shape-inference` 端点测试

### 导出训练好的模型

训练完成后，检查点文件保存在：

```
./checkpoints/{experiment_id}/
├── best.pt          # 最佳模型
├── last.pt          # 最后一个 epoch
└── history.json     # 训练历史
```

加载模型：

```python
import torch
from core.ir.codegen import PyTorchCodeGen

# 生成模型代码
codegen = PyTorchCodeGen(model_ir)
code = codegen.generate()

# 执行代码
namespace = {}
exec(code, namespace)
model_cls = namespace["MyModel"]

# 加载检查点
model = model_cls()
checkpoint = torch.load("./checkpoints/exp-001/best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 🐛 故障排除

### 问题 1：前端无法连接后端

**症状**：浏览器控制台显示 "Network Error"

**解决方案**：
1. 检查后端是否运行：`curl http://localhost:8000/health`
2. 检查 CORS 配置：`backend/api/middleware.py`
3. 检查前端代理配置：`frontend/vite.config.ts`

### 问题 2：Celery 任务不执行

**症状**：训练状态一直显示 "pending"

**解决方案**：
1. 检查 Redis 是否运行：`redis-cli ping`
2. 检查 Celery Worker 是否运行：查看终端日志
3. 检查 Celery 配置：`backend/tasks/celery_app.py`

### 问题 3：CUDA out of memory

**症状**：训练失败，显示 "CUDA out of memory"

**解决方案**：
1. 减小 `batch_size`
2. 减小模型参数（如 `out_channels`）
3. 使用梯度累积
4. 使用混合精度训练（`use_amp: true`）

### 问题 4：数据集下载失败

**症状**：数据集下载超时或失败

**解决方案**：
1. 检查网络连接
2. 手动下载数据集到指定目录
3. 使用本地文件夹作为数据源

---

## 📖 下一步

- 阅读 [完整文档](README.md)
- 查看 [数据流文档](DATAFLOW.md)
- 了解 [系统架构](ARCHITECTURE)
- 查看 [API 文档](http://localhost:8000/api/docs)

---

## 💡 提示

1. **形状推导**：系统会自动推导形状，如果形状不匹配会显示错误
2. **实时保存**：建议定期保存模型，避免丢失工作
3. **GPU 加速**：如果有 GPU，设置 `device: "cuda"` 可以大幅加速训练
4. **批量大小**：根据 GPU 内存调整 `batch_size`，通常 32-128
5. **学习率**：从 0.001 开始，根据训练情况调整

---

**需要帮助？**

- 查看 [常见问题](README.md#常见问题)
- 提交 [GitHub Issue](https://github.com/your-repo/issues)
- 联系：support@example.com

---

**最后更新**：2025-04-09