# LabelMatrix Backend

基于 Ultralytics YOLO 的遥感影像智能处理平台后端模块。

## 功能特性

- 支持多种深度学习任务：
  - 目标检测 (Object Detection)
  - 实例分割 (Instance Segmentation)
  - 图像分类 (Image Classification)
  - 旋转目标检测 (Oriented Bounding Box)

- 支持多种运行模式：
  - 训练 (Train)
  - 推理 (Predict)
  - 继续训练 (Resume)

- 内置HTTP服务器，支持实时状态查询和控制

## 项目结构

```
labelmatrix/
├── __init__.py              # 模块初始化
├── config/                  # 配置模块
│   ├── parser.py           # 配置解析器
│   ├── validator.py        # 配置验证器
│   └── schemas.py          # 配置数据模型
├── engines/                 # 引擎模块
│   ├── base_engine.py      # 抽象基类
│   └── ultralytics_engine.py  # Ultralytics引擎实现
├── server/                  # HTTP服务器
│   └── app.py              # Flask应用
├── state/                   # 状态管理
│   ├── manager.py          # 状态管理器
│   └── models.py           # 状态数据模型
├── handlers/                # 结果处理器
│   ├── tile_processor.py   # 影像分块处理
│   └── result_converter.py # 结果格式转换
├── utils/                   # 工具模块
│   ├── logger.py           # 日志工具
│   ├── gpu_checker.py      # GPU检查工具
│   └── metadata_builder.py # 元数据构建器
└── exceptions/              # 异常定义
    ├── base.py             # 基础异常
    ├── config_errors.py    # 配置错误
    ├── engine_errors.py    # 引擎错误
    └── resource_errors.py  # 资源错误

labelmatrix_worker.py        # 主入口程序
examples/                    # 示例配置文件
environment.yml              # Conda环境配置
```

## 安装

### 使用 Conda（推荐）

```bash
# 创建环境
conda env create -f labelmatrix/environment.yml

# 激活环境
conda activate labelmatrix
```

### 使用 pip

```bash
pip install ultralytics flask flask-cors pyyaml opencv-python scikit-image pillow shapely
```

## 快速开始

### 1. 准备配置文件

参考 `examples/` 目录中的示例配置文件：

```yaml
# train_config.yaml
task_id: "train_building_001"
task_type: "detect"
model_architecture: "yolov8n.pt"
output_dir: "./output"
mode: "train"
data_config: "./datasets/building/data.yaml"
hyperparameters:
  epochs: 100
  batch: 16
  lr0: 0.001
hardware:
  device: "cuda:0"
  workers: 8
```

### 2. 运行训练

```bash
python labelmatrix_worker.py --config train_config.yaml
```

### 3. 查看状态

Worker 启动后会打印端口号：
```
PORT:12345
```

可以通过 HTTP API 查询状态：
```bash
# 获取状态
curl http://127.0.0.1:12345/status

# 停止任务
curl -X POST http://127.0.0.1:12345/stop
```

## 配置说明

### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务唯一标识 |
| `task_type` | string | 任务类型：detect, segment, classify, obb |
| `model_architecture` | string | 预训练模型名称或路径 |
| `output_dir` | string | 输出目录 |

### 可选字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | train | 运行模式：train, predict, resume |
| `data_config` | string | - | 数据集配置文件路径（训练时必需） |
| `resume_from` | string | - | 检查点路径（resume模式必需） |
| `model_path` | string | - | 模型路径（predict模式必需） |

### 超参数配置 (hyperparameters)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | int | 100 | 训练轮数 |
| `batch` | int | 16 | 批次大小 |
| `lr0` | float | 0.001 | 初始学习率 |
| `optimizer` | string | Adam | 优化器 |
| `weight_decay` | float | 0.0005 | 权重衰减 |
| `imgsz` | int | 640 | 输入尺寸 |
| `amp` | bool | True | 混合精度训练 |

### 硬件配置 (hardware)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | string | cuda:0 | 设备：cuda:N 或 cpu |
| `workers` | int | 8 | 数据加载线程数 |
| `min_memory_mb` | int | - | 最小显存要求 |

### 推理配置 (predict)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source` | string | - | 输入影像路径 |
| `conf_thres` | float | 0.5 | 置信度阈值 |
| `iou_thres` | float | 0.45 | IoU阈值 |
| `augment` | bool | False | 测试时增强 |
| `half` | bool | True | 半精度推理 |
| `save_format` | string | geojson | 结果格式 |
| `tile_size` | int | 512 | 分块大小 |
| `tile_overlap` | float | 0.1 | 分块重叠 |

## HTTP API

### GET /status

获取任务状态

响应示例：
```json
{
  "success": true,
  "data": {
    "task_id": "train_building_001",
    "status": "running",
    "progress": 45,
    "metrics": {
      "mAP50": 0.75,
      "mAP50-95": 0.65
    },
    "error": null
  }
}
```

### POST /stop

停止当前任务

响应示例：
```json
{
  "success": true,
  "message": "Stop signal sent"
}
```

### GET /health

健康检查

响应示例：
```json
{
  "success": true,
  "status": "running",
  "task_id": "train_building_001"
}
```

### GET /metrics

获取训练指标

响应示例：
```json
{
  "success": true,
  "data": {
    "metrics": {...},
    "progress": 45,
    "status": "running"
  }
}
```

## 输出结构

```
{output_dir}/
└── {task_id}/
    ├── config.yaml          # 配置文件副本
    ├── state.json           # 任务状态
    ├── metadata.json        # 训练元数据
    ├── logs/
    │   └── worker.log       # 运行日志
    ├── train/               # 训练输出
    │   └── weights/
    │       ├── best.pt      # 最佳模型
    │       └── last.pt      # 最后一轮模型
    └── results/             # 推理结果
        └── predict/
```

## 测试

运行基本功能测试：

```bash
python labelmatrix/examples/test_basic.py
```

