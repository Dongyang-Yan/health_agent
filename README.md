# ST-GCN Golf - 智能高尔夫挥杆分析系统

## 📋 业务简介

**适用对象**：高尔夫教练、培训机构、运动分析专业人员

**解决痛点**：传统高尔夫教学依赖教练肉眼观察，难以量化分析学员挥杆动作，无法精准定位技术问题。

**核心价值**：本系统基于深度学习技术，自动分析高尔夫挥杆视频，识别 8 个关键挥杆阶段，检测 41 种常见技术问题，提供量化评分和改进建议，大幅提升教学效率和准确性。

**产出结果**：
- 📊 **量化评分报告**：满分 80 分的详细评分（每个阶段 10 分）
- 🎥 **问题诊断视频**：标注骨骼关键点的问题片段视频
- 📝 **详细分析日志**：包含各阶段问题检测结果和置信度分数

---

## ⚡ 极速上手（3 步运行，需配置环境。）

> **重要提示**：本项目必须自备预训练模型权重文件，无权重文件无法运行。请确保已获取以下权重文件：
> - `epoch70_model_best.pt`（关节模型）
> - `epoch100_model_bone.pt`（骨骼模型）
> - `epoch25_model.pt`（问题检测模型 1）
> - `epoch30_model.pt`（问题检测模型 2）

```bash
# 第 1 步：启动服务（默认配置）
bash inference/start_services.sh 6

# 第 2 步：放置测试视频到 inference/test_main/ 目录（按视角+球杆类型分类）

# 第 3 步：执行批量测试
python inference/test_batch_flexible.py 6
```

**输出位置**：
- 问题视频：`./problem_clips/`
- 分析日志：`./logs/`
- 测试结果：当前目录下的 `test_*.json` 文件

---

## 🌟 主要功能

- **实时姿态估计**：使用 MMPose 提取人体 17 个骨骼关键点
- **事件帧检测**：通过 ST-GCN 模型自动识别挥杆过程中的 8 个关键阶段
- **问题诊断**：双模型集成（关节 + 骨骼）检测 41 种常见技术问题
- **智能评分**：对 8 个挥杆阶段进行量化评分（满分 80 分）
- **视频生成**：自动生成带有骨骼标注的问题片段视频

---

## 🔧 环境安装

### ⚠️ 重要前置说明

**系统要求**：
- ✅ **操作系统**：Linux
- ✅ **GPU**：NVIDIA GPU，支持 CUDA 11.7+
- ✅ **Python**：3.9
- ✅ **CUDA**：11.7+
- ✅ **内存**：256GB+ RAM
- ✅ **FFmpeg**：4.4（固定版本，用于 GPU 加速解码）

**必需文件**：
- 预训练模型权重文件

### 安装步骤

#### 1. 进入项目目录

```bash
cd st-gcn-golf
```

#### 2. 创建虚拟环境

```bash
conda create -n stgcn python=3.9 -y
conda activate stgcn
```

#### 3. 安装 FFmpeg（固定版本 4.4）

```bash
conda install -c conda-forge ffmpeg=4.4 -y
```

> **注意**：FFmpeg 4.4 是 GPU 加速视频解码的必要依赖，必须使用此版本。

#### 4. 安装 chumpy（优先安装）

```bash
pip install chumpy --no-isolate
```

> **注意**：chumpy 必须单独安装，避免后续依赖冲突。

#### 5. 安装主要依赖

```bash
pip install -r requirements.txt
```

> **重要**：如果 mmcv 安装失败，使用以下备选方案：
>
> **方案 1**：复制项目提供的 mmcv 文件夹到 python site-packages
> ```bash
> cp -r mmcv $CONDA_PREFIX/lib/python3.9/site-packages/
> ```


#### 6. 验证安装

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mmpose; print(f'MMPose version: {mmpose.__version__}')"
```

---

## 🚀 启动服务

### 默认配置

```bash
bash inference/start_services.sh
```

这将在端口 8000-8003 上启动 4 个服务（每个 GPU 一个）。

### 自定义配置

```bash
# 每 GPU 启动 6 个服务（推荐配置）
bash inference/start_services.sh 6

# 完整参数自定义
bash inference/start_services.sh <每GPU服务数> <起始端口> <批处理大小> <检测间隔>

# 示例：每 GPU 3 个服务，端口 8000，批处理 32，检测间隔 5
bash inference/start_services.sh 3 8000 32 5
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| SERVICES_PER_GPU | 每个 GPU 上的服务数量 | 1 |
| START_PORT | 起始端口号 | 8000 |
| BATCH_SIZE | MMPose 批处理大小 | 64 |
| DET_INTERVAL | 检测器间隔 | 10 |

### 健康检查（可选）

```bash
curl http://localhost:8000/health
```

预期响应：
```json
{
  "status": "ok",
  "gpu_id": 0,
  "device": "cuda:0",
  "cuda_available": true,
  "cuda_device_count": 4
}
```

### 停止服务

```bash
bash inference/stop_all.sh
```

---

## 📁 项目结构

```
st-gcn-golf/
├── inference/                    # 主推理服务目录
│   ├── checkpoint/              # 预训练模型权重
│   │   ├── golf/               # 高尔夫专用模型
│   │   └── kinetics_skeleton/  # Kinetics 预训练模型
│   ├── feeder/                 # 数据加载器
│   ├── net/                    # 网络模型定义
│   ├── start_services.sh       # 多 GPU 服务启动脚本
│   ├── stop_all.sh            # 停止所有服务脚本
│   ├── serve_single_gpu.py    # 单 GPU FastAPI 服务
│   ├── test_batch_flexible.py # 灵活批量测试脚本
│   ├── mmpose_exact_copy_optimized.py  # 优化版 MMPose 推理
│   ├── problem_inference.py   # 问题检测推理
│   └── ensemble_inference.py  # 结果集成
├── models/                     # 额外模型文件
├── net/                        # 网络模型
├── work_dir/                   # 工作目录
└── requirements.txt           # Python 依赖
```

## 📖 使用指南

### ⚠️ 小程序调用说明

**重要提示**：当前配置仅用于本地测试，实际部署到微信小程序时需要修改服务地址。

#### 本地测试（当前配置）
- 服务地址：`http://localhost:8000`
- 适用于：本机开发和测试

#### 实际部署（需修改配置）

当部署到服务器供小程序调用时，需要修改以下内容：

1. **修改启动脚本中的监听地址**
   
   编辑 `inference/serve_single_gpu.py`，将：
   ```python
   uvicorn.run(app, host="0.0.0.0", port=args.port)
   ```
   确保 `host="0.0.0.0"` 以允许外部访问。

2. **配置防火墙和安全组**
   - 开放对应端口（默认 8000-8023，取决于启动的服务数量）
   - 云服务器需在安全组中放行相应端口

3. **微信小程序端配置**
   
   将小程序中的 API 地址从 `localhost` 改为服务器实际 IP 或域名


> **总结**：当前配置是测试环境，实际使用时需要根据部署环境修改服务地址、配置 HTTPS，并在小程序后台添加合法域名。

---

### 方式一：批量测试

#### 1. 准备测试视频

将视频按**视角 + 球杆类型**分类放置到 `inference/test_main/` 目录：

```
inference/test_main/
├── 木杆正面/          # 木杆 + 正面视角
│   ├── sample1/
│   │   └── video.mp4
│   └── sample2/
│       └── video.mp4
├── 木杆侧面/          # 木杆 + 侧面视角
├── 铁杆正面/          # 铁杆 + 正面视角
└── 铁杆侧面/          # 铁杆 + 侧面视角
```

> **说明**：程序会根据目录名自动识别视角和球杆类型，无需手动配置参数。

#### 2. 执行批量测试

```bash
# 默认配置（每 GPU 6 个服务）
python inference/test_batch_flexible.py 6

# 指定服务数量（需与启动命令匹配，如启动了 6 个服务/GPU）
python inference/test_batch_flexible.py 6
```

### 方式二：单个视频推理

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/golf_video.mp4",
    "view": 1,
    "club": 1
  }'
```

**请求参数**：

| 参数 | 说明 | 取值 |
|------|------|------|
| video_path | 视频文件路径 | 绝对路径 |
| view | 视角类型 | 0 = 正面，1 = 侧面 |
| club | 球杆类型 | 0 = 铁杆，1 = 木杆 |

---

## 📊 输出成果说明

### 通俗版说明

运行完成后，您将获得以下三类成果：

1. **🎥 问题诊断视频**（`./problem_clips/`）
   - 自动截取包含技术问题的挥杆片段
   - 视频中叠加显示骨骼关键点和连线
   - 标注问题发生的阶段和问题编号
   - **文件名示例**：假设测试视频为 `video.mp4`，检测到第 15 号问题发生在第 120 帧，则输出文件名为：
     ```
     ./problem_clips/P_video_Q15_frame120_with_skeleton.mp4
     ```
     其中：
     - `P_`：表示问题诊断视频前缀
     - `video`：原始视频文件名（不含扩展名）
     - `Q15`：问题 ID（1-41）
     - `frame120`：问题发生的帧号
     - `_with_skeleton`：表示包含骨骼标注

2. **📝 详细分析日志**（`./logs/`）
   - 记录每个挥杆阶段检测到的问题
   - 包含模型预测的置信度分数
   - 提供 8 个阶段的详细评分（满分 80 分）
   - **文件名示例**：假设测试视频为 `video.mp4`，则对应的日志文件为：
     ```
     ./logs/video.txt
     ```
     其中：
     - 日志文件名与原始视频文件名一致（不含扩展名）
     - 使用 `.txt` 格式存储文本日志

3. **📈 批量测试结果**（`test_*.json`）
   - 汇总所有视频的测试结果
   - 包含每个视频的处理时间和性能指标
   - 统计成功率和平均吞吐量
   - **文件名示例**：
     ```
     test_batch_results_20260512_143022.json
     ```
     其中：
     - `test_batch_results_`：固定前缀
     - `20260512_143022`：测试执行时间戳（年月日_时分秒）

### 终端收到请求后的响应格式（技术参考）

```json
{
  "best_qid": 15,
  "best_phase": 3,
  "video": "./problem_clips/P_video_Q15_frame120_with_skeleton.mp4",
  "log": "logs/video_name.txt",
  "gpu_id": 0,
  "frame_count": 150,
  "video_width": 1920,
  "video_height": 1080
}
```

**字段说明**：

| 字段 | 说明 |
|------|------|
| best_qid | 最严重问题 ID（1-41） |
| best_phase | 问题发生阶段（1-8） |
| video | 问题视频路径 |
| log | 日志文件路径 |
| mmpose_time | 姿态估计耗时（秒） |
| stgcn_time | ST-GCN 推理耗时（秒） |
| total_time | 总处理时间（秒） |
| gpu_id | 使用的 GPU ID |
| frame_count | 视频帧数 |
| video_width/height | 视频分辨率 |

---

## ⚙️ 高级配置

### 模型权重路径配置

在 `inference/serve_single_gpu.py` 中修改权重文件路径：

```python
joint_weight_path = "/path/to/epoch70_model_best.pt"      # 关节模型
bone_weight_path = "/path/to/epoch100_model_bone.pt"      # 骨骼模型
problem_weight_path = "/path/to/epoch25_model.pt"         # 问题检测模型 1
problem_weight_path2 = "/path/to/epoch30_model.pt"        # 问题检测模型 2
```

> **重要**：必须确保权重文件路径正确，否则服务无法启动。

### 性能调优参数

| 参数 | 作用 | 调整建议 |
|------|------|----------|
| BATCH_SIZE | MMPose 批处理大小 | 增加可提高吞吐量，但占用更多显存 |
| DET_INTERVAL | 检测器间隔 | 增加可减少负载，但可能影响精度 |
| SERVICES_PER_GPU | 每 GPU 服务数 | 根据显存容量调整 |

### 推荐配置表

| GPU 显存 | 每 GPU 服务数 | 批处理大小 | 检测间隔 |
|----------|---------------|------------|----------|
| 8 GB     | 1             | 32         | 15       |
| 12 GB    | 1             | 64         | 10       |
| 16 GB    | 2             | 64         | 10       |
| 24 GB+   | 6             | 64         | 10       |
| 32 GB+   | 6-8           | 64         | 10       |

---



## ❓ 常见问题

### 1. 服务启动失败

**排查步骤**：

```bash
# 检查 GPU 状态
nvidia-smi

# 查看服务日志
tail -f logs/gpu0_service1.log

# 检查端口是否被占用
netstat -tuln | grep 8000
```

**可能原因**：
- GPU 驱动未正确安装
- CUDA 版本不匹配（需要 11.7）
- 端口已被其他进程占用

### 2. 端口已被占用

**解决方案**：

```bash
# 方案 1：更改起始端口
bash inference/start_services.sh 6 9000

# 方案 2：停止现有服务后重新启动
bash inference/stop_all.sh
bash inference/start_services.sh 6
```

### 3. mmcv 安装失败

**解决方案**：

```bash
# 方案 1：复制项目提供的 mmcv 文件夹
cp -r mmcv $CONDA_PREFIX/lib/python3.9/site-packages/

# 方案 2：使用 wheel 文件安装
pip install mmcv-*.whl

# 方案 3：从 OpenMMLab 官方源安装
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```

### 4. 显存溢出（CUDA Out of Memory）

**解决方案**：

```bash
# 减少每 GPU 服务数
bash inference/start_services.sh 1 8000 32 15

# 或减小批处理大小
bash inference/start_services.sh 6 8000 32 10
```

### 5. FFmpeg 版本不匹配

**检查版本**：

```bash
ffmpeg -version
```

**重新安装**：

```bash
conda install -c conda-forge ffmpeg=4.4 -y
```

### 6. 权重文件路径错误

**症状**：服务启动时报错 "File not found" 或 "Missing keys"

**解决方案**：
- 检查 `inference/serve_single_gpu.py` 中的权重路径配置
- 确保权重文件存在于指定路径
- 使用绝对路径而非相对路径

---


## 🔍 架构详情

### 推理流程

```
视频输入
    ↓
MMPose 姿态估计 (GPU)
    ↓
骨骼数据处理
    ↓
ST-GCN 事件帧检测
    ├─ 关节模型
    └─ 骨骼模型
    ↓
问题检测 (双模型)
    ├─ 骨骼 + 方向模型
    └─ 关节 + 方向模型
    ↓
结果集成
    ↓
评分 & 视频生成
    ↓
输出 (JSON + 视频 + 日志)
```

### 模型集成策略

系统采用双模型集成方法：
1. **骨骼 + 方向模型**：关注身体几何形状和球杆方向
2. **关节 + 方向模型**：分析关节角度和运动模式

最终预测基于置信度分数的加权投票组合。



**注意**：使用前请确保已获得必要的模型权重文件，并根据实际硬件配置调整服务参数。
