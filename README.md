# ModelTrainDemo - 机器学习训练框架

## 📖 项目简介

这是一个用于训练CTR（点击率预估）模型的完整框架，支持深度学习模型的训练、评估和部署。本项目采用模块化设计，配置灵活，适合快速迭代和实验不同的模型方案。

### 主要特点
- ✅ 支持多种深度学习模型（DeepFM、MLP、WideAndDeep）
- ✅ 灵活的特征处理（支持id、类别、连续特征）
- ✅ 自动化模型版本管理
- ✅ 完整的推理和评估流程
- ✅ 配置文件驱动，无需修改代码

---

## 🚀 快速开始

### 1. 环境搭建

```bash
# 安装依赖
pip install -r requirements.txt
```

依赖项包括：
- torch >= 1.9.0（深度学习框架）
- pandas >= 1.3.0（数据处理）
- numpy >= 1.21.0（数值计算）
- scikit-learn >= 1.0.0（机器学习工具）

### 2. 项目结构

```
ModelTrainDemo/
├── data/                          # 数据目录
│   ├── train.csv                  # 训练数据
│   ├── validation.csv             # 验证数据
│   ├── test.csv                   # 测试数据
│   └── generate_data.py          # 数据生成脚本
├── train_configs/                 # 训练配置目录
│   └── v1/                       # v1版本配置
│       ├── train_config.json       # 模型和训练参数配置
│       └── features_process.json   # 特征处理配置
├── trained_models_dir/            # 训练模型保存目录
│   └── v1_20260226_172330/      # 版本号_时间戳
│       ├── configs/              # 备份的配置文件
│       ├── models/               # 模型checkpoint
│       │   ├── best_model.pth     # 最佳模型
│       │   └── epoch_X.pth       # 各个epoch的模型
│       ├── feature_processor.json # 特征处理器
│       └── training_history.json # 训练历史
├── TrainFramework/                # 核心框架代码
│   ├── models/                   # 模型定义
│   │   ├── deepfm.py            # DeepFM、MLP、WideAndDeep模型
│   │   └── __init__.py
│   ├── preprocess.py             # 特征处理
│   ├── train_pipeline.py         # 训练流程
│   └── tools.py                 # 工具函数
├── run_train.py                  # 训练入口脚本
├── predictor.py                  # 模型推理脚本
├── evaluate_model.py             # 模型评估脚本
└── requirements.txt              # 依赖列表
```

### 3. 数据格式说明

CSV文件格式如下：

```
id,features,label
2001,"{\"user_id\": 3001, \"item_id\": 4001, \"age\": 35, \"income\": 48483, \"cate_id\": \"101\", \"brand\": \"BrandB\", \"gender\": \"M\"}",0
```

- `id`: 样本ID
- `features`: JSON格式的特征字典
- `label`: 标签（0或1）

---

## 🎯 模型训练的版本控制

### 版本号命名规则

每个训练实验会自动生成一个带时间戳的版本号：
```
配置名_时间戳
例如：v1_20260226_172330
```

### 如何创建新版本

**步骤1：在 `train_configs/` 下创建新的配置文件夹**

```bash
# 复制现有配置作为模板
cp -r train_configs/v1 train_configs/v2
```

**步骤2：修改 `run_train.py` 中的配置名称**

```python
# run_train.py 文件第177-179行
TRAIN_CONFIG = {
    'config_name': 'v2',  # 修改为你的新版本名
    'configs_root': './train_configs'
}
```

**步骤3：开始训练**

```bash
python run_train.py
```

训练完成后，模型会自动保存到 `trained_models_dir/v2_时间戳/` 目录下。

### 版本管理最佳实践

| 场景 | 建议的版本命名 | 示例 |
|------|---------------|------|
| 基础实验 | v1, v2, v3... | v1, v2 |
| 特征实验 | feat_xxx | feat_age_norm |
| 模型实验 | model_xxx | model_deepfm_v2 |
| 超参调优 | hp_xxx | hp_lr_001 |
| 数据实验 | data_xxx | data_sample_100k |

### 查看已训练的模型版本

```bash
# 查看所有训练版本
ls trained_models_dir/

# 查看某个版本的详细信息
cat trained_models_dir/v1_20260226_172330/metadata.json
```

---

## 🔧 模型选择

### 支持的模型类型

框架目前支持三种模型：

| 模型名称 | 适合场景 | 特点 |
|---------|---------|------|
| **DeepFM** | 通用CTR场景 | 结合了FM的线性特征和DNN的非线性能力，表现稳定 |
| **MLP** | 特征关系简单 | 纯神经网络，适合特征交互较少的场景 |
| **WideAndDeep** | 需要记忆和泛化 | 结合线性模型和深度学习的优点 |

### 如何切换模型

**步骤1：打开 `train_configs/v1/train_config.json`**

**步骤2：修改 `model_name` 字段**

```json
{
  "model_config": {
    "model_name": "DeepFM",  // 改为 "MLP" 或 "WideAndDeep"
    ...
  }
}
```

**步骤3：开始训练**

```bash
python run_train.py
```

### 模型对比建议

如果你不确定用哪个模型，可以按以下步骤实验：

1. **先用DeepFM跑一个baseline**
2. **尝试MLP，对比性能**
3. **根据结果选择最优模型**
4. **记录实验结果，便于后续参考**

---

## ⚙️ 模型参数修改

### 1. 模型结构参数

位置：`train_configs/v1/train_config.json` → `model_config`

```json
{
  "model_config": {
    "model_name": "DeepFM",
    "embedding_dim": 8,              // Embedding维度（推荐：8、16、32）
    "hidden_units": [64, 32, 16],   // Deep部分的隐藏层节点数
    "dropout_rate": 0.2              // Dropout率（0-1之间，防止过拟合）
  }
}
```

**参数说明：**
- **embedding_dim**: 控制特征向量的维度，越大模型表达能力越强，但计算量和参数量也会增加
- **hidden_units**: 隐藏层的结构，如 `[64, 32, 16]` 表示3层，分别有64、32、16个节点
- **dropout_rate**: 随机丢弃神经元的比例，防止过拟合，建议范围 0.1-0.5

### 2. 训练参数

位置：`train_configs/v1/train_config.json` → `training_config`

```json
{
  "training_config": {
    "epochs": 5,                    // 训练轮数
    "batch_size": 1024,             // 批次大小
    "learning_rate": 0.001,         // 学习率（重要！）
    "device": "cpu",                // 设备："cpu" 或 "cuda"
    "optimizer": "adam",            // 优化器
    "loss_function": "bce_loss"     // 损失函数
  }
}
```

**参数说明：**
- **epochs**: 训练轮数，可以根据训练曲线调整
- **batch_size**: 每次训练使用的样本数，越大训练越快但显存占用越多
- **learning_rate**: 学习率，控制模型参数更新速度（关键参数！）
  - 太小：训练慢，可能陷入局部最优
  - 太大：训练不稳定，可能不收敛
  - 推荐值：0.0001, 0.001, 0.01

### 3. 特征处理参数

位置：`train_configs/v1/features_process.json`

#### id类特征（使用hash_bucket）

```json
"user_id": {
  "field": "user",
  "type": "id",
  "processor": "hash_bucket",
  "bucket_size": 10000  // 桶大小，建议设为unique值数量的2-10倍
}
```

**bucket_size设置原则：**
- 桶太小：不同ID可能碰撞，信息丢失
- 桶太大：Embedding参数多，计算开销大
- 建议：`unique数量 * 2~10`

#### 类别特征（使用vocab）

```json
"brand": {
  "field": "item",
  "type": "categorical",
  "processor": "vocab",
  "vocab_list": ["BrandA", "BrandB", "BrandC", "BrandD"]  // 预定义词表
}
```

**词表说明：**
- 如果不填 `vocab_list`，会从训练数据自动构建
- 建议预定义词表，保证训练和测试一致

#### 连续特征

```json
"age": {
  "field": "user",
  "type": "continuous",
  "processor": "minmax",  // 或 "zscore"
  "min": 18,             // minmax专用：最小值
  "max": 65              // minmax专用：最大值
}
```

**归一化方式：**
- **minmax**: 归一化到 [0, 1]，适合数据分布已知的情况
- **zscore**: 标准化（均值0，标准差1），适合数据分布未知的情况

### 4. Early Stopping参数

位置：`train_configs/v1/train_config.json` → `task_config` → `early_stopping`

```json
{
  "early_stopping": {
    "patience": 3,      // 容忍多少个epoch不提升
    "metric": "auc",    // 监控的指标
    "mode": "max"       // "max"表示越大越好，"min"表示越小越好
  }
}
```

**工作原理：**
- 如果验证集指标连续 `patience` 个epoch没有提升，就提前停止训练
- 避免过拟合，节省训练时间

---

## 🚀 模型部署（使用predictor.py）

`predictor.py` 是模型的推理脚本，用于将训练好的模型部署到生产环境。

### 1. 初始化推理器

```python
from predictor import Predictor

# 加载模型
predictor = Predictor(
    model_dir='./trained_models_dir/v1_20260226_172330',  # 模型目录
    checkpoint_name='best_model.pth',                     # checkpoint文件名
    device='cpu'                                          # 设备类型
)
```

### 2. 单条推理

```python
# 方式1：直接传入字典
features = {
    'user_id': 12345,
    'item_id': 67890,
    'cate_id': '101',
    'brand': 'BrandA',
    'gender': 'M',
    'age': 30,
    'income': 50000
}
prediction = predictor.predict_single(features)
print(f"预测概率: {prediction:.4f}")

# 方式2：传入JSON字符串
features_json = '{"user_id": 12345, "item_id": 67890, ...}'
prediction = predictor.predict_single(features_json)
```

### 3. 批量推理

```python
# 批量推理（列表）
batch_features = [
    {'user_id': 12345, 'item_id': 67890, ...},
    {'user_id': 12346, 'item_id': 67891, ...},
    {'user_id': 12347, 'item_id': 67892, ...}
]
predictions = predictor.predict_batch(batch_features, batch_size=32)
```

### 4. 从CSV文件推理

```python
# CSV格式：id,features,label
result_df = predictor.predict_from_csv(
    csv_path='./data/test.csv',
    features_col='features',
    id_col='id'
)
print(result_df.head())
```

### 5. 从DataFrame推理

```python
import pandas as pd

# 已经解析好的DataFrame
df = pd.DataFrame([
    {'user_id': 12345, 'item_id': 67890, ...},
    {'user_id': 12346, 'item_id': 67891, ...}
])
predictions = predictor.predict_from_dataframe(df)
```

### 部署示例：Flask API服务

```python
from flask import Flask, request, jsonify
from predictor import Predictor

app = Flask(__name__)

# 加载模型
predictor = Predictor(
    model_dir='./trained_models_dir/v1_20260226_172330',
    checkpoint_name='best_model.pth'
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')
    prediction = predictor.predict_single(features)
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 注意事项

1. **OOV（未登录词）处理**：
   - id类特征：未登录的ID映射到0
   - 类别特征：未登录的类别映射到0
   - 连续特征：NaN值按正常流程处理

2. **特征完整性**：
   - 推理时的特征必须与训练时完全一致
   - 缺少特征会导致错误

3. **性能优化**：
   - 使用批量推理可以提高效率
   - 合理设置 `batch_size`（32、64、128、256）

---

## 📊 模型评估（使用evaluate_model.py）

`evaluate_model.py` 用于评估模型在测试集上的性能，生成详细的评估报告。

### 1. 基础评估

```bash
python evaluate_model.py
```

### 2. 自定义配置

编辑 `evaluate_model.py` 的 `if __name__ == '__main__'` 部分：

```python
if __name__ == '__main__':
    # 配置
    model_dir = './trained_models_dir/v1_20260226_172330'  # 模型目录
    checkpoint_name = 'best_model.pth'                     # checkpoint文件名
    test_file = './data/test.csv'                          # 测试文件路径
    threshold = 0.5                                        # 分类阈值
    batch_size = 32                                        # 批次大小
    device = None                                          # 设备类型
    multi_threshold = False                                 # 是否多阈值评估

    # 初始化评估器
    evaluator = ModelEvaluator(model_dir, checkpoint_name, device)

    # 评估
    evaluator.evaluate_from_csv(
        test_file,
        threshold=threshold,
        batch_size=batch_size
    )
```

### 3. 评估指标说明

#### 基础指标

| 指标 | 说明 | 范围 | 越大越好 |
|------|------|------|---------|
| **AUC** | ROC曲线下面积，衡量模型区分正负样本的能力 | 0.5-1.0 | ✅ |
| **LogLoss** | 对数损失，衡量预测概率的准确性 | 0-∞ | ❌ |
| **Accuracy** | 准确率，整体预测正确的比例 | 0-1.0 | ✅ |
| **Precision** | 精确率，预测为正的真实为正的比例 | 0-1.0 | ✅ |
| **Recall** | 召回率，真实为正的预测为正的比例 | 0-1.0 | ✅ |
| **F1-Score** | 精确率和召回率的调和平均 | 0-1.0 | ✅ |

#### 混淆矩阵

```
                预测为正    预测为负
真实为正        TP          FN
真实为负        FP          TN
```

- **TP (True Positive)**: 真正例 - 预测为正且真实为正
- **TN (True Negative)**: 真负例 - 预测为负且真实为负
- **FP (False Positive)**: 假正例 - 预测为正但真实为负
- **FN (False Negative)**: 假负例 - 预测为负但真实为正

#### 其他指标

- **Sensitivity (灵敏度)**: 召回率的别名
- **Specificity (特异度)**: 真负例率 = TN / (TN + FP)
- **Negative Predictive Value**: 负预测值 = TN / (TN + FN)

### 4. 多阈值评估

如果你想对比不同阈值下的表现，可以：

```python
# 设置 multi_threshold = True
multi_threshold = True

# 运行评估
python evaluate_model.py
```

会输出多个阈值（0.1, 0.2, ..., 0.9）下的评估结果。

### 5. 输出文件

评估会生成 `prediction_results.csv`，包含：

| 列名 | 说明 |
|------|------|
| id | 样本ID |
| label | 真实标签 |
| prediction | 预测概率 |
| predicted_label | 预测标签（0或1） |

### 6. 如何选择最佳阈值

1. **根据业务需求**：
   - 追求高召回（如风控场景）：降低阈值
   - 追求高精确（如广告推荐）：提高阈值

2. **查看多阈值评估结果**：
   ```python
   multi_threshold = True
   ```
   选择平衡 Precision 和 Recall 的阈值

3. **使用 F1-Score**：
   - F1-Score 最大的阈值通常是平衡点

---

## 📝 完整工作流程示例

### 场景：训练一个新模型并评估性能

#### 第1步：创建新配置

```bash
# 创建v2配置
cp -r train_configs/v1 train_configs/v2
```

#### 第2步：修改参数

编辑 `train_configs/v2/train_config.json`：
```json
{
  "model_config": {
    "model_name": "DeepFM",
    "embedding_dim": 16,        // 改大一点
    "hidden_units": [128, 64, 32],
    "dropout_rate": 0.3
  },
  ...
}
```

#### 第3步：修改训练入口

编辑 `run_train.py`：
```python
TRAIN_CONFIG = {
    'config_name': 'v2',  // 改为v2
    'configs_root': './train_configs'
}
```

#### 第4步：开始训练

```bash
python run_train.py
```

训练完成后，查看输出目录：
```bash
ls trained_models_dir/v2_时间戳/
```

#### 第5步：评估模型

编辑 `evaluate_model.py`：
```python
model_dir = './trained_models_dir/v2_时间戳'
```

运行评估：
```bash
python evaluate_model.py
```

#### 第6步：对比不同版本

如果训练了多个版本，可以通过以下方式对比：

| 版本 | AUC | LogLoss | Accuracy |
|------|-----|---------|----------|
| v1 | 0.85 | 0.32 | 0.78 |
| v2 | 0.87 | 0.28 | 0.80 |

选择性能最好的版本用于生产。

---

## ❓ 常见问题

### Q1: 训练时显存不足怎么办？

**A: 减小 `batch_size`**
```json
{
  "batch_size": 512  // 从1024改为512
}
```

**A: 减小 `embedding_dim` 和 `hidden_units`**
```json
{
  "embedding_dim": 4,            // 从8改为4
  "hidden_units": [32, 16, 8]  // 从[64,32,16]改为[32,16,8]
}
```

### Q2: 如何提高模型性能？

**A: 调整学习率**
```json
{
  "learning_rate": 0.0001  // 尝试不同的值：0.0001, 0.001, 0.01
}
```

**A: 增加模型复杂度**
```json
{
  "embedding_dim": 16,            // 增加embedding维度
  "hidden_units": [128, 64, 32],  // 增加隐藏层节点
  "dropout_rate": 0.2             // 降低dropout
}
```

**A: 特征工程**
- 增加更多有效特征
- 优化特征处理方式

### Q3: 训练集效果好但测试集差怎么办？

**这是过拟合问题，解决方案：**

**A: 增加 Dropout**
```json
{
  "dropout_rate": 0.5  // 从0.2增加到0.5
}
```

**A: 减小模型复杂度**
```json
{
  "embedding_dim": 4,
  "hidden_units": [32, 16]
}
```

**A: 增加 Early Stopping 的 patience**
```json
{
  "early_stopping": {
    "patience": 5  // 从3增加到5
  }
}
```

### Q4: 推理时出现 OOV 错误？

**A: 确保特征处理器正确加载**
```python
# 检查 feature_processor.json 是否存在
predictor = Predictor(model_dir, checkpoint_name)
```

**A: 确认特征完整性**
- 推理时的特征名必须与训练时完全一致
- 检查 `features_process.json` 中的特征列表

### Q5: 如何使用GPU训练？

**A: 修改设备配置**
```json
{
  "device": "cuda"  // 从 "cpu" 改为 "cuda"
}
```

**A: 确保 CUDA 可用**
```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
```

### Q6: 如何恢复训练中断的模型？

**A: 加载某个 epoch 的 checkpoint**
```python
from predictor import Predictor

# 加载 epoch_3.pth
predictor = Predictor(
    model_dir='./trained_models_dir/v1_时间戳',
    checkpoint_name='epoch_3.pth'
)
```

### Q7: 如何导出模型用于部署？

**A: 整个目录都可以用于部署**
```bash
# 部署时需要这些文件
v1_时间戳/
├── configs/
│   └── train_config.json
├── models/
│   └── best_model.pth
└── feature_processor.json
```

**A: 使用 predictor.py 进行推理**
```python
from predictor import Predictor
predictor = Predictor(model_dir)
prediction = predictor.predict_single(features)
```

---

## 📞 技术支持

如有问题，请查看：
1. 训练日志：`trained_models_dir/版本号/logs/training.log`
2. 训练历史：`trained_models_dir/版本号/training_history.json`

---

## 📄 许可证

本项目仅供学习参考使用。
