# 列名配置说明

## 概述

数据列名现在通过配置文件中的 `data_columns` 字段指定，而不是硬编码。

## 配置位置

在 `train_configs/{config_name}/train_config.json` 的 `task_config` 中：

```json
{
  "task_config": {
    "data_columns": {
      "id": "id",
      "features": "features",
      "label": "label"
    }
  }
}
```

## 配置字段说明

| 字段 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| id | string | ID 列名 | "id" |
| features | string | features 列名（JSON 格式） | "features" |
| label | string | 标签列名 | "label" |

## 示例配置

### 示例 1: 默认三列格式

数据格式：
```csv
id,features,label
1,"{""user_id"": 1001...}",1
```

配置：
```json
{
  "data_columns": {
    "id": "id",
    "features": "features",
    "label": "label"
  }
}
```

### 示例 2: 自定义列名

数据格式：
```csv
sample_id,feature_json,target
1001,"{""user_id"": 1001...}",1
```

配置：
```json
{
  "data_columns": {
    "id": "sample_id",
    "features": "feature_json",
    "label": "target"
  }
}
```

### 示例 3: 中文列名

数据格式：
```csv
样本ID,特征,标签
1,"{""user_id"": 1001...}",1
```

配置：
```json
{
  "data_columns": {
    "id": "样本ID",
    "features": "特征",
    "label": "标签"
  }
}
```

### 示例 4: 旧多列格式（不支持 JSON）

如果使用旧格式，需要先转换为三列 JSON 格式。

## 代码修改说明

### 1. train_config.json

新增 `data_columns` 配置项：

```json
"task_config": {
  "data_columns": {
    "id": "id",
    "features": "features",
    "label": "label"
  },
  ...
}
```

### 2. TrainFramework/train_pipeline.py

**DataManager 类**：

```python
def __init__(self, data_paths, feature_processor, batch_size, column_config):
    self.column_config = column_config  # 新增列名配置
    ...

def load_data(self, file_path):
    # 从配置读取列名
    id_col = self.column_config.get('id', 'id')
    features_col = self.column_config.get('features', 'features')
    label_col = self.column_config.get('label', 'label')
    
    # 使用配置的列名
    df[features_col]  # 解析 features
    labels = df[label_col].values
    ...
```

### 3. run_train.py

**flatten_train_config 函数**：

```python
def flatten_train_config(train_config):
    flat_config = {
        ...
        'column_config': train_config['task_config'].get('data_columns', {
            'id': 'id',
            'features': 'features',
            'label': 'label'
        })
    }
    return flat_config
```

## 兼容性

如果配置中未指定 `data_columns`，会使用默认值：

```python
'data_columns': {
  'id': 'id',
  'features': 'features',
  'label': 'label'
}
```

## 使用场景

### 场景 1: 标准格式

CSV 列名：`id, features, label`

配置：
```json
"data_columns": {
  "id": "id",
  "features": "features",
  "label": "label"
}
```

### 场景 2: 不同命名规范

CSV 列名：`sample_id, feature_json, target`

配置：
```json
"data_columns": {
  "id": "sample_id",
  "features": "feature_json",
  "label": "target"
}
```

### 场景 3: 多语言环境

CSV 列名：`样本ID, 特征, 标签`

配置：
```json
"data_columns": {
  "id": "样本ID",
  "features": "特征",
  "label": "标签"
}
```

## 错误处理

如果列名不存在，会抛出清晰的错误信息：

```
ValueError: Label column 'label' not found in data. 
Available columns: ['id', 'feature_json', 'target']
```

这会帮助你快速发现配置错误。

## 测试验证

### 测试不同列名

创建测试数据 `test_custom_columns.csv`：

```csv
sample_id,feature_json,target
1001,"{""user_id"": 1001...}",1
```

修改配置 `train_configs/v1/train_config.json`：

```json
{
  "task_config": {
    "data_columns": {
      "id": "sample_id",
      "features": "feature_json",
      "label": "target"
    }
  }
}
```

运行训练：

```bash
python run_train.py
```

## 注意事项

1. **大小写敏感**
   - 列名区分大小写
   - `"features"` 和 `"Features"` 是不同的

2. **空格处理**
   - CSV 列名不要包含前后空格
   - 配置中的列名也要保持一致

3. **默认值**
   - 未配置时使用默认值
   - 建议在配置文件中明确指定

4. **JSON 列名**
   - features 列的内容必须是 JSON 格式
   - 使用 `json.loads()` 解析

## 迁移指南

### 从硬编码迁移到配置

旧代码：
```python
features_col = 'features'  # 硬编码
label_col = 'label'        # 硬编码
```

新代码：
```python
features_col = self.column_config.get('features', 'features')
label_col = self.column_config.get('label', 'label')
```

### 更新现有配置

1. 打开 `train_configs/v1/train_config.json`
2. 添加 `data_columns` 配置
3. 根据你的数据列名填写
4. 保存并运行训练

## 总结

✅ **优势**：
- 灵活配置不同列名
- 无需修改代码
- 支持多语言环境
- 清晰的错误提示

📝 **配置方式**：
- 在 `train_config.json` 中配置
- 使用 `task_config.data_columns` 字段
- 支持 id, features, label 三个配置项
