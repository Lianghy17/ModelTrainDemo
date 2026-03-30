# 新数据格式说明

## 数据格式变更

从多列格式改为三列 JSON 格式：

### 旧格式（多列）
```csv
user_id,item_id,age,income,cate_id,brand,gender,click
1001,2001,25,50000,101,BrandA,M,1
1002,2002,30,60000,102,BrandB,F,0
```

### 新格式（三列 JSON）
```csv
id,features,label
1,"{""user_id"": 1001, ""item_id"": 2001, ""age"": 25, ""income"": 50000, ""cate_id"": ""101"", ""brand"": ""BrandA"", ""gender"": ""M""}",1
2,"{""user_id"": 1002, ""item_id"": 2002, ""age"": 30, ""income"": 60000, ""cate_id"": ""102"", ""brand"": ""BrandB"", ""gender"": ""F""}",0
```

## 格式说明

### 列定义

1. **id** (int)
   - 样本唯一标识符
   - 示例: 1, 2, 3...

2. **features** (JSON string)
   - 包含所有特征的 JSON 对象
   - 键值对格式，双引号转义
   - 示例: `{"user_id": 1001, "age": 25, ...}`

3. **label** (int)
   - 标签值（0 或 1）
   - 示例: 0, 1

### 特征字段

features JSON 中包含的字段：

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| user_id | int | 用户ID | 1001 |
| item_id | int | 商品ID | 2001 |
| age | int | 年龄 | 25 |
| income | int | 收入 | 50000 |
| cate_id | string | 分类ID | "101" |
| brand | string | 品牌 | "BrandA" |
| gender | string | 性别 | "M" |

## 生成数据

### 快速生成（1000 条样本）

```bash
cd data
python3 quick_generate.py
```

### 完整生成（500,000 条样本）

```bash
cd data
python3 generate_data.py
```

## 代码修改

### 1. 数据生成脚本

- `quick_generate.py` - 快速生成脚本（已更新）
- `generate_data.py` - 完整生成脚本（已更新）

修改内容：
- 生成三列格式（id, features, label）
- features 字段为 JSON 字符串

### 2. 训练代码

- `TrainFramework/train_pipeline.py` - 训练管道（已更新）
- `train_configs/v1/train_config.json` - 配置文件（已更新）
- `run_train.py` - 训练入口（已更新）

修改内容：
- `DataManager.load_data()` - 解析 JSON 格式的 features
- `DataManager` - 移除 label_col 参数（固定使用 'label'）
- `TrainPipeline` - 移除 label_column 配置

### 3. 特征处理器

无需修改，`FeatureProcessor` 自动处理从 JSON 解析的特征 DataFrame。

## 使用示例

### 查看示例数据

```bash
cat data/sample_json_format.csv
```

### 生成新数据

```bash
cd data
python3 quick_generate.py
```

### 训练模型

```bash
cd ..
python run_train.py
```

## 注意事项

1. **JSON 转义**
   - CSV 中的 JSON 字符串需要转义双引号
   - 使用 `json.dumps(features, ensure_ascii=False)` 生成

2. **字段顺序**
   - 固定顺序: id, features, label
   - features 中的字段顺序不影响功能

3. **数据类型**
   - id: int
   - features: JSON string
   - label: int (0 或 1)

4. **特征值类型**
   - 数值型: int (如 age, income, user_id, item_id)
   - 类别型: string (如 cate_id, brand, gender)

## 兼容性

新格式完全向下兼容，只需要：
1. 重新生成数据集
2. 使用更新后的训练代码
3. 无需修改特征处理器配置

## 迁移步骤

从旧格式迁移到新格式：

1. 备份旧数据
   ```bash
   mv data/train.csv data/train_old.csv
   mv data/validation.csv data/validation_old.csv
   mv data/test.csv data/test_old.csv
   ```

2. 生成新格式数据
   ```bash
   cd data
   python3 quick_generate.py
   ```

3. 开始训练
   ```bash
   cd ..
   python run_train.py
   ```
