"""
测试 JSON 格式数据加载
"""
import pandas as pd
import json

# 读取示例数据
df = pd.read_csv('sample_json_format.csv')

print("="*60)
print("测试 JSON 格式数据加载")
print("="*60)

print(f"\n总行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

print("\n前 3 行数据:")
print(df.head(3))

# 解析第一行的 features
first_features_json = df.iloc[0]['features']
print(f"\n第一行 features (JSON 字符串):")
print(first_features_json)

# 解析 JSON
features = json.loads(first_features_json)
print(f"\n解析后的 features:")
print(f"  类型: {type(features)}")
print(f"  内容: {features}")

print("\n所有 features 解析测试:")
features_list = []
for idx, row in df.iterrows():
    try:
        features = json.loads(row['features'])
        features_list.append(features)
        print(f"  第 {idx+1} 行: ✓ 成功解析")
    except json.JSONDecodeError as e:
        print(f"  第 {idx+1} 行: ✗ 解析失败 - {e}")

# 转换为 DataFrame
features_df = pd.DataFrame(features_list)
print(f"\n转换后的 DataFrame:")
print(features_df)

# 测试提取标签
labels = df['label'].values
print(f"\n标签:")
print(labels)

print("\n" + "="*60)
print("测试完成！")
print("="*60)
