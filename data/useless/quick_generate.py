"""
快速数据生成脚本 - 简化版
快速生成扩大的训练数据集
格式：id, features(json), label
"""
import random
import csv
import os
import json


def generate_csv(filename, num_samples, id_start):
    """生成 CSV 文件（三列格式：id, features, label）"""
    # 使用脚本所在目录作为输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, filename)
    print(f"生成 {output_path} ({num_samples} 条样本)...")

    # 特征定义
    cate_ids = ['101', '102', '103']
    cate_weights = [0.4, 0.4, 0.2]

    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD']
    brand_weights = [0.35, 0.3, 0.2, 0.15]

    genders = ['M', 'F']

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'features', 'label'])

        for i in range(num_samples):
            sample_id = id_start + i

            # 生成年龄（18-65，正态分布）
            age = int(random.gauss(35, 10))
            age = max(18, min(65, age))

            # 生成收入（30000-150000）
            income = int(random.gauss(60000, 15000))
            income = max(30000, min(150000, income))

            # 生成分类ID
            cate_id = random.choices(cate_ids, weights=cate_weights, k=1)[0]

            # 生成品牌
            brand = random.choices(brands, weights=brand_weights, k=1)[0]

            # 生成性别
            gender = random.choice(genders)

            # 生成点击标签（基于特征的概率）
            click_rate = 0.5

            # 年龄因素
            if 30 < age < 45:
                click_rate += 0.1

            # 收入因素
            if income > 55000:
                click_rate += 0.05

            # 品牌因素
            if brand == 'BrandA':
                click_rate += 0.08
            elif brand == 'BrandD':
                click_rate -= 0.05

            # 性别因素
            if gender == 'F':
                click_rate += 0.03

            # 限制范围
            click_rate = max(0.1, min(0.9, click_rate))
            label = 1 if random.random() < click_rate else 0

            # 构建 features JSON
            features = {
                'user_id': sample_id + 1000,  # 模拟 user_id
                'item_id': sample_id + 2000,  # 模拟 item_id
                'age': age,
                'income': income,
                'cate_id': cate_id,
                'brand': brand,
                'gender': gender
            }

            # 转换为 JSON 字符串
            features_json = json.dumps(features, ensure_ascii=False)

            writer.writerow([sample_id, features_json, label])

            # 每 100 条打印进度
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{num_samples} 条")

    print(f"✓ 完成！已保存到 {output_path}\n")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("快速数据生成脚本 (JSON 格式)")
    print("="*60 + "\n")

    # 配置参数
    configs = [
        ('train.csv', 1000, 1),
        ('validation.csv', 200, 1001),
        ('test.csv', 200, 2001)
    ]

    # 生成数据集
    for filename, num_samples, id_start in configs:
        generate_csv(filename, num_samples, id_start)

    # 统计信息
    print("="*60)
    print("生成完成！")
    print("="*60)
    print(f"\n总样本数: {sum(c[1] for c in configs)}")
    print(f"  - train.csv: {configs[0][1]} 条")
    print(f"  - validation.csv: {configs[1][1]} 条")
    print(f"  - test.csv: {configs[2][1]} 条")
    print("\n数据格式：id, features(json), label")
    print("现在可以开始训练了！")
    print("运行: cd .. && python run_train.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
