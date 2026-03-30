"""
数据扩充脚本 - 生成更多的训练样本
根据现有数据和特征配置生成扩充的数据集
"""
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict


class DataGenerator:
    """数据生成器"""

    def __init__(self, config_path):
        """
        初始化数据生成器

        Args:
            config_path: 特征配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.feature_config = json.load(f)

        self.cate_ids = ['101', '102', '103']
        self.brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD']
        self.genders = ['M', 'F']

    def generate_user_id(self, start_id, count):
        """生成用户ID"""
        return [start_id + i for i in range(count)]

    def generate_item_id(self, start_id, count):
        """生成商品ID"""
        return [start_id + i for i in range(count)]

    def generate_age(self, count, min_age=18, max_age=65):
        """生成年龄 - 正态分布"""
        mean_age = 35
        std_age = 10
        ages = np.random.normal(mean_age, std_age, count)
        ages = np.clip(ages, min_age, max_age).astype(int)
        return ages

    def generate_income(self, count, base_income=60000, std_income=15000):
        """生成收入 - 正态分布"""
        incomes = np.random.normal(base_income, std_income, count)
        incomes = np.clip(incomes, 30000, 150000).astype(int)
        return incomes

    def generate_cate_id(self, count):
        """生成分类ID - 按权重分布"""
        cate_id = self.feature_config['cate_id'].get('vocab_list', ['101', '102', '103'])
        weights = [0.4, 0.4, 0.2]  # 101和102更常见
        return np.random.choice(cate_id, count, p=weights)

    def generate_brand(self, count):
        """生成品牌 - 按权重分布"""
        brand = self.feature_config['brand'].get('vocab_list', ['BrandA', 'BrandB', 'BrandC', 'BrandD'])
        weights = [0.35, 0.30, 0.20, 0.15]  # BrandA 最常见
        return np.random.choice(brand, count, p=weights)

    def generate_gender(self, count):
        """生成性别 - 均衡分布"""
        return np.random.choice(self.genders, count)

    def generate_click(self, count, click_rate=0.5):
        """生成点击标签"""
        # 根据某些特征调整点击率
        clicks = []

        for i in range(count):
            # 基础点击率
            base_rate = click_rate

            # 年龄因素：中年用户点击率稍高
            if hasattr(self, '_age_buffer'):
                if self._age_buffer[i] > 30 and self._age_buffer[i] < 45:
                    base_rate += 0.1

            # 收入因素：中高收入用户点击率稍高
            if hasattr(self, '_income_buffer'):
                if self._income_buffer[i] > 55000:
                    base_rate += 0.05

            # 品牌因素：BrandA 点击率稍高
            if hasattr(self, '_brand_buffer'):
                if self._brand_buffer[i] == 'BrandA':
                    base_rate += 0.08
                elif self._brand_buffer[i] == 'BrandD':
                    base_rate -= 0.05

            # 性别因素
            if hasattr(self, '_gender_buffer'):
                if self._gender_buffer[i] == 'F':
                    base_rate += 0.03

            # 限制点击率范围
            base_rate = max(0.1, min(0.9, base_rate))

            clicks.append(np.random.choice([0, 1], p=[1 - base_rate, base_rate]))

        return clicks

    def generate_dataset(self, num_samples, id_start):
        """生成数据集（三列格式：id, features, label）"""
        print(f"生成 {num_samples} 条样本...")

        # 生成特征数据
        ages = self.generate_age(num_samples)
        incomes = self.generate_income(num_samples)
        cate_ids = self.generate_cate_id(num_samples)
        brands = self.generate_brand(num_samples)
        genders = self.generate_gender(num_samples)

        # 保存中间结果用于生成 click
        self._age_buffer = ages
        self._income_buffer = incomes
        self._brand_buffer = brands
        self._gender_buffer = genders

        labels = self.generate_click(num_samples)

        # 生成数据
        data = []
        for i in range(num_samples):
            sample_id = id_start + i

            # 构建 features JSON
            features = {
                'user_id': sample_id + 1000,
                'item_id': sample_id + 2000,
                'age': int(ages[i]),
                'income': int(incomes[i]),
                'cate_id': cate_ids[i],
                'brand': brands[i],
                'gender': genders[i]
            }

            features_json = json.dumps(features, ensure_ascii=False)

            data.append({
                'id': sample_id,
                'features': features_json,
                'label': int(labels[i])
            })

        # 创建 DataFrame
        df = pd.DataFrame(data)

        return df


def main():
    """主函数"""
    print("="*80)
    print("数据集扩充脚本")
    print("="*80)

    # 获取脚本所在目录（data 文件夹）的绝对路径
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(DATA_DIR, '..', 'train_configs', 'v1', 'features_process.json')

    print(f"\n工作目录: {DATA_DIR}")
    print(f"配置文件路径: {CONFIG_PATH}")

    # 数据集配置
    DATASET_CONFIG = {
        'train': {
            'num_samples': 500000,  # 训练集样本数
            'id_start': 1
        },
        'validation': {
            'num_samples': 500000,   # 验证集样本数
            'id_start': 500001
        },
        'test': {
            'num_samples': 500000,   # 测试集样本数
            'id_start': 1000001
        }
    }

    # 初始化数据生成器
    generator = DataGenerator(CONFIG_PATH)

    # 生成数据集
    for dataset_name, config in DATASET_CONFIG.items():
        print(f"\n生成 {dataset_name} 数据集...")
        df = generator.generate_dataset(
            num_samples=config['num_samples'],
            id_start=config['id_start']
        )

        # 保存数据集
        output_path = os.path.join(DATA_DIR, f'{dataset_name}.csv')
        print(f"保存路径: {output_path}")
        df.to_csv(output_path, index=False)

        # 打印统计信息
        print(f"\n{dataset_name} 数据集统计:")
        print(f"  总样本数: {len(df)}")
        print(f"  标签分布: {df['label'].value_counts().to_dict()}")
        print(f"  正样本率: {df['label'].mean():.2%}")
        print(f"  已保存到: {output_path}")

    print("\n" + "="*80)
    print("数据集生成完成！")
    print("="*80)

    # 打印总体统计
    print("\n总体数据集统计:")
    total_samples = sum(config['num_samples'] for config in DATASET_CONFIG.values())
    print(f"  总样本数: {total_samples}")
    print(f"  训练集: {DATASET_CONFIG['train']['num_samples']} ({DATASET_CONFIG['train']['num_samples']/total_samples:.1%})")
    print(f"  验证集: {DATASET_CONFIG['validation']['num_samples']} ({DATASET_CONFIG['validation']['num_samples']/total_samples:.1%})")
    print(f"  测试集: {DATASET_CONFIG['test']['num_samples']} ({DATASET_CONFIG['test']['num_samples']/total_samples:.1%})")


if __name__ == "__main__":
    main()
