import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from collections import defaultdict
import logging
import hashlib
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureProcessor:
    def __init__(self, feature_config):
        """
        初始化特征处理器

        Args:
            feature_config (dict): 特征配置字典，来自features_process.json
        """
        self.feature_config = feature_config
        self.processors = {}
        self.vocab_maps = {}
        self.fitted_features = set()  # 记录已拟合的特征
        self.seen_ids = {}  # 记录hash_bucket特征在训练集中见过的值

    def validate_config(self, df):
        """
        验证配置与数据的一致性

        Args:
            df (pd.DataFrame): 输入数据

        Returns:
            bool: 验证是否通过
        """
        data_columns = set(df.columns)
        config_features = set(self.feature_config.keys())

        # 检查配置中是否存在数据中没有的特征
        missing_in_data = config_features - data_columns
        if missing_in_data:
            logger.warning(f"Features in config but not in data: {missing_in_data}")

        # 检查数据中是否存在配置中没有的特征
        missing_in_config = data_columns - config_features
        if missing_in_config:
            logger.info(f"Features in data but not in config: {missing_in_config}")

        return True

    def fit(self, df):
        """
        根据训练数据拟合特征处理器

        Args:
            df (pd.DataFrame): 训练数据
        """
        logger.info("Starting feature fitting process...")

        # 验证配置
        self.validate_config(df)

        # 为每个配置的特征创建处理器
        for feature_name, config in self.feature_config.items():
            if feature_name not in df.columns:
                logger.warning(f"Feature '{feature_name}' not found in data, skipping...")
                continue

            try:
                self._fit_single_feature(feature_name, config, df[feature_name])
                self.fitted_features.add(feature_name)
                logger.info(f"Successfully fitted feature: {feature_name}")
            except Exception as e:
                logger.error(f"Failed to fit feature '{feature_name}': {str(e)}")
                raise

    def _fit_single_feature(self, feature_name, config, feature_series):
        """
        拟合单个特征的处理器

        Args:
            feature_name (str): 特征名称
            config (dict): 特征配置
            feature_series (pd.Series): 特征数据
        """
        processor_type = config['processor']

        if processor_type == 'minmax':
            self._fit_minmax_processor(feature_name, config, feature_series)
        elif processor_type == 'zscore':
            self._fit_zscore_processor(feature_name, config, feature_series)
        elif processor_type == 'vocab':
            self._fit_vocab_processor(feature_name, config, feature_series)
        elif processor_type == 'hash_bucket':
            self._fit_hash_bucket_processor(feature_name, config, feature_series)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")

    def _fit_minmax_processor(self, feature_name, config, feature_series):
        """拟合MinMax处理器"""
        scaler = MinMaxScaler()

        # 检查是否提供了预设的min/max值
        if 'min' in config and 'max' in config:
            min_val = config['min']
            max_val = config['max']
            logger.info(f"Using preset min/max for {feature_name}: [{min_val}, {max_val}]")
            scaler.fit(np.array([[min_val], [max_val]]))
        else:
            # 从数据中计算
            values = feature_series.values.reshape(-1, 1)
            scaler.fit(values)
            logger.info(f"Calculated min/max from data for {feature_name}: "
                        f"[{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")

        self.processors[feature_name] = scaler

    def _fit_zscore_processor(self, feature_name, config, feature_series):
        """拟合Z-Score处理器"""
        scaler = StandardScaler()

        # 检查是否提供了预设的均值/标准差
        if 'mean' in config and 'std' in config:
            mean_val = config['mean']
            std_val = config['std']
            logger.info(f"Using preset mean/std for {feature_name}: ({mean_val}, {std_val})")
            scaler.mean_ = np.array([mean_val])
            scaler.scale_ = np.array([std_val])
            scaler.var_ = np.array([std_val ** 2])
        else:
            # 从数据中计算
            values = feature_series.values.reshape(-1, 1)
            scaler.fit(values)
            logger.info(f"Calculated mean/std from data for {feature_name}: "
                        f"({scaler.mean_[0]:.2f}, {scaler.scale_[0]:.2f})")

        self.processors[feature_name] = scaler

    def _fit_vocab_processor(self, feature_name, config, feature_series):
        """拟合词汇表处理器"""
        vocab_list = config.get('vocab_list', [])

        if vocab_list:
            # 使用预定义的词汇表
            logger.info(f"Using preset vocabulary for {feature_name} with {len(vocab_list)} items")
        else:
            # 从数据中构建词汇表
            unique_values = feature_series.unique()
            vocab_list = sorted(unique_values)
            logger.info(f"Built vocabulary from data for {feature_name} with {len(vocab_list)} items")

        # 索引从1开始，0保留给未登录词（OOV）
        vocab_map = {val: idx + 1 for idx, val in enumerate(vocab_list)}
        self.vocab_maps[feature_name] = vocab_map
        self.processors[feature_name] = len(vocab_list)

    def _fit_hash_bucket_processor(self, feature_name, config, feature_series):
        """拟合哈希分桶处理器"""
        bucket_size = config.get('bucket_size', 1000)
        logger.info(f"Setting hash bucket size for {feature_name}: {bucket_size}")

        # 记录训练集中出现过的所有id值
        unique_values = set()
        for val in feature_series.values:
            if not pd.isna(val):
                unique_values.add(str(val))

        logger.info(f"Recorded {len(unique_values)} unique ids for {feature_name}")

        self.processors[feature_name] = {
            'bucket_size': bucket_size,
            'feature_name': feature_name
        }
        self.seen_ids[feature_name] = unique_values

    def transform(self, df):
        """
        转换特征

        Args:
            df (pd.DataFrame): 待转换的数据

        Returns:
            dict: 转换后的特征字典
        """
        logger.info("Starting feature transformation...")
        processed_features = {}

        for feature_name, config in self.feature_config.items():
            if feature_name not in df.columns:
                logger.warning(f"Feature '{feature_name}' not found in data, skipping...")
                continue

            if feature_name not in self.fitted_features:
                logger.warning(f"Feature '{feature_name}' not fitted, skipping...")
                continue

            try:
                processed_values = self._transform_single_feature(
                    feature_name, config, df[feature_name]
                )
                processed_features[feature_name] = processed_values
                logger.debug(f"Transformed feature: {feature_name}")
            except Exception as e:
                logger.error(f"Failed to transform feature '{feature_name}': {str(e)}")
                raise

        logger.info(f"Successfully transformed {len(processed_features)} features")
        return processed_features

    def _transform_single_feature(self, feature_name, config, feature_series):
        """
        转换单个特征

        Args:
            feature_name (str): 特征名称
            config (dict): 特征配置
            feature_series (pd.Series): 特征数据

        Returns:
            np.array: 转换后的特征值
        """
        processor_type = config['processor']
        raw_values = feature_series.values

        if processor_type == 'minmax':
            return self._transform_minmax(feature_name, raw_values)
        elif processor_type == 'zscore':
            return self._transform_zscore(feature_name, raw_values)
        elif processor_type == 'vocab':
            return self._transform_vocab(feature_name, raw_values)
        elif processor_type == 'hash_bucket':
            return self._transform_hash_bucket(feature_name, raw_values, config)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")

    def _transform_minmax(self, feature_name, raw_values):
        """MinMax转换"""
        scaler = self.processors[feature_name]
        processed_values = scaler.transform(raw_values.reshape(-1, 1)).flatten()
        return processed_values

    def _transform_zscore(self, feature_name, raw_values):
        """Z-Score转换"""
        scaler = self.processors[feature_name]
        processed_values = scaler.transform(raw_values.reshape(-1, 1)).flatten()
        return processed_values

    def _transform_vocab(self, feature_name, raw_values):
        """词汇表转换"""
        vocab_map = self.vocab_maps[feature_name]
        unknown_idx = 0  # 未登录词映射到0

        processed_values = []
        for val in raw_values:
            if pd.isna(val):
                processed_values.append(unknown_idx)
            else:
                processed_values.append(vocab_map.get(str(val), unknown_idx))

        return np.array(processed_values)

    def _transform_hash_bucket(self, feature_name, raw_values, config):
        """哈希分桶转换"""
        processor_info = self.processors[feature_name]
        bucket_size = processor_info['bucket_size']
        feature_prefix = processor_info['feature_name']
        seen_values = self.seen_ids.get(feature_name, set())

        processed_values = []

        for val in raw_values:
            if pd.isna(val):
                processed_values.append(0)  # NaN值映射到0
            else:
                val_str = str(val)
                # 检查是否为oov（训练集中未见过）
                if val_str not in seen_values:
                    processed_values.append(0)  # oov映射到0
                else:
                    # 使用MD5哈希确保跨平台一致性
                    # 将特征名作为前缀，确保不同特征的相同值产生不同哈希
                    hash_input = f"{feature_prefix}_{val_str}".encode('utf-8')
                    hash_object = hashlib.md5(hash_input)
                    hash_hex = hash_object.hexdigest()
                    # 取哈希值的前8位转换为整数，然后取模
                    hash_val = int(hash_hex[:8], 16) % bucket_size
                    processed_values.append(hash_val)

        return np.array(processed_values)

    def save_processor(self, filepath):
        """
        保存特征处理器到文件

        Args:
            filepath (str): 保存路径
        """
        processor_data = {
            'processors': {},
            'vocab_maps': self.vocab_maps,
            'feature_config': self.feature_config,
            'fitted_features': list(self.fitted_features),
            'seen_ids': self.seen_ids
        }

        # 保存处理器信息
        for feature_name, processor in self.processors.items():
            config = self.feature_config[feature_name]
            processor_type = config['processor']

            if processor_type == 'minmax':
                processor_data['processors'][feature_name] = {
                    'type': 'minmax_scaler',
                    'min_': float(processor.data_min_[0]),
                    'max_': float(processor.data_max_[0]),
                    'scale_': float(processor.scale_[0])
                }
            elif processor_type == 'zscore':
                processor_data['processors'][feature_name] = {
                    'type': 'standard_scaler',
                    'mean_': float(processor.mean_[0]),
                    'scale_': float(processor.scale_[0]),
                    'var_': float(processor.var_[0])
                }
            elif processor_type == 'vocab':
                processor_data['processors'][feature_name] = {
                    'type': 'vocab',
                    'vocab_size': processor
                }
            elif processor_type == 'hash_bucket':
                processor_data['processors'][feature_name] = {
                    'type': 'hash_bucket',
                    'bucket_size': processor['bucket_size'],
                    'feature_name': processor['feature_name']
                }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(processor_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Feature processor saved to {filepath}")

    def load_processor(self, filepath):
        """
        从文件加载特征处理器

        Args:
            filepath (str): 加载路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            processor_data = json.load(f)

        self.feature_config = processor_data.get('feature_config', {})
        self.vocab_maps = processor_data.get('vocab_maps', {})
        self.fitted_features = set(processor_data.get('fitted_features', []))
        self.seen_ids = processor_data.get('seen_ids', {})

        # 重建处理器
        for feature_name, proc_info in processor_data['processors'].items():
            proc_type = proc_info['type']

            if proc_type == 'minmax_scaler':
                scaler = MinMaxScaler()
                scaler.data_min_ = np.array([proc_info['min_']])
                scaler.data_max_ = np.array([proc_info['max_']])
                scaler.scale_ = np.array([proc_info['scale_']])
                scaler.data_range_ = scaler.data_max_ - scaler.data_min_
                scaler.min_ = np.array([proc_info['min_']])  # 添加 min_ 属性
                self.processors[feature_name] = scaler
            elif proc_type == 'standard_scaler':
                scaler = StandardScaler()
                scaler.mean_ = np.array([proc_info['mean_']])
                scaler.scale_ = np.array([proc_info['scale_']])
                scaler.var_ = np.array([proc_info['var_']])
                self.processors[feature_name] = scaler
            elif proc_type == 'vocab':
                self.processors[feature_name] = proc_info['vocab_size']
            elif proc_type == 'hash_bucket':
                self.processors[feature_name] = {
                    'bucket_size': proc_info['bucket_size'],
                    'feature_name': proc_info['feature_name']
                }

        logger.info(f"Feature processor loaded from {filepath}")

    def get_feature_info(self):
        """
        获取特征处理信息

        Returns:
            dict: 特征处理信息
        """
        info = {}
        for feature_name, config in self.feature_config.items():
            info[feature_name] = {
                'processor_type': config['processor'],
                'fitted': feature_name in self.fitted_features,
                'dimension': self._get_feature_dimension(feature_name, config)
            }
        return info

    def _get_feature_dimension(self, feature_name, config):
        """获取特征维度"""
        processor_type = config['processor']

        if processor_type in ['minmax', 'zscore']:
            return 1
        elif processor_type == 'vocab':
            return self.processors.get(feature_name, 0) + 1  # +1 for unknown
        elif processor_type == 'hash_bucket':
            return self.processors.get(feature_name, 1000)
        else:
            return 1

if __name__ == '__main__':
    # 示例用法
    train_df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'income': [50000, 60000, 70000, 80000, 90000],
        'education': ['High School', 'College', 'Graduate', 'Postgraduate', 'PhD']
    })
    test_df = pd.DataFrame({
        'age': [30, 40, 50, 60, 70],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'income': [60000, 70000, 80000, 90000, 100000],
        'education': ['High School', 'College', 'Graduate','UnKnown','UnKnown']
    })
    feature_config = {
        'age': {'processor': 'minmax'},
        'gender': {'processor': 'vocab'},
        'income': {'processor': 'zscore'},
        'education': {'processor': 'vocab'}
    }
    # 创建特征处理器
    feature_processor = FeatureProcessor(feature_config)

    # 拟合训练数据
    feature_processor.fit(train_df)

    # 转换特征
    train_data  = feature_processor.transform(train_df)
    test_data = feature_processor.transform(test_df)
    print(train_data)
    print('-'*100)
    print(test_data)
    # # 保存处理器
    # feature_processor.save_processor('feature_processor.json')

    # # 加载处理器
    # new_processor = FeatureProcessor()
    # new_processor.load_processor('feature_processor.json')
    #

