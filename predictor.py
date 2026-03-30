import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from typing import Union, List, Dict, Any
from TrainFramework.preprocess import FeatureProcessor
from TrainFramework.models.deepfm import DeepFM, MLP, WideAndDeep

logger = logging.getLogger(__name__)


class Predictor:
    """推理类：加载模型和特征处理器，支持单条和批量推理"""

    def __init__(self, model_dir: str, checkpoint_name: str = 'best_model.pth', device: str = None):
        """
        初始化推理器

        Args:
            model_dir: 模型目录，包含feature_processor.json和models文件夹
            checkpoint_name: 模型checkpoint文件名，默认为best_model.pth
            device: 设备类型，默认为None（自动选择）
        """
        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        self.device = self._get_device(device)

        logger.info(f"Initializing Predictor...")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Checkpoint: {checkpoint_name}")
        logger.info(f"Device: {self.device}")

        # 加载特征处理器
        self.feature_processor = self._load_feature_processor()

        # 加载模型配置
        self.model_config = self._load_model_config()

        # 预计算特征维度
        self.discrete_feature_dims, self.continuous_features = self._calculate_feature_dims()

        # 加载模型
        self.model = self._load_model()

        logger.info("Predictor initialized successfully!")

    def _get_device(self, device: str = None) -> torch.device:
        """获取设备"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)

    def _load_feature_processor(self):
        """加载特征处理器"""

        processor_path = os.path.join(self.model_dir, 'feature_processor.json')
        if not os.path.exists(processor_path):
            raise FileNotFoundError(f"Feature processor not found: {processor_path}")

        logger.info(f"Loading feature processor from: {processor_path}")
        feature_processor = FeatureProcessor({})
        feature_processor.load_processor(processor_path)
        return feature_processor

    def _load_model_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        config_path = os.path.join(self.model_dir, 'configs', 'train_config.json')
        if not os.path.exists(config_path):
            # 尝试在模型目录根目录查找
            config_path = os.path.join(self.model_dir, 'train_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config not found in {self.model_dir}")

        logger.info(f"Loading model config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        model_config = config_data['model_config']
        logger.info(f"Model name: {model_config['model_name']}")
        return model_config

    def _load_model(self) -> nn.Module:
        """加载模型"""
        checkpoint_path = os.path.join(self.model_dir, 'models', self.checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # 创建模型
        model_name = self.model_config['model_name']
        embedding_dim = self.model_config['embedding_dim']
        hidden_units = self.model_config['hidden_units']
        dropout_rate = self.model_config['dropout_rate']

        model_class = {
            'DeepFM': DeepFM,
            'MLP': MLP,
            'WideAndDeep': WideAndDeep
        }.get(model_name)

        if model_class is None:
            raise ValueError(f"Unsupported model: {model_name}")

        # 创建模型实例（使用计算的特征维度）
        model = model_class(
            self.discrete_feature_dims,
            self.continuous_features,
            embedding_dim,
            hidden_units,
            dropout_rate
        )
        model.to(self.device)

        # 加载模型权重
        if self.checkpoint_name == 'best_model.pth':
            # best_model.pth只包含state_dict
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            # epoch_x.pth包含完整的checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict)

        model.eval()
        logger.info("Model loaded and set to eval mode")
        return model

    def _calculate_feature_dims(self) -> tuple:
        """计算特征维度"""
        discrete_feature_dims = {}
        continuous_features = []

        for feature_name, config in self.feature_processor.feature_config.items():
            processor_type = config.get('processor', '')

            if processor_type == 'vocab':
                vocab_size = len(self.feature_processor.vocab_maps.get(feature_name, {}))
                discrete_feature_dims[feature_name] = vocab_size + 1
            elif processor_type == 'hash_bucket':
                bucket_info = self.feature_processor.processors.get(feature_name, {})
                if isinstance(bucket_info, dict):
                    bucket_size = bucket_info.get('bucket_size', 1000)
                else:
                    bucket_size = 1000
                discrete_feature_dims[feature_name] = bucket_size
            elif processor_type in ['minmax', 'zscore']:
                continuous_features.append(feature_name)

        logger.info(f"Discrete feature dims: {discrete_feature_dims}")
        logger.info(f"Continuous features: {continuous_features}")

        return discrete_feature_dims, continuous_features

    def predict_single(self, features_json: Union[str, Dict[str, Any]]) -> float:
        """
        单条推理

        Args:
            features_json: 特征数据，可以是JSON字符串或字典

        Returns:
            float: 预测概率
        """
        # 解析JSON
        if isinstance(features_json, str):
            try:
                features_dict = json.loads(features_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        else:
            features_dict = features_json

        # 转换为DataFrame
        df = pd.DataFrame([features_dict])

        # 特征处理
        processed_features = self.feature_processor.transform(df)

        # 转换为模型输入格式
        model_input = {}
        for feature_name, feature_values in processed_features.items():
            model_input[feature_name] = torch.tensor([feature_values[0]], dtype=torch.float32).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(model_input)
            prediction = output.squeeze().cpu().item()

        return prediction

    def predict_batch(self, features_list: List[Union[str, Dict[str, Any]]],
                      batch_size: int = 32) -> List[float]:
        """
        批量推理

        Args:
            features_list: 特征数据列表，每个元素可以是JSON字符串或字典
            batch_size: 批次大小

        Returns:
            List[float]: 预测概率列表
        """
        if len(features_list) == 0:
            return []

        # 解析所有JSON
        parsed_features = []
        for features in features_list:
            if isinstance(features, str):
                try:
                    features_dict = json.loads(features)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    continue
            else:
                features_dict = features
            parsed_features.append(features_dict)

        if not parsed_features:
            logger.warning("No valid features to predict")
            return []

        # 转换为DataFrame
        df = pd.DataFrame(parsed_features)

        # 特征处理
        processed_features = self.feature_processor.transform(df)

        # 批量推理
        all_predictions = []
        num_samples = len(df)

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)

                # 构建批次输入
                batch_input = {}
                for feature_name, feature_values in processed_features.items():
                    batch_input[feature_name] = torch.tensor(
                        feature_values[start_idx:end_idx], dtype=torch.float32
                    ).to(self.device)

                # 推理
                output = self.model(batch_input)
                predictions = output.squeeze().cpu().numpy().tolist()

                all_predictions.extend(predictions)

        return all_predictions

    def predict_from_csv(self, csv_path: str, features_col: str = 'features',
                         id_col: str = 'id', batch_size: int = 32) -> pd.DataFrame:
        """
        从CSV文件进行批量推理

        Args:
            csv_path: CSV文件路径
            features_col: 特征列名（JSON格式）
            id_col: ID列名
            batch_size: 批次大小

        Returns:
            pd.DataFrame: 包含ID和预测结果的DataFrame
        """
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)

        # 检查列是否存在
        if features_col not in df.columns:
            raise ValueError(f"Features column '{features_col}' not found")
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found")

        # 解析JSON特征
        logger.info("Parsing JSON features...")
        features_list = []
        for idx, row in df.iterrows():
            try:
                features = json.loads(row[features_col])
                features_list.append(features)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON at row {idx}: {e}")
                raise

        # 批量推理
        logger.info(f"Running batch prediction for {len(features_list)} samples...")
        predictions = self.predict_batch(features_list, batch_size=batch_size)

        # 构建结果DataFrame
        result_df = pd.DataFrame({
            id_col: df[id_col],
            'prediction': predictions
        })

        logger.info(f"Prediction completed. Result shape: {result_df.shape}")
        return result_df

    def predict_from_dataframe(self, df: pd.DataFrame, batch_size: int = 32) -> List[float]:
        """
        从DataFrame进行批量推理（特征已经解析）

        Args:
            df: 特征DataFrame
            batch_size: 批次大小

        Returns:
            List[float]: 预测概率列表
        """
        # 特征处理
        processed_features = self.feature_processor.transform(df)

        # 批量推理
        all_predictions = []
        num_samples = len(df)

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)

                # 构建批次输入
                batch_input = {}
                for feature_name, feature_values in processed_features.items():
                    batch_input[feature_name] = torch.tensor(
                        feature_values[start_idx:end_idx], dtype=torch.float32
                    ).to(self.device)

                # 推理
                output = self.model(batch_input)
                predictions = output.squeeze().cpu().numpy().tolist()

                all_predictions.extend(predictions)

        return all_predictions


if __name__ == '__main__':
    # 示例用法

    # 初始化推理器
    model_dir = '/Users/lianghaoyun/project/ModelTrainDemo/trained_models_dir/v1_20260226_172330'
    predictor = Predictor(model_dir)

    # 单条推理示例
    print("=" * 80)
    print("单条推理示例:")
    print("=" * 80)
    single_feature = {
        'user_id': 12345,
        'item_id': 67890,
        'cate_id': 1,
        'brand': 100,
        'gender': 'M',
        'age': 30,
        'income': 50000
    }
    prediction = predictor.predict_single(single_feature)
    print(f"特征: {single_feature}")
    print(f"预测概率: {prediction:.4f}")
    print()

    # 批量推理示例
    print("=" * 80)
    print("批量推理示例:")
    print("=" * 80)
    batch_features = [
        {'user_id': 12345, 'item_id': 67890, 'cate_id': 1, 'brand': 100, 'gender': 'M', 'age': 30, 'income': 50000},
        {'user_id': 12346, 'item_id': 67891, 'cate_id': 2, 'brand': 101, 'gender': 'F', 'age': 25, 'income': 40000},
        {'user_id': 12347, 'item_id': 67892, 'cate_id': 1, 'brand': 100, 'gender': 'M', 'age': 35, 'income': 60000}
    ]
    predictions = predictor.predict_batch(batch_features)
    for i, (feat, pred) in enumerate(zip(batch_features, predictions)):
        print(f"样本{i + 1}: 预测概率={pred:.4f}, 特征={feat}")
    print()

    # 从CSV推理示例（需要CSV文件）
    print("=" * 80)
    print("从CSV推理示例:")
    print("=" * 80)
    try:
        csv_path = './data/test.csv'
        if os.path.exists(csv_path):
            result_df = predictor.predict_from_csv(csv_path)
            print(f"推理结果前5行:")
            print(result_df.head())
        else:
            print(f"CSV文件不存在: {csv_path}")
    except Exception as e:
        print(f"从CSV推理失败: {e}")
