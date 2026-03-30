import torch
import json
import numpy as np
from TrainFramework.preprocess import FeatureProcessor


# 在tools.py中添加更完善的预测工具
class BatchPredictor:
    def __init__(self, model_path, processor_path, model_config, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载特征处理器
        self.feature_processor = FeatureProcessor({})
        self.feature_processor.load_processor(processor_path)

        # 根据配置创建模型
        model_name = model_config['model_name']
        feature_dims = model_config['feature_dims']
        embedding_dim = model_config['embedding_dim']
        hidden_units = model_config['hidden_units']
        dropout_rate = model_config['dropout_rate']

        # 创建模型实例
        if model_name == 'DeepFM':
            from TrainFramework.models.deepfm import DeepFM
            self.model = DeepFM(feature_dims, embedding_dim, hidden_units, dropout_rate)
        elif model_name == 'MLP':
            from TrainFramework.models.deepfm import MLP
            self.model = MLP(feature_dims, embedding_dim, hidden_units, dropout_rate)
        elif model_name == 'WideAndDeep':
            from TrainFramework.models.deepfm import WideAndDeep
            self.model = WideAndDeep(feature_dims, embedding_dim, hidden_units, dropout_rate)

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def predict_batch(self, df, batch_size=1000):
        """批量预测"""
        predictions = []

        # 分批处理大数据集
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_predictions = self._predict_single_batch(batch_df)
            predictions.extend(batch_predictions)

        return np.array(predictions)

    def _predict_single_batch(self, df):
        """单批次预测"""
        # 处理特征
        features = self.feature_processor.transform(df)

        # 转换为tensor
        feature_tensors = {}
        for feature_name, feature_values in features.items():
            feature_tensors[feature_name] = torch.tensor(
                feature_values, dtype=torch.float32
            ).unsqueeze(1).to(self.device)  # 添加batch维度

        # 预测
        with torch.no_grad():
            outputs = self.model(feature_tensors)
            batch_predictions = outputs.squeeze().cpu().numpy()

        return batch_predictions.tolist()


class ModelPredictor:
    def __init__(self, model_path, processor_path, model_class, feature_dims, device='cpu'):
        self.device = torch.device(device)

        # 加载特征处理器
        self.feature_processor = FeatureProcessor({})
        self.feature_processor.load_processor(processor_path)

        # 加载模型
        self.model = model_class(feature_dims)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df):
        """对DataFrame进行预测"""
        # 处理特征
        features = self.feature_processor.transform(df)

        # 转换为tensor
        feature_tensors = {}
        for feature_name, feature_values in features.items():
            feature_tensors[feature_name] = torch.tensor(feature_values, dtype=torch.float32).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(feature_tensors)
            predictions = outputs.squeeze().cpu().numpy()

        return predictions


def load_training_history(history_path):
    """加载训练历史"""
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_training_history(history):
    """绘制训练历史曲线"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['loss'], label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # 验证指标曲线
    ax2.plot(history['val_auc'], label='Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig
