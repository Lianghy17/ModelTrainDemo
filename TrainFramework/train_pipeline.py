import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, log_loss
import json
import logging

logger = logging.getLogger(__name__)


class CTRDataset(Dataset):
    """CTR 数据集类 - 支持 JSON 格式的特征"""
    def __init__(self, features_dict, labels):
        """
        Args:
            features_dict: 字典，key 是特征名，value 是特征值数组
            labels: 标签数组
        """
        self.features = features_dict
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_features = {}
        for feature_name, feature_values in self.features.items():
            sample_features[feature_name] = torch.tensor(feature_values[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_features, label


class DataManager:
    """数据管理器：负责数据加载、处理和DataLoader创建（支持 JSON 格式）"""

    def __init__(self, data_paths, feature_processor, batch_size, column_config):
        """
        Args:
            data_paths: 数据文件路径字典
            feature_processor: 特征处理器
            batch_size: 批次大小
            column_config: 列名配置字典
                - id: ID 列名
                - features: features 列名
                - label: label 列名
        """
        self.data_paths = data_paths
        self.feature_processor = feature_processor
        self.batch_size = batch_size
        self.column_config = column_config

    def load_data(self, file_path):
        """加载并预处理数据（支持 JSON 格式）"""
        logger.info(f"Loading data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)

        # 从配置获取列名
        id_col = self.column_config.get('id', 'id')
        features_col = self.column_config.get('features', 'features')
        label_col = self.column_config.get('label', 'label')

        logger.info(f"Column mapping: id={id_col}, features={features_col}, label={label_col}")

        # 检查列名是否存在
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data. Available columns: {df.columns.tolist()}")
        if features_col not in df.columns:
            raise ValueError(f"Features column '{features_col}' not found in data. Available columns: {df.columns.tolist()}")

        # 解析 JSON 格式的 features
        logger.info(f"Parsing JSON features from column '{features_col}'...")
        features_list = []
        for idx, row in df.iterrows():
            try:
                features = json.loads(row[features_col])
                features_list.append(features)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON at row {idx}: {e}")
                raise

        # 将 features 转换为 DataFrame
        features_df = pd.DataFrame(features_list)
        labels = df[label_col].values

        logger.info(f"Loaded {len(df)} samples with {len(features_df.columns)} features")
        return features_df, labels

    def create_dataloaders(self):
        """创建训练和验证的DataLoader"""
        logger.info("Creating dataloaders...")

        train_path = self.data_paths['train']
        val_path = self.data_paths['validation']

        logger.info(f"Loading training data from: {train_path}")
        train_df, train_labels = self.load_data(train_path)
        logger.info("Fitting feature processor...")
        self.feature_processor.fit(train_df)
        train_features = self.feature_processor.transform(train_df)

        train_dataset = CTRDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(f"Loading validation data from: {val_path}")
        val_df, val_labels = self.load_data(val_path)
        val_features = self.feature_processor.transform(val_df)
        val_dataset = CTRDataset(val_features, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader

    def calculate_feature_dims(self):
        """计算特征维度，返回离散特征的维度和连续特征的列表"""
        logger.info("Calculating feature dimensions...")
        train_path = self.data_paths['train']
        logger.info(f"Loading training data from: {train_path}")
        train_df, _ = self.load_data(train_path)
        self.feature_processor.fit(train_df)

        # 分别存储离散特征（需要embedding）和连续特征（不需要embedding）
        discrete_feature_dims = {}
        continuous_features = []

        for feature_name in self.feature_processor.feature_config.keys():
            config = self.feature_processor.feature_config.get(feature_name, {})
            processor_type = config.get('processor', '')

            if processor_type == 'vocab':
                vocab_size = len(self.feature_processor.vocab_maps.get(feature_name, {}))
                discrete_feature_dims[feature_name] = vocab_size + 1
                logger.debug(f"{feature_name}: vocab size = {vocab_size + 1}")
            elif processor_type == 'hash_bucket':
                bucket_info = self.feature_processor.processors.get(feature_name, {})
                if isinstance(bucket_info, dict):
                    bucket_size = bucket_info.get('bucket_size', 1000)
                else:
                    bucket_size = 1000
                discrete_feature_dims[feature_name] = bucket_size
                logger.debug(f"{feature_name}: bucket size = {bucket_size}")
            elif processor_type in ['minmax', 'zscore']:
                continuous_features.append(feature_name)
                logger.debug(f"{feature_name}: continuous feature")

        logger.info(f"Discrete features: {discrete_feature_dims}")
        logger.info(f"Continuous features: {continuous_features}")

        return discrete_feature_dims, continuous_features

    def save_feature_processor(self, output_dir):
        """保存特征处理器"""
        processor_path = os.path.join(output_dir, 'feature_processor.json')
        self.feature_processor.save_processor(processor_path)
        logger.info(f"Feature processor saved to {processor_path}")


class ModelManager:
    """模型管理器：负责模型创建、保存和加载"""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None

    def create_model(self, discrete_feature_dims, continuous_features):
        """根据配置创建模型"""
        model_name = self.config['model_name']
        embedding_dim = self.config['embedding_dim']
        hidden_units = self.config['hidden_units']
        dropout_rate = self.config['dropout_rate']

        logger.info(f"Creating model: {model_name}")
        logger.info(f"Discrete features: {list(discrete_feature_dims.keys())}")
        logger.info(f"Continuous features: {continuous_features}")
        logger.info(f"Embedding dim: {embedding_dim}, Hidden units: {hidden_units}, Dropout: {dropout_rate}")

        try:
            from TrainFramework.models.deepfm import DeepFM, MLP, WideAndDeep

            model_class = {
                'DeepFM': DeepFM,
                'MLP': MLP,
                'WideAndDeep': WideAndDeep
            }.get(model_name)

            if model_class is None:
                raise ValueError(f"Unsupported model: {model_name}")

            self.model = model_class(discrete_feature_dims, continuous_features, embedding_dim, hidden_units, dropout_rate)
            self.model.to(self.device)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )

            logger.info(f"Model created successfully on device: {self.device}")

        except ImportError as e:
            raise ImportError(f"Failed to import model {model_name}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to create model {model_name}: {str(e)}")

    def save_model(self, output_dir, epoch, train_loss, val_auc):
        """保存模型检查点"""
        model_path = os.path.join(output_dir, 'models', f'epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_auc': val_auc
        }, model_path)
        logger.debug(f"Saved checkpoint: {model_path}")

    def save_best_model(self, output_dir):
        """保存最佳模型"""
        best_model_path = os.path.join(output_dir, 'models', 'best_model.pth')
        torch.save(self.model.state_dict(), best_model_path)
        logger.info(f"Saved best model: {best_model_path}")


class Trainer:
    """训练器：负责训练和评估逻辑"""

    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.criterion = nn.BCELoss()
        self.train_history = {'loss': [], 'val_auc': [], 'val_logloss': []}

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_features, batch_labels in train_loader:
            batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
            batch_labels = batch_labels.to(self.device)

            outputs = self.model(batch_features)
            loss = self.criterion(outputs.squeeze(), batch_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                predictions = outputs.squeeze().cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(batch_labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_predictions)
        logloss = log_loss(all_labels, all_predictions)
        return auc, logloss

    def train(self, train_loader, val_loader, model_manager, output_dir):
        """执行训练循环"""
        best_val_auc = 0
        patience_counter = 0

        logger.info(f"Starting training for {self.config['epochs']} epochs...")

        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_auc, val_logloss = self.evaluate(val_loader)

            self.train_history['loss'].append(train_loss)
            self.train_history['val_auc'].append(val_auc)
            self.train_history['val_logloss'].append(val_logloss)

            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}: "
                       f"Loss={train_loss:.4f}, Val_AUC={val_auc:.4f}, Val_LogLoss={val_logloss:.4f}")

            model_manager.save_model(output_dir, epoch + 1, train_loss, val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                model_manager.save_best_model(output_dir)
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return best_val_auc

    def save_training_history(self, output_dir):
        """保存训练历史"""
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")


class TrainPipeline:
    """训练流程协调器：组合各个组件，执行完整训练流程（支持 JSON 格式）"""

    def __init__(self, train_config, feature_processor, output_dir):
        self.train_config = train_config
        self.feature_processor = feature_processor
        self.output_dir = output_dir
        self.device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def run_training(self):
        """执行完整训练流程"""
        try:
            os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)

            data_manager = DataManager(
                self.train_config['data_paths'],
                self.feature_processor,
                self.train_config['batch_size'],
                self.train_config['column_config']
            )
            discrete_feature_dims, continuous_features = data_manager.calculate_feature_dims()

            model_manager = ModelManager(self.train_config, self.device)
            model_manager.create_model(discrete_feature_dims, continuous_features)

            train_loader, val_loader = data_manager.create_dataloaders()
            data_manager.save_feature_processor(self.output_dir)

            trainer = Trainer(
                model_manager.model,
                model_manager.optimizer,
                self.device,
                self.train_config
            )
            best_val_auc = trainer.train(train_loader, val_loader, model_manager, self.output_dir)
            trainer.save_training_history(self.output_dir)

            logger.info(f"Training completed! Best validation AUC: {best_val_auc:.4f}")
            return best_val_auc

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
