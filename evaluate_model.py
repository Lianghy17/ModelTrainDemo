import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import logging

from predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估类"""

    def __init__(self, model_dir: str, checkpoint_name: str = 'best_model.pth', device: str = None):
        """
        初始化评估器

        Args:
            model_dir: 模型目录
            checkpoint_name: checkpoint文件名
            device: 设备类型
        """
        logger.info(f"Loading model from: {model_dir}")
        self.predictor = Predictor(model_dir, checkpoint_name, device)

    def evaluate_from_csv(self, csv_path: str, features_col: str = 'features',
                          label_col: str = 'label', id_col: str = 'id',
                          threshold: float = 0.5, batch_size: int = 32) -> dict:
        """
        从CSV文件评估模型

        Args:
            csv_path: CSV文件路径
            features_col: 特征列名
            label_col: 标签列名
            id_col: ID列名
            threshold: 分类阈值
            batch_size: 批次大小

        Returns:
            dict: 评估指标字典
        """
        logger.info(f"Loading test data from: {csv_path}")

        # 加载CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")

        # 检查列是否存在
        required_cols = [features_col, label_col, id_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

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
        logger.info(f"Running predictions with batch_size={batch_size}...")
        predictions = self.predictor.predict_batch(features_list, batch_size=batch_size)
        predictions = np.array(predictions)

        # 获取真实标签
        true_labels = df[label_col].values

        # 计算评估指标
        metrics = self._calculate_metrics(true_labels, predictions, threshold)

        # 保存预测结果
        results_df = pd.DataFrame({
            id_col: df[id_col],
            label_col: true_labels,
            'prediction': predictions,
            'predicted_label': (predictions >= threshold).astype(int)
        })
        results_path = os.path.join(os.path.dirname(csv_path), 'prediction_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Prediction results saved to: {results_path}")

        # 打印详细报告
        self._print_report(metrics, results_df, threshold)

        return metrics

    def _calculate_metrics(self, true_labels: np.ndarray, predictions: np.ndarray,
                          threshold: float = 0.5) -> dict:
        """计算评估指标"""
        metrics = {}

        # 转换为预测标签
        pred_labels = (predictions >= threshold).astype(int)

        # 基础指标
        metrics['AUC'] = roc_auc_score(true_labels, predictions)
        metrics['LogLoss'] = log_loss(true_labels, predictions)
        metrics['Accuracy'] = accuracy_score(true_labels, pred_labels)
        metrics['Precision'] = precision_score(true_labels, pred_labels)
        metrics['Recall'] = recall_score(true_labels, pred_labels)
        metrics['F1-Score'] = f1_score(true_labels, pred_labels)

        # 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        metrics['Confusion_Matrix'] = cm
        metrics['True_Positive'] = cm[1, 1]
        metrics['True_Negative'] = cm[0, 0]
        metrics['False_Positive'] = cm[0, 1]
        metrics['False_Negative'] = cm[1, 0]

        # 计算其他指标
        tn, fp, fn, tp = cm.ravel()
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['Negative_Predictive_Value'] = tn / (tn + fn) if (tn + fn) > 0 else 0

        # 分位数信息
        metrics['Prediction_Mean'] = float(np.mean(predictions))
        metrics['Prediction_Std'] = float(np.std(predictions))
        metrics['Prediction_Min'] = float(np.min(predictions))
        metrics['Prediction_Max'] = float(np.max(predictions))
        metrics['Prediction_Median'] = float(np.median(predictions))

        return metrics

    def _print_report(self, metrics: dict, results_df: pd.DataFrame, threshold: float):
        """打印评估报告"""
        print("\n" + "=" * 80)
        print("模型评估报告".center(80))
        print("=" * 80)

        # 基础指标
        print("\n【基础指标】")
        print("-" * 80)
        print(f"{'指标':<30} {'值':<20} {'说明'}")
        print("-" * 80)
        print(f"{'AUC (ROC)':<30} {metrics['AUC']:<20.4f} {'模型区分正负样本的能力'}")
        print(f"{'LogLoss (对数损失)':<30} {metrics['LogLoss']:<20.4f} {'越小越好，完美分类为0'}")
        print(f"{'Accuracy (准确率)':<30} {metrics['Accuracy']:<20.4f} {'整体预测准确度'}")
        print(f"{'Precision (精确率)':<30} {metrics['Precision']:<20.4f} {'预测为正的真实比例'}")
        print(f"{'Recall (召回率)':<30} {metrics['Recall']:<20.4f} {'正样本被正确预测的比例'}")
        print(f"{'F1-Score':<30} {metrics['F1-Score']:<20.4f} {'精确率和召回率的调和平均'}")

        # 混淆矩阵
        print("\n【混淆矩阵】")
        print("-" * 80)
        cm = metrics['Confusion_Matrix']
        print(f"{'':<15} {'预测为正':<15} {'预测为负':<15}")
        print("-" * 80)
        print(f"{'真实为正':<15} {cm[1, 1]:<15} {cm[1, 0]:<15}")
        print(f"{'真实为负':<15} {cm[0, 1]:<15} {cm[0, 0]:<15}")
        print("-" * 80)
        print(f"{'True Positive (TP)':<30} {metrics['True_Positive']:<20}")
        print(f"{'True Negative (TN)':<30} {metrics['True_Negative']:<20}")
        print(f"{'False Positive (FP)':<30} {metrics['False_Positive']:<20}")
        print(f"{'False Negative (FN)':<30} {metrics['False_Negative']:<20}")

        # 其他指标
        print("\n【其他指标】")
        print("-" * 80)
        print(f"{'Sensitivity (灵敏度)':<30} {metrics['Sensitivity']:<20.4f}")
        print(f"{'Specificity (特异度)':<30} {metrics['Specificity']:<20.4f}")
        print(f"{'Negative Predictive Value':<30} {metrics['Negative_Predictive_Value']:<20.4f}")

        # 预测值统计
        print("\n【预测值统计】")
        print("-" * 80)
        print(f"{'均值':<15} {metrics['Prediction_Mean']:<15.4f}")
        print(f"{'标准差':<15} {metrics['Prediction_Std']:<15.4f}")
        print(f"{'最小值':<15} {metrics['Prediction_Min']:<15.4f}")
        print(f"{'最大值':<15} {metrics['Prediction_Max']:<15.4f}")
        print(f"{'中位数':<15} {metrics['Prediction_Median']:<15.4f}")

        # 分类报告
        print("\n【详细分类报告】")
        print("-" * 80)
        print(classification_report(results_df['label'], results_df['predicted_label'],
                                   digits=4, target_names=['负样本(0)', '正样本(1)']))

        # 分布信息
        print("\n【预测分布】")
        print("-" * 80)
        pred_dist = results_df['predicted_label'].value_counts()
        print(f"{'预测为负样本数量':<30} {pred_dist.get(0, 0):<20}")
        print(f"{'预测为正样本数量':<30} {pred_dist.get(1, 0):<20}")
        print(f"{'真实负样本数量':<30} {(results_df['label'] == 0).sum():<20}")
        print(f"{'真实正样本数量':<30} {(results_df['label'] == 1).sum():<20}")

        print("\n" + "=" * 80)

    def evaluate_with_multiple_thresholds(self, csv_path: str, features_col: str = 'features',
                                          label_col: str = 'label', id_col: str = 'id',
                                          thresholds: list = None, batch_size: int = 32) -> pd.DataFrame:
        """
        使用多个阈值进行评估

        Args:
            csv_path: CSV文件路径
            features_col: 特征列名
            label_col: 标签列名
            id_col: ID列名
            thresholds: 阈值列表
            batch_size: 批次大小

        Returns:
            pd.DataFrame: 包含不同阈值下评估结果的DataFrame
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        logger.info(f"Evaluating with thresholds: {thresholds}")

        # 加载数据并预测
        df = pd.read_csv(csv_path)
        features_list = []
        for idx, row in df.iterrows():
            features = json.loads(row[features_col])
            features_list.append(features)

        predictions = np.array(self.predictor.predict_batch(features_list, batch_size=batch_size))
        true_labels = df[label_col].values

        # 计算每个阈值的指标
        results = []
        for threshold in thresholds:
            pred_labels = (predictions >= threshold).astype(int)

            metrics = {
                'Threshold': threshold,
                'Accuracy': accuracy_score(true_labels, pred_labels),
                'Precision': precision_score(true_labels, pred_labels),
                'Recall': recall_score(true_labels, pred_labels),
                'F1-Score': f1_score(true_labels, pred_labels),
                'True_Positive': confusion_matrix(true_labels, pred_labels)[1, 1],
                'True_Negative': confusion_matrix(true_labels, pred_labels)[0, 0],
                'False_Positive': confusion_matrix(true_labels, pred_labels)[0, 1],
                'False_Negative': confusion_matrix(true_labels, pred_labels)[1, 0]
            }
            results.append(metrics)

        results_df = pd.DataFrame(results)

        # 打印结果
        print("\n" + "=" * 80)
        print("不同阈值下的评估结果".center(80))
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)

        return results_df


if __name__ == '__main__':
    # 配置
    model_dir = './trained_models_dir/v1_20260226_172330'  # 模型目录
    checkpoint_name = 'best_model.pth'  # checkpoint文件名
    test_file = './data/test.csv'  # 测试文件路径
    threshold = 0.5  # 分类阈值
    batch_size = 32  # 批次大小
    device = None  # 设备类型 (None=自动选择, 'cpu', 'cuda')
    multi_threshold = False  # 是否使用多阈值评估

    # 初始化评估器
    evaluator = ModelEvaluator(model_dir, checkpoint_name, device)

    if multi_threshold:
        # 多阈值评估
        evaluator.evaluate_with_multiple_thresholds(
            test_file,
            threshold=threshold,
            batch_size=batch_size
        )
    else:
        # 单阈值评估
        evaluator.evaluate_from_csv(
            test_file,
            threshold=threshold,
            batch_size=batch_size
        )
