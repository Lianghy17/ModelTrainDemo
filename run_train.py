import os
import sys
import json
import shutil
from datetime import datetime
import logging

from TrainFramework.train_pipeline import TrainPipeline
from TrainFramework.preprocess import FeatureProcessor




def setup_logging(output_dir, log_level=logging.INFO):
    """配置日志系统"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def list_available_configs(configs_root):
    """列出所有可用的训练配置"""
    if not os.path.exists(configs_root):
        print(f"Configs directory not found: {configs_root}")
        return []

    config_dirs = []
    for item in os.listdir(configs_root):
        item_path = os.path.join(configs_root, item)
        if os.path.isdir(item_path):
            # 检查是否包含必要的配置文件
            train_config = os.path.join(item_path, 'train_config.json')
            feature_config = os.path.join(item_path, 'features_process.json')
            if os.path.exists(train_config) and os.path.exists(feature_config):
                config_dirs.append(item)
                print(f"  - {item}")

    return config_dirs


def load_config(config_dir, logger):
    """加载训练配置和特征配置"""
    train_config_path = os.path.join(config_dir, 'train_config.json')
    feature_config_path = os.path.join(config_dir, 'features_process.json')

    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Train config not found: {train_config_path}")
    if not os.path.exists(feature_config_path):
        raise FileNotFoundError(f"Feature config not found: {feature_config_path}")

    with open(train_config_path, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    with open(feature_config_path, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)

    logger.info(f"Loaded train config from: {train_config_path}")
    logger.info(f"Loaded feature config from: {feature_config_path}")

    return train_config, feature_config


def flatten_train_config(train_config):
    """将嵌套的训练配置展平，保持向后兼容"""
    flat_config = {
        'model_name': train_config['model_config']['model_name'],
        'embedding_dim': train_config['model_config']['embedding_dim'],
        'hidden_units': train_config['model_config']['hidden_units'],
        'dropout_rate': train_config['model_config']['dropout_rate'],
        'epochs': train_config['training_config']['epochs'],
        'batch_size': train_config['training_config']['batch_size'],
        'learning_rate': train_config['training_config']['learning_rate'],
        'device': train_config['training_config']['device'],
        'early_stopping_patience': train_config['task_config']['early_stopping']['patience'],
        'data_paths': train_config['task_config']['data_paths'],
        'column_config': train_config['task_config'].get('data_columns', {
            'id': 'id',
            'features': 'features',
            'label': 'label'
        })
    }
    return flat_config


def create_output_directory(output_dir, config_dir, logger):
    """创建输出目录并复制配置文件"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'configs'), exist_ok=True)

    config_backup_dir = os.path.join(output_dir, 'configs')
    if os.path.exists(config_backup_dir):
        shutil.rmtree(config_backup_dir)
    shutil.copytree(config_dir, config_backup_dir)

    logger.info(f"Created output directory: {output_dir}")
    logger.info(f"Backed up config files to: {config_backup_dir}")


def save_training_metadata(output_dir, config_name, timestamp, train_config):
    """保存训练元数据"""
    metadata = {
        'config_name': config_name,
        'timestamp': timestamp,
        'model_name': train_config['model_config']['model_name'],
        'epochs': train_config['training_config']['epochs'],
        'batch_size': train_config['training_config']['batch_size'],
        'learning_rate': train_config['training_config']['learning_rate']
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main(config_name, config_dir, train_config, feature_config):
    """主函数：执行训练流程"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('./trained_models_dir', f"{config_name}_{timestamp}")

    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("Starting Training Pipeline")
    logger.info("="*80)

    try:
        create_output_directory(output_dir, config_dir, logger)
        save_training_metadata(output_dir, config_name, timestamp, train_config)

        flat_train_config = flatten_train_config(train_config)

        logger.info(f"Config: {config_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {flat_train_config['model_name']}")
        logger.info(f"Epochs: {flat_train_config['epochs']}")
        logger.info(f"Batch size: {flat_train_config['batch_size']}")
        logger.info(f"Learning rate: {flat_train_config['learning_rate']}")
        logger.info(f"Data paths:")
        logger.info(f"  - Train: {flat_train_config['data_paths']['train']}")
        logger.info(f"  - Validation: {flat_train_config['data_paths']['validation']}")
        logger.info(f"  - Test: {flat_train_config['data_paths']['test']}")

        feature_processor = FeatureProcessor(feature_config)
        logger.info(f"Initialized FeatureProcessor with {len(feature_config)} features")

        trainer = TrainPipeline(
            train_config=flat_train_config,
            feature_processor=feature_processor,
            output_dir=output_dir
        )

        logger.info("Starting model training...")
        trainer.run_training()

        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved in: {output_dir}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # 配置训练参数
    TRAIN_CONFIG = {
        'config_name': 'v1',  # train_configs 下的配置文件夹名称
        'configs_root': './train_configs'  # 配置文件根目录
    }


    # 显示当前使用的配置
    print(f"\n当前配置: {TRAIN_CONFIG['config_name']}")
    print(f"配置路径: {os.path.join(TRAIN_CONFIG['configs_root'], TRAIN_CONFIG['config_name'])}")

    # 显示可用配置
    print("\n可用的训练配置:")
    available_configs = list_available_configs(TRAIN_CONFIG['configs_root'])
    if not available_configs:
        print("  (无可用配置)")

    print("\n提示: 如需切换配置，请修改文件顶部的 TRAIN_CONFIG 字典")
    print("="*80 + "\n")

    # 加载配置
    config_name = TRAIN_CONFIG['config_name']
    config_dir = os.path.join(TRAIN_CONFIG['configs_root'], config_name)

    if not os.path.exists(config_dir):
        raise ValueError(f"Configuration folder {config_dir} does not exist. Please check TRAIN_CONFIG in run_train.py")

    print(f"正在加载配置文件...")
    train_config_path = os.path.join(config_dir, 'train_config.json')
    feature_config_path = os.path.join(config_dir, 'features_process.json')

    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Train config not found: {train_config_path}")
    if not os.path.exists(feature_config_path):
        raise FileNotFoundError(f"Feature config not found: {feature_config_path}")

    with open(train_config_path, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    with open(feature_config_path, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)


    # 调用 main 函数执行训练
    main(config_name, config_dir, train_config, feature_config)
