import os
import logging
from config.config import Config
from data.dataloader import DataLoader
from models.tcn_mlp import TCN_MLP_Model
from trainers.trainer import Trainer
from models.lstm_mlp import LSTM_MLP_Model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting program...")
    
    # 初始化配置
    config = Config()
    logger.info(f"Data path: {config.data_path}")
    
    # 检查数据文件
    if not os.path.exists(config.data_path):
        logger.error(f"Data file not found at {config.data_path}")
        return
    
    try:
        # 创建数据加载器
        logger.info("Creating data loaders...")
        data_loader = DataLoader(config)
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()
        
        # 创建模型
        logger.info("Creating model...")
        model = LSTM_MLP_Model(config)
        
        # 创建训练器并开始训练
        logger.info("Starting training...")
        trainer = Trainer(model, config)
        trainer.train(train_loader, val_loader, test_loader)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()