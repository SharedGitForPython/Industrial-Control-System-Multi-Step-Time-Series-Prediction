# data/dataloader.py
import mindspore.dataset as ds
import numpy as np
from data.dataset import TimeSeriesDataset

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset = TimeSeriesDataset(config)
        
    def _create_loader(self, X, Y, shuffle=False):
        """
        创建数据加载器
        Args:
            X: 输入数据
            Y: 标签数据
            shuffle: 是否打乱数据
        Returns:
            dataset: MindSpore数据集对象
        """
        dataset = ds.GeneratorDataset(
            list(zip(X.astype(np.float32), Y.astype(np.float32))),
            column_names=['data', 'label'],
            shuffle=shuffle
        )
        return dataset.batch(self.config.batch_size)

    def get_data_loaders(self):
        """
        获取训练、验证和测试数据加载器
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        # 生成序列数据
        X, Y = self.dataset.generate_sequences(30, self.config.horizon)
        N = X.shape[0]
        
        # 划分数据集
        train_X, train_Y = X[:int(N*0.6)], Y[:int(N*0.6)]
        val_X, val_Y = X[int(N*0.6):int(N*0.8)], Y[int(N*0.6):int(N*0.8)]
        test_X, test_Y = X[int(N*0.8):], Y[int(N*0.8):]
        
        # 创建数据加载器
        train_loader = self._create_loader(train_X, train_Y, shuffle=True)
        val_loader = self._create_loader(val_X, val_Y)
        test_loader = self._create_loader(test_X, test_Y)
        
        return train_loader, val_loader, test_loader