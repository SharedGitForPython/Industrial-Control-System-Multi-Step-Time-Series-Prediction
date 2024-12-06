import numpy as np

class TimeSeriesDataset:
    def __init__(self, config):
        self.config = config
        
        # 加载数据
        self.data = np.loadtxt(
            config.data_path,
            delimiter=',', 
            skiprows=1,
            usecols=range(1, config.sensor_num+1)
        )
        
    def generate_sequences(self, x_len, y_len):
        """
        生成时间序列训练数据
        Args:
            x_len: 输入序列长度
            y_len: 输出序列长度
        Returns:
            X: 输入序列数据
            Y: 输出序列数据
        """
        point_num = self.data.shape[0]
        sample_num = point_num - x_len - y_len + 1
        X = np.zeros((sample_num, x_len, self.config.sensor_num))
        Y = np.zeros((sample_num, y_len, self.config.sensor_num))
        
        for i in range(sample_num):
            X[i] = self.data[i:i+x_len]
            Y[i] = self.data[i+x_len:i+x_len+y_len]
            
        return X, Y

