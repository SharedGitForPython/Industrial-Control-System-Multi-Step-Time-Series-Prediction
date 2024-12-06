# models/tcn_mlp.py
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class TCN_MLP_Model(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Spatial MLP
        self.spatial_mlp = nn.SequentialCell([
            nn.Dense(config.sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, config.sensor_num)
        ])

        # Temporal CNN
        self.tcn = nn.SequentialCell([
            nn.Conv2d(1, 1, (3,1), pad_mode='valid'),
            nn.Conv2d(1, 1, (3,1), pad_mode='valid')
        ])

        self.final_conv1 = nn.Conv2d(1, 1, (26,1), pad_mode='valid')
        self.step_embedding = nn.Embedding(config.horizon, config.sensor_num)
        self.final_conv2 = nn.Conv2d(1, 1, (28,1), pad_mode='valid')

    def construct(self, x, iter_step):
        # Base prediction
        h1 = self.spatial_mlp(x)  # [batch_size, seq_len, sensor_num]
        x = x + h1
        x1 = ops.expand_dims(x, 1)  # [batch_size, 1, seq_len, sensor_num]
        x1 = self.tcn(x1)
        y1 = self.final_conv1(x1)
        y1 = ops.squeeze(y1, axis=1)  # [batch_size, 1, sensor_num]

        # Residual prediction 
        step_embed = self.step_embedding(
            ms.Tensor([iter_step] * x.shape[0], ms.int32)
        )  # [batch_size, sensor_num]
        step_embed = ops.expand_dims(step_embed, 1)  # [batch_size, 1, sensor_num]
        
        x2 = ops.concat((x, step_embed, y1), axis=1)  # [batch_size, seq_len+2, sensor_num]
        x2 = ops.expand_dims(x2, 1)  # [batch_size, 1, seq_len+2, sensor_num]
        x2 = self.tcn(x2)
        y2 = self.final_conv2(x2)
        y2 = ops.squeeze(y2, axis=1)  # [batch_size, 1, sensor_num]

        return y1 + y2  # [batch_size, 1, sensor_num]