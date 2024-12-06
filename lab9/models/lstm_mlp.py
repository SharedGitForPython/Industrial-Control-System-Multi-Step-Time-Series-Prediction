import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class LSTM_MLP_Model(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Spatial MLP - keeping this part as it handles spatial dependencies
        self.spatial_mlp = nn.SequentialCell([
            nn.Dense(config.sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, config.sensor_num)
        ])
        self.lstm_hidden_size = 64
        self.lstm = nn.LSTM(
            input_size=config.sensor_num,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Adjust final layers for LSTM output
        self.projection = nn.Dense(self.lstm_hidden_size * 2, config.sensor_num)
        self.step_embedding = nn.Embedding(config.horizon, config.sensor_num)
        
        # Final prediction layer
        self.final_layer = nn.Dense(config.sensor_num * 2, config.sensor_num)

    def construct(self, x, iter_step):
        # Spatial feature extraction
        h1 = self.spatial_mlp(x)  # [batch_size, seq_len, sensor_num]
        x = x + h1  # residual connection
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        y1 = self.projection(lstm_out[:, -1:, :])  # [batch_size, 1, sensor_num]
        
        # Step embedding for horizon-aware predictions
        step_embed = self.step_embedding(
            ms.Tensor([iter_step] * x.shape[0], ms.int32)
        )  # [batch_size, sensor_num]
        step_embed = ops.expand_dims(step_embed, 1)  # [batch_size, 1, sensor_num]
        
        # Combine predictions
        combined = ops.concat((y1, step_embed), axis=-1)  # [batch_size, 1, sensor_num*2]
        final_pred = self.final_layer(combined)  # [batch_size, 1, sensor_num]
        
        return final_pred