import os
class Config:
    def __init__(self):
        # Project root dir
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Data paths
        self.data_dir = os.path.join(self.root_dir, 'datasets')
        self.data_path = os.path.join(self.data_dir, 'raw', 'jl_data_train.csv')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')
        # Output paths
        self.checkpoint_dir = os.path.join(self.root_dir, 'checkpoints')
        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.exp_dir = os.path.join(self.root_dir, 'experiments', 'exp_results')
        # Create directories
        for path in [self.data_dir, os.path.dirname(self.data_path), 
                    self.processed_data_dir, self.checkpoint_dir, 
                    self.log_dir, self.exp_dir]:
            os.makedirs(path, exist_ok=True)
        # Data params
        self.sensor_num = 23
        self.horizon = 5
        self.PV_index = list(range(9))
        self.OP_index = list(range(9,18))
        self.DV_index = list(range(18,23))
        # Model params
        self.hidden_dims = [128, 64, 32]
        # Training params
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 35
        self.patience = 5