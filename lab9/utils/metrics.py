import numpy as np

def compute_metrics(pred, target):
    """Compute evaluation metrics"""
    mae = np.mean(np.abs(pred - target))
    return mae