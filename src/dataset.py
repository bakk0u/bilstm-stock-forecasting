import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def make_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    """
    features: [N, F]
    target:   [N,]
    Return:
      X: [N-lookback, lookback, F]
      y: [N-lookback,]
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(target[i])
    return np.asarray(X), np.asarray(y)
