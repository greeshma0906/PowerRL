import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class CustomDataset(Dataset):
    def __init__(self, h5_file):
        """
        Custom Dataset to load X (code representations) and Y (descriptions).
        """
        self.h5f = h5py.File(h5_file, "r")
        self.X = self.h5f["X"]
        self.Y = self.h5f["Y"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y
