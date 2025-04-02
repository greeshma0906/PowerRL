import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class CustomDataset(Dataset):
    def __init__(self, h5_file):
        """
        Custom Dataset to load X (code representations) and Y (descriptions).
        """
        # Open HDF5 file
        with h5py.File(h5_file, "r") as h5f:
            self.X = torch.tensor(h5f["X"][:], dtype=torch.float32)  # Code representations
            self.Y = torch.tensor(h5f["Y"][:], dtype=torch.long)      # Target descriptions

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


