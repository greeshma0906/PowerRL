import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class CustomDataset(Dataset):
    def __init__(self, h5_file):
        """
        Dataset for loading preprocessed code summarization data from an HDF5 file.
        """
        self.h5_file = h5_file  # Store the file path
        with h5py.File(h5_file, "r") as h5f:
            self.dataset_size = len(h5f["X"])  # Get dataset size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Returns a single data point (X, Y) from the dataset.
        """
        with h5py.File(self.h5_file, "r") as h5f:
            x = torch.tensor(h5f["X"][idx], dtype=torch.long)  # Source sequence
            y = torch.tensor(h5f["Y"][idx], dtype=torch.long)  # Target sequence
        return x, y

def collate_fn(batch):
    """
    Pads sequences to the maximum length in the batch.
    """
    x_batch, y_batch = zip(*batch)

    # Get max sequence lengths in batch
    x_max_len = max(x.shape[0] for x in x_batch)
    y_max_len = max(y.shape[0] for y in y_batch)

    # Initialize padded tensors
    x_padded = torch.zeros(len(batch), x_max_len, dtype=torch.long)
    y_padded = torch.zeros(len(batch), y_max_len, dtype=torch.long)

    # Apply padding
    for i, (x, y) in enumerate(batch):
        x_padded[i, :x.shape[0]] = x
        y_padded[i, :y.shape[0]] = y

    return x_padded, y_padded

def get_dataloader(h5_file, batch_size=32, shuffle=True):
    """
    Returns a DataLoader for the dataset.
    """
    dataset = CustomDataset(h5_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)




