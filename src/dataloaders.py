import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess_data import save_path


class GroupedTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Stores a list of numpy arrays with mmap_mode=r"""
    def __init__(
        self, x_data: list[np.ndarray], y_data: list[np.ndarray],
    ) -> None:
        assert len(x_data) == len(y_data), "Number of groups doesn't match"
        self.x_data = x_data
        self.y_data = y_data
        group_lengths = [a.shape[0] for a in x_data]
        self.group_idx = np.repeat(
            np.arange(len(x_data)),
            group_lengths,
        )
        self.cum_lengths = np.cumsum([0] + group_lengths)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, index):
        group_index = self.group_idx[index]
        local_index = index - self.cum_lengths[group_index]
        x_tensor = torch.from_numpy(
            self.x_data[group_index][local_index]
        ).to(torch.float32)
        y_tensor = torch.from_numpy(
            self.y_data[group_index][local_index]
        ).to(torch.float64)
        return x_tensor, y_tensor


def create_dataloader(
    split: str,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """
    Make dataloader from a list of song files.
    If split='train', it will suffle batches even among different songs
    """
    assert split in ["train", "test"], "Invalid split"
    torch.cuda.empty_cache()

    x_array, y_array = [], []
    for f in os.listdir(os.fsencode(f"{save_path}\\{split}_data")):
        file = os.fsdecode(f)
        assert file.endswith(".npy"), f"Invalid file encountered: {file}"
        song_vals = np.load(f"{save_path}\\{split}_data\\{file}")
        labels = np.load(f"{save_path}\\{split}_labels\\{file}")
        x_array.append(song_vals)
        y_array.append(labels)
    dataset = GroupedTensorDataset(x_array, y_array)

    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    } if num_workers > 0 else {}
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split=="train"), **workers_args,
    )
    return dataloader
