import os
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from typing import Literal

from preprocess_data import save_path

max_time = 90 * (22050 // 512)  # max length of batch is 90s


class GroupedTensorDataset(Dataset[tuple[Tensor, Tensor]]):
    """Stores a list of numpy arrays with mmap_mode=r"""
    def __init__(
        self,
        x_data: list[np.ndarray],
        y_data: list[np.ndarray],
    ) :
        assert len(x_data) == len(y_data), "Number of groups doesn't match"
        self.x_data = x_data
        self.y_data = y_data
        group_lengths = [a.shape[0] for a in x_data]
        self.group_idx = np.repeat(
            np.arange(len(x_data)),
            group_lengths,
        )
        self.cum_lengths = np.cumsum([0] + group_lengths)

    def __len__(self) -> int:
        return self.cum_lengths[-1]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        group_index = self.group_idx[index]
        local_index = index - self.cum_lengths[group_index]
        x_tensor = torch.from_numpy(
            self.x_data[group_index][local_index]
        ).to(torch.float32)
        y_tensor = torch.from_numpy(
            self.y_data[group_index][local_index]
        ).to(torch.float64)
        return x_tensor, y_tensor


class LazyTensorDataset(Dataset[tuple[Tensor, ...]]):
    """Stores a list of files (e.g. ["1727", "1728"]) and loads in __getitem__"""
    def __init__(
        self,
        files: list[str],
        split: Literal["train", "test"],
        root: str = save_path,
    ):
        self.files = files
        self.split = split
        self.root = root

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """
        Returns:
            x_tensor: (batch, time, notes)
            y_tensor: (batch, time, notes)
        """
        x_file = f"{self.root}\\{self.split}_data\\{self.files[index]}.npy"
        x_array = np.load(x_file)
        x_tensor = torch.from_numpy(x_array).to(torch.float32)

        y_file = f"{self.root}\\{self.split}_labels\\{self.files[index]}.npy"
        y_array = np.load(y_file)
        y_tensor = torch.from_numpy(y_array).to(torch.float64)

        return x_tensor, y_tensor

def collate_samples(batches: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """
    Concatenate tensors along the first dimension
    batches: [(x1, y1, ...), (x2, y2, ...), ...]
    output: (X, Y, ...)
    """
    output = []
    for x in zip(*batches):
        output.append(torch.cat(x, dim=0))
    return tuple(output)


def create_dataloader(
    split: Literal["train", "test"],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """
    Make dataloader from a list of song files.
    If split='train', it will suffle batches even among different songs
    """
    torch.cuda.empty_cache()

    x_array, y_array = [], []
    for f in os.listdir(os.fsencode(f"{save_path}\\{split}_data")):
        file = os.fsdecode(f)
        assert file.endswith(".npy"), f"Invalid file: {file}"
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
        dataset, batch_size, shuffle=(split=="train"), **workers_args,
    )
    return dataloader


def create_lazy_dataloader(
    split: Literal["train", "test"],
    batch_size: int,
    num_workers: int = 0,
    root: str = save_path,
) -> DataLoader[tuple[Tensor, ...]]:
    """
    Make "lazy" dataloader from a list of song files.
    Note that each element in the dataloader will be a whole song.
    """
    torch.cuda.empty_cache()

    files = []
    for f in os.listdir(os.fsencode(f"{root}\\{split}_data")):
        file = os.fsdecode(f)
        assert file.endswith(".npy"), f"Invalid file: {file}"
        files.append(file[:-len(".npy")])
    dataset = LazyTensorDataset(files, split, root=root)

    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 12,
    } if num_workers > 0 else {}
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=(split=="train"),
        collate_fn=collate_samples,
        **workers_args,
    )
    return dataloader
