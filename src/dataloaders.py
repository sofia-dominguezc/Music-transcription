import os
import zarr
import torch
import numpy as np
from math import ceil
from torch import Tensor
from dataclasses import dataclass
from typing import Literal, Iterator
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info

from .preprocessing import SR, PROCESSED_DATASET_PATH


class DistributedTensorDataset(Dataset[tuple[Tensor, Tensor]]):
    """Loading is distributed accross workers, and each caches its own data

    Args:
        zarr_path: path to the zarr store. Includes data, labels and cu_seqlens
        time_chunk: number of seconds in a single item
    """

    def __init__(self, zarr_path: str, time_chunk: float = 4):
        self.zarr_data: zarr.Group = zarr.open_group(zarr_path, mode='r')
        # self.cu_seqlens = self.zarr_data.attrs["cu_seqlens"]  # ignore for now
        self.x_tensor: torch.Tensor
        self.y_tensor: torch.Tensor

        self.pid = -1
        self.time_batch = round(time_chunk * SR / self.zarr_data.attrs["hop_length"])
        self.total_length = self.zarr_data.attrs["num_entries"]

    @property
    def _slice_of_worker(self) -> slice:
        worker_info = get_worker_info()
        if worker_info is None:
            return slice(0, len(self))
        length = ceil(len(self) / worker_info.num_workers)
        offset = worker_info.id * length
        return slice(offset, offset + length + self.time_batch)

    def _cache_worker_data(self):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.x_tensor = torch.from_numpy(self.zarr_data['data'][self._slice_of_worker])
            self.y_tensor = torch.from_numpy(self.zarr_data['labels'][self._slice_of_worker])

    def __len__(self) -> int:
        return self.total_length - (self.time_batch - 1)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:  # local index
        self._cache_worker_data()
        return (
            self.x_tensor[index: index + self.time_batch].float(),
            self.y_tensor[index: index + self.time_batch].long(),
        )


@dataclass(frozen=True, kw_only=True)
class DistributedSampler(Sampler[list[int]]):
    """Sample random local indices for each worker, in order.
    This solves the naive 36% (1/e) repetitions per epoch.
    """

    length: int
    batch_size: int
    num_processes: int

    def __len__(self) -> int:
        return self.length

    def _num_items(self, worker: int) -> int:
        local_length = ceil(self.length / self.num_processes)
        offset = worker * local_length
        return min(self.length, offset + local_length) - offset

    def __iter__(self) -> Iterator[list[int]]:
        permutations = [
            torch.randperm(self._num_items(worker)).tolist()
            for worker in range(self.num_processes)
        ]
        for local_idx in range(0, len(permutations[0]), self.batch_size):
            for worker in range(self.num_processes):
                if local_idx >= len(permutations[worker]):
                    continue
                yield permutations[worker][local_idx: local_idx + self.batch_size]


class LazyTensorDataset(Dataset[tuple[Tensor, ...]]):
    """Stores a list of files (e.g. ["1727_0", "1728_5"]) and loads in __getitem__"""
    def __init__(
        self,
        files: list[str],
        split: Literal["train", "test"],
        root: str = PROCESSED_DATASET_PATH,
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
        x_file = os.path.join(self.root, f"{self.split}_data", f"{self.files[index]}.npy")
        x_array = np.load(x_file)
        x_tensor = torch.from_numpy(x_array).to(torch.float32)

        y_file = os.path.join(self.root, f"{self.split}_labels", f"{self.files[index]}.npy")
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


def create_distributed_dataloader(
    split: Literal["train", "test"],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """
    Make dataloader from a list of song files.
    If split='train', it will suffle batches even among different songs
    """
    torch.cuda.empty_cache()
    zarr_path = f"{PROCESSED_DATASET_PATH}\\{split}.zarr"
    dataset = DistributedTensorDataset(zarr_path)
    return DataLoader(
        dataset,
        batch_sampler=DistributedSampler(
            length=len(dataset),
            batch_size=batch_size,
            num_processes=num_workers or 1,
        ),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=1,
    )


def create_lazy_dataloader(
    split: Literal["train", "test"],
    batch_size: int,
    num_workers: int = 0,
    root: str = PROCESSED_DATASET_PATH,
) -> DataLoader[tuple[Tensor, ...]]:
    """
    Make "lazy" dataloader from a list of song files.
    Note that each element in the dataloader will be a whole song.
    """
    torch.cuda.empty_cache()

    files = []
    for f in os.listdir(os.fsencode(f"{root}\\{split}_data")):
        file = os.fsdecode(f)
        data_id, ext = file.split('.')
        assert ext == "npy", f"Invalid file: {file}"
        files.append(data_id)
    dataset = LazyTensorDataset(files, split, root=root)

    worker_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    } if num_workers > 0 else {}
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=(split=="train"),
        collate_fn=collate_samples,
        **worker_args,
    )
    return dataloader
