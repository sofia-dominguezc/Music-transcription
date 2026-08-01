import os
import csv
import zarr
import stat
import shutil
from tqdm import tqdm
from enum import Enum
from typing import Iterator
from concurrent import futures
from dataclasses import dataclass

import librosa
import numpy as np

DATASET_PATH = os.path.join("data", "musicnet")
PROCESSED_DATASET_PATH = os.path.join("data", "musicnet_processed")

SR = 22050
N_OCTAVES = 8
TIME_CHUNK = 1024

class Split(Enum):
    TRAIN = "train"
    TEST = "test"


def _onexc_remove(func, path, exc):
    """Remove read only files (useful for Windows One Drive)"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def load_song(song: str, split: Split) -> np.ndarray:
    song_path = f"{DATASET_PATH}\\{split.value}_data\\{song}.wav"
    song_vals, song_sr = librosa.load(song_path)
    assert song_sr == SR, f"invalid sr {song_sr}"
    return song_vals


def normalized_cqt(
    raw_song: np.ndarray, *, hop_length: int, bins_per_note: int
) -> np.ndarray:
    spect = librosa.cqt(
        raw_song,
        sr=SR,
        hop_length=hop_length,
        n_bins=N_OCTAVES*bins_per_note*12,
        bins_per_octave=bins_per_note*12,
        filter_scale=0.5,
        scale=True,
    )
    return np.abs(spect.T)**0.3  # (time, freq)


def load_labels(song: str, split: Split) -> Iterator[tuple[int, int, int]]:
    song_path = f"{DATASET_PATH}\\{split.value}_labels\\{song}.csv"
    with open(song_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield int(row['start_time']), int(row['end_time']), int(row['note'])


def get_one_hot_labels(
    raw_labels: Iterator[tuple[int, int, int]],
    *,
    time_length: int,
    hop_length: int,
    only_note_names: bool,
) -> np.ndarray:
    """
    Boolean array of if a note is on or not at each of `time_length` times.
    Uses `hop_length` as stretch factor, i.e., `song_length ~ time_length * hop_length`
    """
    n_notes = 12 * N_OCTAVES if not only_note_names else 12
    labels = np.full((time_length, n_notes), False, dtype=bool)
    for start_time, end_time, note in raw_labels:
        if only_note_names:
            note = note % 12
        else:
            note = note - 24  # shift by C1
            if note < 0 or note >= n_notes:
                continue
        start_frame = round(start_time / hop_length)
        end_frame = round(end_time / hop_length)
        labels[start_frame:end_frame, note] = True
    return labels


@dataclass(kw_only=True)
class DataProcessor:
    """Store spectogram of a song and corresponding one-hot labels into
    one (time, freq) and one (time, n_notes) zarr array.

    Note that with small changes to [_transform] we can save time-series directly.
    """

    split: Split
    bins_per_note: int = 4
    only_note_names: bool = False
    hop_length: int = 512
    num_workers: int = 10

    @property
    def freq_dim(self):
        return N_OCTAVES * 12 * self.bins_per_note

    @property
    def n_notes(self):
        return 12 * N_OCTAVES if not self.only_note_names else 12

    def __post_init__(self):
        zarr_path = f"{PROCESSED_DATASET_PATH}\\{self.split.value}.zarr"
        shutil.rmtree(zarr_path, onexc=_onexc_remove)
        self.zarr_group = zarr.create_group(zarr_path, overwrite=True)

        size = 1_000_000
        self.zarr_group.create_array(
            "data", shape=(size, self.freq_dim), chunks=(TIME_CHUNK, self.freq_dim), dtype=np.float32
        )
        self.zarr_group.create_array(
            "labels", shape=(size, self.n_notes), chunks=(TIME_CHUNK, self.n_notes), dtype=bool
        )
        self.zarr_group.attrs["store_size"] = size

        self.zarr_group.attrs["num_entries"] = 0
        self.zarr_group.attrs["cu_seqlens"] = []
        self.zarr_group.attrs["hop_length"] = self.hop_length
        self.zarr_group.attrs["bins_per_note"] = self.bins_per_note
        self.zarr_group.attrs["only_note_names"] = self.only_note_names

    def _resize(self, size: int):
        self.zarr_group['data'].resize((size, self.freq_dim))
        self.zarr_group['labels'].resize((size, self.n_notes))
        self.zarr_group.attrs["store_size"] = size

    def _transform(self, song_name) -> tuple[np.ndarray, np.ndarray]:
        raw_song = load_song(song_name, self.split)
        spect = normalized_cqt(  # (time, freq)
            raw_song,
            hop_length=self.hop_length,
            bins_per_note=self.bins_per_note,
        ).astype(np.float32)

        raw_labels = load_labels(song_name, self.split)
        labels = get_one_hot_labels(  # (time, notes)
            raw_labels,
            time_length=spect.shape[0],
            hop_length=self.hop_length,
            only_note_names=self.only_note_names,
        ).astype(np.bool)

        return spect, labels

    def _log(self, idx: int, data: np.ndarray, labels: np.ndarray):
        assert (length := data.shape[0]) == labels.shape[0]
        self.zarr_group['data'][idx: idx + length] = data
        self.zarr_group['labels'][idx: idx + length] = labels
        self.zarr_group.attrs["num_entries"] = idx + length
        self.zarr_group.attrs["cu_seqlens"].append(idx)

    def run(self):
        exc = futures.ProcessPoolExecutor(max_workers=self.num_workers)
        all_futures = []
        for f in os.listdir(os.fsencode(f"{DATASET_PATH}\\{self.split.value}_labels")):
            file = os.fsdecode(f)
            song_name, ext = file.split('.')
            if ext != "csv":
                continue
            all_futures.append(exc.submit(self._transform, song_name))

        total = 0
        for fs in tqdm(all_futures):
            data, labels = fs.result()
            new_total = total + data.shape[0]
            if new_total > self.zarr_group.attrs["store_size"]:
                self._resize(2 * new_total)
            self._log(total, data, labels)
            total = new_total
        if total < self.zarr_group.attrs["store_size"]:
            self._resize(total)
        exc.shutdown()


if __name__ == "__main__":
    processor = DataProcessor(
        split=Split.TEST,
        bins_per_note=4,
        only_note_names=False,
        hop_length=512,
        num_workers=10,
    )
    processor.run()
