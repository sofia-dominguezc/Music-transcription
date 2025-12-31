import os
from math import ceil
from concurrent import futures
from tqdm import tqdm
from typing import Literal

import numpy as np
import pandas as pd
import librosa

dataset_path = os.path.join("data", "musicnet")
processed_path = os.path.join("data", "musicnet_processed")


def load_song(song: str, split: Literal["train", "test"]) -> np.ndarray:
    """Load song"""
    song_path = f"{dataset_path}\\{split}_data\\{song}.wav"
    song_vals, song_sr = librosa.load(song_path)
    assert song_sr == 22050, "Invalid sr"
    return song_vals


def batched_q_transform(
    song_vals: np.ndarray,
    batch_seconds: int | float,
    bins_per_note: int,
    n_octaves: int,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """
    Calculate the constant q-transform of the song and return its batched log.
    The constant q-transform is like a FT but logarithmic in frequency.
    """
    # spectogram
    raw_spect = librosa.cqt(
        song_vals,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_octaves*bins_per_note*12,
        bins_per_octave=bins_per_note*12,
        filter_scale=0.25,
        fmin=librosa.note_to_hz('C1'),
        scale=True,
    )
    spect = np.abs(raw_spect.T)**0.3  # (time, freq)
    spect = (spect - spect.mean()) / spect.std()
    # split into batches
    n_full_time, n_freq = spect.shape
    n_time = int(batch_seconds * sr / hop_length)
    n_batch = ceil(n_full_time / n_time)
    full_spect = np.zeros((n_batch * n_time, n_freq))
    full_spect[:n_full_time] = spect
    return full_spect.reshape((n_batch, n_time, n_freq))


def load_labels(song: str, split: Literal["train", "test"], all_notes: bool) -> pd.DataFrame:
    """Load labels of a song. Time is in sample space"""
    song_path = f"{dataset_path}\\{split}_labels\\{song}.csv"

    with open(song_path, "r") as f:
        df = pd.read_csv(f)
    df = df.rename(columns={"start_time": "start", "end_time": "end"})

    if not all_notes:
        df["note"] = df["note"] % 12
    return df[["start", "end", "note"]].astype(int)


def one_hot_labels(
    raw_labels: pd.DataFrame,
    n_batch: int,
    n_time: int,
    n_octaves: int,
    hop_length: int,
    all_notes: bool,
) -> np.ndarray:
    """
    Returns a boolean array determining if a given window of the stft contains
    a note or not. Index t is the window centered at sample time t * hop_length

    Args:
        raw_labels: dataframe with (start, end, note) tuples
        n_batch: number of batches
        n_time: length of each batch
    """
    n_notes = 12 * n_octaves if all_notes else 12
    labels = np.full((n_batch * n_time, n_notes), False, dtype=bool)
    for _, row in raw_labels.iterrows():
        start, end, note = row
        note = note - 24 if all_notes else note  # NOTE: depends on n_octaves
        if note < 0 or note >= n_notes:
            continue
        lower = round(start / hop_length)
        upper = round(end / hop_length)
        labels[lower:upper, note] = True
    return labels.reshape(n_batch, n_time, n_notes)


def process_song(
    song: str,
    split: Literal["train", "test"],
    batch_seconds: int | float,
    bins_per_note: int,
    n_octaves,
    sr: int,
    hop_length: int,
    all_notes: bool,
    batch_size: int = 60,
):
    """
    Loads song, calculates the batched spectogram, puts the labels in
    one hot format, and saves everything to .npy files.

    Args:
        song: name of file
        split: name of data split
        batch_seconds: number of seconds on each batch (default 1)
        bins_per_note: number of frequency samples between notes
        sr: sapmling rate
        hop_length: distance between applications of q-transform
        all_notes: if fase, considers notese modulo 12
        n_batches: number of batches (of size batch_seconds) to save on each file
    """
    song_vals = load_song(song, split)
    spect = batched_q_transform(  # (batch, time, freq)
        song_vals, batch_seconds, bins_per_note, n_octaves, sr, hop_length
    ).astype(np.float16)

    raw_labels = load_labels(song, split, all_notes)
    labels = one_hot_labels(  # (batch, time, notes)
        raw_labels, spect.shape[0], spect.shape[1], n_octaves, hop_length, all_notes,
    ).astype(bool)

    for idx in range(0, spect.shape[0], batch_size):
        i = idx // batch_size
        np.save(f"{processed_path}\\{split}_data\\{song}_{i}.npy", spect[idx: idx + batch_size])
        np.save(f"{processed_path}\\{split}_labels\\{song}_{i}.npy", labels[idx: idx + batch_size])


def process_data(split: Literal["train", "test"], num_workers: int = 8, **args):
    """
    Load and process all songs in parallel.
    args: arguments for process_song
    """
    for info in ["data", "labels"]:
        try:
            os.makedirs(f"{processed_path}", exist_ok=True)
            os.mkdir(f"{processed_path}\\{split}_{info}")
        except FileExistsError:
            print(
                f"Note: {split}_{info} directory already exists. "
                "Old files may remain there."
            )
            pass

    executor = futures.ProcessPoolExecutor(max_workers=num_workers)
    process_futures = []
    print(f"Loading and processing {split}ing data and labels...")
    for f in os.listdir(os.fsencode(f"{dataset_path}\\{split}_data")):
        file = os.fsdecode(f)
        song, extension = file.split('.')
        assert extension == "wav", f"Invalid file encountered."
        process_futures.append(
            executor.submit(process_song, song, split, **args)
        )
    pbar = tqdm(total=len(process_futures))
    for f in futures.as_completed(process_futures):
        f.result()
        pbar.update(1)
    pbar.clear()
    executor.shutdown()


if __name__ == "__main__":
    process_data(
        split="train",
        num_workers=12,
        batch_seconds=5.02,
        bins_per_note=8,
        n_octaves=8,
        sr=22050,
        hop_length=256,
        all_notes=True,
        batch_size=1,
    )
