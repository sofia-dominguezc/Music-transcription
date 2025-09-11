import os
from math import ceil
from concurrent import futures
from tqdm import tqdm
from typing import Literal

import numpy as np
import pandas as pd
import librosa

dataset_path = "data\\musicnet"
save_path = "data\\musicnet_processed"
n_octaves = 8

# NOTE: songs are asssumed to have sr=22050 and labels are in 44100


def load_song(song: str, split: Literal["train", "test"]) -> np.ndarray:
    """Load song"""
    song_path = f"{dataset_path}\\{split}_data\\{song}.wav"
    song_vals, song_sr = librosa.load(song_path)
    assert song_sr == 22050, "Invalid sr"
    return song_vals


def batched_q_transform(
    song_vals: np.ndarray,
    batch_seconds: int,
    bins_per_note: int,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """
    Calculate the constant q-transform of the song and return its batched log.
    The constant q-transform is like a FT but logarithmic in frequency.
    """
    # spectogram
    raw_spect = np.abs(librosa.cqt(
        song_vals,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_octaves*bins_per_note*12,
        bins_per_octave=bins_per_note*12,
        filter_scale=0.5,
    )).T
    spect = librosa.amplitude_to_db(raw_spect)  # (full_time, freq)
    spect: np.ndarray = (spect - spect.mean()) / spect.std()
    n_full_time, n_freq = spect.shape
    n_time = batch_seconds * (sr // hop_length)
    n_batch = ceil(n_full_time / n_time)
    # split into batches
    flat_spect = np.zeros((n_batch * n_time, n_freq))
    flat_spect[:n_full_time] = spect
    batched_spect = flat_spect.reshape((n_batch, n_time, n_freq))
    return batched_spect


def load_labels(song: str, split: Literal["train", "test"], all_notes: bool) -> pd.DataFrame:
    """Load labels of a song. Time is in sample space"""
    song_path = f"{dataset_path}\\{split}_labels\\{song}.csv"

    with open(song_path, "r") as f:
        df = pd.read_csv(f)
    df = df.rename(columns={"start_time": "start", "end_time": "end"})

    df[["start", "end"]] /= 2  # adjust to real sr
    if not all_notes:
        df["note"] = df["note"] % 12
    return df[["start", "end", "note"]].astype(int)


def one_hot_labels(
    raw_labels: pd.DataFrame,
    n_batch: int,
    n_time: int,
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
    batch_seconds: int,
    bins_per_note: int,
    sr: int,
    hop_length: int,
    all_notes: bool,
) -> None:
    """
    Loads song, calculates the batched spectogram, puts the labels in
    one hot format, and saves everything to .npy files.
    hop_length: number of samples between consecutive windows for stft
    n_freq: size of frequency dimension. Interpolates between available ones
    time_repeat: number of time indices to repeat per batch on each direction
    only_note_name: if true, then considers notes modulo 12
    """
    song_vals = load_song(song, split)
    spect = batched_q_transform(  # (batch, time, freq)
        song_vals, batch_seconds, bins_per_note, sr, hop_length
    ).astype(np.float16)

    raw_labels = load_labels(song, split, all_notes)
    labels = one_hot_labels(  # (batch, time, notes)
        raw_labels, spect.shape[0], spect.shape[1], hop_length, all_notes,
    ).astype(bool)

    for idx, (batch_spect, batch_label) in enumerate(zip(spect, labels)):
        np.save(f"{save_path}\\{split}_data\\{song}_{idx}.npy", batch_spect)
        np.save(f"{save_path}\\{split}_labels\\{song}_{idx}.npy", batch_label)


def process_data(split: Literal["train", "test"], num_workers: int = 8, **args) -> None:
    """
    Load and process all songs in parallel.
    args: arguments for process_song
    """
    for info in ["data", "labels"]:
        try:
            os.mkdir(f"{save_path}\\{split}_{info}")
        except FileExistsError:
            pass

    executor = futures.ProcessPoolExecutor(max_workers=num_workers or 8)
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
        num_workers=8,
        batch_seconds=60,
        bins_per_note=4,
        sr=22050,
        hop_length=512,
        all_notes=True,
    )
