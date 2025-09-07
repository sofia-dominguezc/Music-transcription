import os
from math import ceil
import pandas as pd
import numpy as np
import librosa
from concurrent import futures
from tqdm import tqdm

dataset_path = "data\\musicnet"
save_path = "data\\musicnet_processed"

# NOTE: sampling rate assumptions are hard coded
# Songs are asssumed to have 22050 and labels are in 44100


def load_song(song: str, split: str) -> np.ndarray:
    """Load song"""
    assert split in ["train", "test"], "Invalid split"
    song_path = f"{dataset_path}\\{split}_data\\{song}.wav"
    song_vals, song_sr = librosa.load(song_path)
    assert song_sr == 22050
    return song_vals


def batched_q_transform(
    song_vals: np.ndarray,
    batch_seconds: float,
    bins_per_note: int,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """
    Calculate the constant q-transform of the song and return its batched log.
    The constant q-transform is like a FT but logarithmic in frequency.
    """
    # spectogram
    spect = np.abs(librosa.cqt(
        song_vals, sr=sr, hop_length=hop_length,
        n_bins=8*bins_per_note*12, bins_per_octave=bins_per_note*12,
    ))
    db_spect = librosa.amplitude_to_db(spect)  # (freq, time)
    db_spect = (db_spect - db_spect.mean()) / db_spect.std()
    # variables
    n_freq, n_full_time = db_spect.shape
    n_time = int(batch_seconds * sr / hop_length)
    n_batch = ceil(n_full_time / n_time)
    # split into batches
    flat_spect = np.zeros((n_freq, n_batch * n_time))
    flat_spect[:, :n_full_time] = db_spect
    batched_spect = flat_spect.reshape((n_freq, n_batch, n_time))
    batched_spect = np.transpose(batched_spect, (1, 2, 0))  # (t_batch, time, freq)
    return batched_spect


def load_labels(song: str, split: str, all_notes: bool) -> pd.DataFrame:
    """Load labels of a song. Time is in sample space"""
    assert split in ["train", "test"], "Invalid split"
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
    num_samples: int,
    batch_seconds: float,
    sr: int,
    hop_length: int,
    all_notes: bool,
) -> np.ndarray:
    """
    Returns a boolean array determining if a given window of the stft contains
    a note or not. Index t is the window centered at sample time t * hop_length
    num_samples: length of the signal of the corresponding label
    out: shape (*time, n_notes)
    """
    n_time = ceil(num_samples / hop_length)
    new_n_time = int(batch_seconds * sr / hop_length)
    n_batch = ceil(n_time / new_n_time)
    n_notes = 12 * 8 if all_notes else 12
    labels = np.full((n_batch * new_n_time, n_notes), False, dtype=bool)
    for _, row in raw_labels.iterrows():
        start, end, note = row
        note = note - 12 if all_notes else note
        if note < 0 or note >= n_notes:  # using C1 to B8
            continue
        lower = round(start / hop_length)
        upper = round(end / hop_length)
        labels[lower:upper, note] = True
    return labels.reshape(n_batch, new_n_time, n_notes)


def process_song(
    song: str,
    split: str,
    batch_seconds: float,
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
    spect = batched_q_transform(
        song_vals, batch_seconds, bins_per_note, sr, hop_length
    ).astype(np.float32)

    raw_labels = load_labels(song, split, all_notes)
    labels = one_hot_labels(
        raw_labels, song_vals.shape[0], batch_seconds, sr, hop_length, all_notes,
    ).astype(bool)

    np.save(f"{save_path}\\{split}_data\\{song}.npy", spect)
    np.save(f"{save_path}\\{split}_labels\\{song}.npy", labels)


def process_data(split: str, **args) -> None:
    """
    Load and process all songs in parallel.
    args: arguments for process_song
    """
    assert split in ["train", "test"], "Invalid split"
    for info in ["data", "labels"]:
        try:
            os.mkdir(f"{save_path}\\{split}_{info}")
        except FileExistsError:
            pass

    executor = futures.ProcessPoolExecutor(max_workers=8)
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
        split="test",
        batch_seconds=4,
        bins_per_note=4,
        sr=22050,
        hop_length=512,
        all_notes=True,
    )
