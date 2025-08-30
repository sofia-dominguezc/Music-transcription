import os
from pathlib import Path
from math import ceil
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import DataLoader, Dataset
from concurrent import futures
from tqdm import tqdm

dataset_path = Path.cwd() / "datasets" / "musicnet"

# NOTE: sampling rate assumptions are hard coded
# Songs are asssumed to have 22050 and labels are in 44100


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


def load_song(song: str, split: str) -> np.ndarray:
    """Load song"""
    assert split in ["train", "test"], "Invalid split"
    song_path = dataset_path / f"{split}_data" / f"{song}.wav"
    song_vals, song_sr = librosa.load(song_path)
    return song_vals


def batched_spectogram(
    song_vals: np.ndarray, batch_seconds: float, sr: int, n_fft: int,
    hop_length: int, low_freq_spect: int, high_freq_spect: int,
) -> np.ndarray:
    """
    Convert song into spectogram in batches.
    batch_seconds: length of time of each batch.
    hop_length: number of overlap in samples between consecutive windows for stft
    out: has shape (batch, time, freq)
    """
    spect = np.abs(
        librosa.stft(song_vals, n_fft=n_fft, hop_length=hop_length),  # (n_freq, time)
    )[low_freq_spect:high_freq_spect]  # from ~30Hz to ~4000Hz
    n_freq, n_time = spect.shape
    batch_len = int(batch_seconds * sr / hop_length)  # batch lenght in time
    n_batch = ceil(n_time / batch_len)
    # split into batches
    flat_spect = np.zeros((n_freq, n_batch * batch_len))
    flat_spect[:, :n_time] = spect
    batched_spect = flat_spect.reshape((n_freq, n_batch, batch_len))
    batched_spect = np.transpose(batched_spect, (1, 2, 0))  # (batch, time, freq)
    return batched_spect


def interpolate_repeat_spectogram(
    spect: np.ndarray, n_vals: int, time_repeat: int, low_freq_spect: int,
) -> np.ndarray:
    """
    Takes the log of the frequency values in the array and interpolates
    the values of the spectogram. Also adds time repetition to each batch
    spect: (..., n_time, n_freq)
    time_repeat: number of repeated elements added at each end of each batch
    out:   (..., n_time, n_vals)
    """
    n_time, n_freq = spect.shape[-2:]
    # Take log of freq and interpolate
    freqs = np.log2(np.arange(n_freq) + low_freq_spect)  # indices correspond to scaled frequencies
    interp_freqs = np.linspace(freqs[0], freqs[-1], n_vals)
    put_idx = np.searchsorted(freqs, interp_freqs)  # where to put interp_freqs
    low_idx = np.clip(put_idx - 1, a_min=0, a_max=None)
    high_idx = np.clip(put_idx, a_min=None, a_max=n_freq - 1)
    alpha = (interp_freqs - freqs[low_idx]) / (freqs[high_idx] - freqs[low_idx] + 1e-12)
    spect_interp = (1 - alpha) * spect[..., low_idx] + alpha * spect[..., high_idx]
    # add temporal repetition
    if not time_repeat:
        return spect_interp
    spect_repeat = np.zeros((*spect.shape[:-2], n_time + 2 * time_repeat, n_vals))
    spect_repeat[:, time_repeat:-time_repeat] = spect_interp
    spect_repeat[:-1, -time_repeat:] = spect_interp[1:, :time_repeat]
    spect_repeat[1:, :time_repeat] = spect_interp[:-1, -time_repeat:]
    return spect_repeat


def batched_q_transform(
    song_vals: np.ndarray, batch_seconds: float, bins_per_octave: int,
    sr: int, hop_length: int,
) -> np.ndarray:
    """
    Calculate the constant q-transform of the song.
    The constant q-transform is like a FT but logarithmic in frequency.
    """
    spect = np.abs(librosa.cqt(
        song_vals, sr=sr, hop_length=hop_length,
        n_bins=8*bins_per_octave, bins_per_octave=bins_per_octave,
    ))
    db_spect = librosa.amplitude_to_db(spect)  # (freq, time)
    db_spect = (db_spect - db_spect.mean()) / db_spect.std()
    # variables
    n_freq, n_time = db_spect.shape
    new_n_time = int(batch_seconds * sr / hop_length)
    n_batch = ceil(n_time / new_n_time)
    # split into batches
    flat_spect = np.zeros((n_freq, n_batch * new_n_time))
    flat_spect[:, :n_time] = spect
    batched_spect = flat_spect.reshape((n_freq, n_batch, new_n_time))
    batched_spect = np.transpose(batched_spect, (1, 2, 0))  # (batch, time, freq)
    return batched_spect


def load_labels(song: str, split: str, all_notes: bool) -> pd.DataFrame:
    """Load labels of a song. Time is in sample space"""
    assert split in ["train", "test"], "Invalid split"
    song_path = dataset_path / f"{split}_labels" / f"{song}.csv"
    with open(song_path, "r") as f:
        df = pd.read_csv(f)
    df = df.rename(columns={"start_time": "start", "end_time": "end"})
    df[["start", "end"]] /= 2  # adjust to real sr
    if not all_notes:
        df["note"] = df["note"] % 12
    return df[["start", "end", "note"]].astype(int)


def one_hot_labels(
    raw_labels: pd.DataFrame, num_samples: int, batch_seconds: float,
    sr: int, hop_length: int, all_notes: bool,
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
    song: str, split: str, batch_seconds: float, bins_per_octave: int,
    sr: int, hop_length: int, all_notes: bool,
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
        song_vals, batch_seconds, bins_per_octave, sr, hop_length
    ).astype(np.float32)
    raw_labels = load_labels(song, split, all_notes)
    labels = one_hot_labels(
        raw_labels, song_vals.shape[0], batch_seconds, sr, hop_length, all_notes,
    ).astype(bool)
    assert spect.shape[:-1] == labels.shape[:-1], (
        f"data and label shapes are off: {spect.shape}, {labels.shape}"
    )
    np.save(dataset_path / f"{split}_data_npy" / f"{song}.npy", spect)
    np.save(dataset_path / f"{split}_labels_npy" / f"{song}.npy", labels)


def process_data(split: str, **args) -> None:
    """
    Load and process all songs in parallel.
    """
    assert split in ["train", "test"], "Invalid split"
    for info in ["data", "labels"]:
        try:
            os.mkdir(dataset_path / f"{split}_{info}_npy")
        except FileExistsError:
            pass

    executor = futures.ProcessPoolExecutor(max_workers=8)
    process_futures = []
    print(f"Loading and processing {split}ing data and labels...")
    for f in os.listdir(os.fsencode(dataset_path / f"{split}_data")):
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


def create_dataloader(
    split: str, batch_size: int, num_workers: int = 0
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """
    Make dataloader from a list of song files.
    If split='train', it will suffle batches even among different songs
    """
    assert split in ["train", "test"], "Invalid split"
    torch.cuda.empty_cache()
    # load data
    x_array, y_array = [], []
    for f in os.listdir(os.fsencode(dataset_path / f"{split}_data_npy")):
        file = os.fsdecode(f)
        assert file.endswith(".npy"), f"Invalid file encountered."
        song_vals = np.load(dataset_path / f"{split}_data_npy" / file)
        labels = np.load(dataset_path / f"{split}_labels_npy" / file)
        x_array.append(song_vals)
        y_array.append(labels)
    dataset = GroupedTensorDataset(x_array, y_array)
    # return dataloader
    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    } if num_workers > 0 else {}
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split=="train"), **workers_args,
    )
    return dataloader


if __name__ == "__main__":
    pass
    # Test loading songs
    # song = load_song('1727', 'train', sr=22050)
    # print(song.shape, song.shape[0] / 22050)
    # qt = q_transform(song, sr=22050, hop_length=512, bins_per_octave=36)
    # print(qt.shape, qt.size, qt.dtype)

    # spect = batched_spectogram(
    #     song, batch_seconds=1, n_fft=2048, hop_length=512, low_freq_spect=3, 
    # )
    # print(spect.shape)
    # spect = interpolate_repeat_spectogram(spect, n_vals=250, time_repeat=3)
    # print(spect.shape)

    # # Test loading labels
    # df = load_labels("1727", "train")
    # print(df.drop_duplicates(subset='note').sort_values(by='note'))
    # labels = one_hot_labels(df, n_fft=2048, hop_length=512, spect.shape[:2])
    # print(labels[0])
    # print(labels.shape)

    # import matplotlib.pyplot as plt
    # sr = 22050
    # y = song
    # print(y.shape)
    # C = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=36))
    # db_C = librosa.amplitude_to_db(C, ref=np.max)
    # db_C = (db_C - db_C.mean()) / db_C.std()
    # C = np.log(C)
    # C = (C - C.mean()) / C.std()
    # print(C.shape)
    # fig, ax = plt.subplots()
    # import random
    # samples = random.sample(db_C.reshape(-1).tolist(), k=10000)
    # samples2 = random.sample(C.reshape(-1).tolist(), k=10000)
    # ax.hist(samples2, bins=50, label="Amplitude values", density=True, alpha=0.5)
    # ax.hist(samples, bins=50, label="dB values", density=True, alpha=0.5)
    # x = np.linspace(-3, 3, 100)
    # y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    # ax.plot(x, y, label="Normal distribution")
    # ax.set_title('Distribution of constant Q-transform values')
    # ax.legend(loc='upper left')
    # plt.show()
