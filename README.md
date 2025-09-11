The goal is to train an AI that can recognize which notes are being played, but the inputs to the model will be the spectogram instead of the raw sound.

paper: https://projects.iq.harvard.edu/files/kakade/files/1611-09827-2017.pdf

## Usage

`python src`

## Arguments

- `--action`, `-a`. Options: `train`, `train-val`, `test`, `process-train`, `process-test`. **Required.**
- `--lr`: Learning rate for training.
- `--num-epochs`: Number of training epochs.
- `--batch-size`: Number of samples per training batch.
- `--num-workers`: Number of worker processes for data loading.
- `--gamma`: Learning rate scheduler multiplier.
- `--milestones`: Epochs at which to adjust the learning rate.

- `--thresholds`: Probability thresholds for test accuracy calculation. Must be in [0, 1].
- `--allowed-errors`: Allowed errors per frame for test performance.

- `--c`: Multiplier for the number of convolutional channels.
- `--test-dev`: If set, runs test loop using dev model.

- `--batch-seconds`: Duration (in seconds) of each batch.
- `--bins-per-octave`: Number of frequency bins per octave.
- `--only-note-names`: If set, processes notes modulo 12.
- `--sr`: Audio sample rate.
- `--hop-length`: Step size between FFT windows.

## TODO

- Make the system that processes a song and outputs the notes in real time.
- Make `__main__` so it supports pre-processing, training, and evaluation
- Consider adding a long range attention. Code for criss-cross transformers and masking are in the repo's history
- Consider training the model from raw labels instead of q-transform
- Add support for `all_notes=False`

## Experiments:

### Older

Multiple convolution layers or with very large kernels doesn't work very well

I also noticed that repeating the linear layer per channel doesn't loose much
  performance, if after that we use a max over channels and residual connection

I also noticed that the maxpool is not too important in convolutional layers

The temporal part of the convolution doesn't seem to help as much

### 4 important findings

- linear layers or long-range convolutions w.r.t. time are useless, but small ones help a bit

- convolutions w.r.t. frequency must include at most 1.5 notes above/below

- sinusoidal positional encoding performs the same as a 1d convolution

- self-attention over short time range (1s) improves accuracy from 25% (CNN + MLP) to 30% (current architecture)

### Short range transformer

Using n_layers=4, n_heads=4, head_dim=32, c=3, embed_dim=192 (1.8M params):
  best lr: 5e-4, test_acc = 30% (43% in notes)
  validation stopped improving at loss=3.9, acc=30% while train_loss made it to loss=3.7
