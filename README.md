The goal is to train an AI that can recognize which notes are being played, but the inputs to the model will be the spectogram instead of the raw sound.

paper: https://projects.iq.harvard.edu/files/kakade/files/1611-09827-2017.pdf

### Usage

```
python src
```

### Arguments

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

### Pre-processing

DONE:
- Calculated the spectogram for the training data.
- Interpolated frequency values (in log space) to make the model easier to train
- Made the batches overlap with each other a bit, around 7% on each side
- Made the label data into a one-hot (12, time) array

PLAN:
- Move everything into the stft time space. A note is predicted at a "time" t if the window corresponding that "time" t intersects the note interval.
- A note is predicted at a real time t if the window closest to it predicts the note. This introduces an error of about 0.1s if window=2048 and 0.05s if window=4096, which is acceptable for my use case.

### Training

DONE:
- I did the CNN architecture.
- I did the train/test logic and lightling module

PLAN:
- I need something with temporal information to protect against short notes but I need temporal positional encoding. This means a CNN is best
- I shouldn't use a time window too large because most details will be irrelevant. Maybe a temporal kernel size of 3 or 5 is best. This information is captured in the MLP layers better.

TODO:
- Fine tune hyperparameters
- Add loggers or trackers like the one I used in computer vision

### Output

TODO:
- Make the system that processes a song and outputs the notes in real time.

## Notes about transformer architecture

lr=0.001 is optimal for n_layers=3, n_heads=4, head_dim=32, c=3, embed_dim=output_dim//3 (main model)
- However, this architecture (2.3M params) overfits the data:
    when train_loss > 6, then val_loss ~ train_loss
    when train_loss < 6 (it got to 3.4 in 50 epochs), val_loss increases (up to 9 in this case)
    validation accuracy is super bad always
- This probably happens because 1/3 of the parameters are the positional encoding. This is dumb
    Either use non-learnable positional encodings, or reduce number of params a lot
    Another option is to make the model smaller
