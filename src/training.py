from typing import cast, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import lightning as pl

from preprocess_data import save_path

model_weights = "parameters\\model_weights.pth"
dev_model_weights = "parameters\\dev_model_weights.pth"


# class MusicModel(nn.Module):
#     """
#     Model that takes a spectogram of shape (frames, freq) and returns
#     an array (frames, n_notes) where at each time is the probability distribuion
#     of possible notes.
#     """
#     def __init__(
#         self, c: int, n_freq: int, all_notes: bool,
#     ) -> None:
#         """
#         c: multiplier of features
#         """
#         super().__init__()
#         n_notes = 12 * 8 if all_notes else 12
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, c, (5, 9), padding=(2, 4)),
#             nn.GELU(),
#             nn.MaxPool2d((1, 4)),
#         )
#         self.linear = nn.Linear(c * (n_freq // 4), n_notes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (..., frames, freq)
#         logits: (..., frames, n_notes)
#         """
#         x = x.unsqueeze(-3)  # (..., 1, T, F)
#         x = self.conv(x)  # (..., C, T, F)
#         x = x.transpose(-2, -3).flatten(-2, -1)  # (..., T, C*F)
#         x = self.linear(x)  # (..., T, n_notes)
#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, n_freq: int, n_time: int) -> None:
#         super().__init__()
#         pe = torch.zeros(n_time, n_freq)
#         position = torch.arange(0, n_time).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, n_freq, 2) * (-log(10000) / n_freq)).float()
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (..., C, T, F)
#         Add positional encoding across time T for each frequency bin
#         """
#         return x + self.pe

# def positional_encoding(n_freq: int, n_time: int):
#     pe = torch.zeros(n_time, n_freq, device='cuda')
#     position = torch.arange(0, n_time).unsqueeze(1).float()
#     div_term = torch.exp(torch.arange(0, n_freq, 2) * (-log(10000) / n_freq)).float()
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#     return pe


# class SelfAttention(nn.Module):
#     def __init__(self, n_freq: int, qk_dim: int, n_heads: int) -> None:
#         super().__init__()
#         assert qk_dim % n_heads == 0, "Head dims must divide evenly"

#         self.qk_dim = qk_dim
#         self.v_dim = qk_dim
#         self.n_heads = n_heads
#         self.qk_head_dim = qk_dim // n_heads
#         self.v_head_dim = qk_dim // n_heads

#         self.q_proj = nn.Linear(n_freq, qk_dim)
#         self.k_proj = nn.Linear(n_freq, qk_dim)
#         self.v_proj = nn.Linear(n_freq, qk_dim)
#         self.out_proj = nn.Linear(qk_dim, n_freq)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: shape (..., C, T, F)
#         Attention is applied over T
#         """
#         orig_shape = x.shape  # (..., C, T, F)
#         x = x.flatten(0, -3)  # [B*C, T, F]
#         *BC_T, _ = x.shape

#         Q = self.q_proj(x)  # [B*C, T, qk_dim]
#         K = self.k_proj(x)  # [B*C, T, qk_dim]
#         V = self.v_proj(x)  # [B*C, T, v_dim]

#         # Reshape for multihead attention
#         Q = Q.view(*BC_T, self.n_heads, self.qk_head_dim).transpose(-2, -3)  # [B*C, n_heads, T, qk_head_dim]
#         K = K.view(*BC_T, self.n_heads, self.qk_head_dim).transpose(-2, -3)  # [B*C, n_heads, T, qk_head_dim]
#         V = V.view(*BC_T, self.n_heads, self.v_head_dim).transpose(-2, -3)  # [B*C, n_heads, T, v_head_dim]

#         # Scaled dot-product attention
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.qk_head_dim ** 0.5)  # [B*C, n_heads, T, T]
#         attn_weights = nn.functional.softmax(scores, dim=-1)  # [B*C, n_heads, T, T]
#         attn_output = torch.matmul(attn_weights, V)  # [B*C, n_heads, T, v_head_dim]

#         # Concatenate heads
#         attn_output = attn_output.transpose(-2, -3).contiguous().view(*BC_T, self.v_dim)  # [B*C, T, v_dim]

#         # Final output projection back to n_freq
#         out = self.out_proj(attn_output)  # [B*C, T, n_freq]
#         out = out.view(*orig_shape)  # (..., C, T, F)
#         return out

class Transpose(nn.Module):
    """Transposes the desired pairs of dimensions in order"""
    def __init__(self, *dims: tuple[int, int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dim0, dim1 in self.dims:
            x = x.transpose(dim0, dim1)
        return x


class BatchDims(nn.Module):
    """Batch all dimensions except the last 2 and then run the nn.Modules"""
    def __init__(self, *args: nn.Module) -> None:
        super().__init__()
        self.nn_modules = nn.Sequential(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *B, C, L = x.shape
        x = self.nn_modules(x.reshape(-1, C, L))
        return x.view(*B, C, L)


class MusicModel(nn.Module):
    def __init__(self, c: int, all_notes: bool) -> None:
        super().__init__()
        n_notes = 12 * 8 if all_notes else 12
        tm, fq = 3, 3  # out-reach of each dimension (over 2)

        self.conv = nn.Sequential(
            nn.Conv2d(1, c//2, kernel_size=(2*tm+1, 2*fq+1), padding=(tm, fq), stride=(1, 2)),
            nn.GELU(),
            nn.Conv2d(c//2, c, kernel_size=(2*tm+1, 2*fq+1), padding=(tm, fq), stride=(1, 2)),
            nn.GELU(),
        )

        F = 8*12
        self.linear = nn.Sequential(  # (..., C, T, F)
            Transpose((-3, -2)),  # (..., T, C, F)
            nn.Flatten(-2, -1),  # (..., T, c*F)
            nn.Linear(c*F, c*F),
            nn.GELU(),
            nn.Linear(c*F, F),  # (..., T, F)
        )
        # self.conv2 = nn.Conv2d(c + 2, 1, kernel_size=1)
        # self.linear = nn.Conv3d(c, 1, 1)

        # pe = positional_encoding(n_freq=n_freq//4, n_time=n_time)
        # self.register_buffer('pe', pe)
        # self.attn = SelfAttention(n_freq=n_freq//4, qk_dim=12*n_heads, n_heads=n_heads)

        # self.linear1 = nn.Linear(n_freq // 4, n_notes)
        # self.linear2 = nn.Linear(n_freq // 4, n_notes)
        # self.linear3 = nn.Linear(n_freq // 2, n_notes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., T, F)
        """
        y = x.unsqueeze(-3)  # (*B, 1, T, F)
        y = y[..., ::4] + self.conv(y)
        y = torch.max(y, dim=-3)[0] + self.linear(y)
        return y

        # y = x.unsqueeze(-3)  # (..., 1, T, F)
        # z = y[..., ::4] + self.conv(y)  # (..., C, T, F')
        # w = y[..., ::4] + self.freq_conv(y)  # (..., C, T, F')
        # z = y[..., ::4] + torch.max(self.linear1(z), dim=-3, keepdim=True).values  # (..., T, notes)
        # w = y[..., ::4] + torch.max(self.linear2(w), dim=-3, keepdim=True).values  # (..., T, notes)
        # out = torch.cat((z, w), dim=-3).transpose(-3, -2).flatten(-2, -1)
        # return self.linear3(out)

        # *B, T, F = x.shape
        # x = x.reshape(-1, F).unsqueeze(-2)  # (B*T, 1, F)
        # x = self.freq_conv(x)
        # x = torch.max(x, dim=-2).values  # (B*T, F')
        # return x.view(*B, T, -1)

        # B = x.shape[:-2]
        # x = x + self.pe.expand(*x.shape[:-2], -1, -1)  # (..., C', T, F')
        # x = x.flatten(0, -3)  # (... * C', T, F')
        # x = x + self.attn(x)  # Self-attn over time
        # x = x.unflatten(0, B) # (..., C', T, F')


class LitMusicModel(pl.LightningModule):
    best_val_acc: float

    def __init__(
        self, model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        allowed_errors: list[int] = [0],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.allowed_errors = allowed_errors
        self.best_val_acc = 0

    def configure_optimizers(self):  # type: ignore
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)  # (..., T, n_notes)
        loss = self.loss_fn(logits, y)

        self.log("loss_step", 100 * loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", 100 * loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get('loss_epoch')
        if loss is not None:
            print(f"\nEpoch {self.current_epoch} - train_loss: {loss:.4f}")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)  # (..., T, n_notes)
        loss = self.loss_fn(logits, y)
        correct = torch.sum((logits >= 0) != y.bool(), dim=-1) == 0
        acc = 100 * torch.sum(correct) / correct.nelement()
        self.log("val_loss", 100 * loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.trainer.callback_metrics['val_acc']
        if acc > self.best_val_acc:
            torch.save(self.model.state_dict(), dev_model_weights)
            self.best_val_acc = acc.item()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)  # (..., T, n_notes)
        # full case
        for e in self.allowed_errors:
            correct = torch.sum((logits >= 0) != y.bool(), dim=-1) <= e
            acc = 100 * torch.sum(correct) / correct.nelement()
            self.log(
                f"Test accuracy (errors={e})",
                acc, on_epoch=True, prog_bar=True,
            )
        # only notes case
        label = y.unflatten(-1, (12, -1)).any(dim=-1)
        pred = (logits.unflatten(-1, (12, -1)) >= 0).any(dim=-1)
        for e in self.allowed_errors:
            correct = torch.sum(pred != label, dim=-1) <= e
            acc = 100 * torch.sum(correct) / correct.nelement()
            self.log(
                f"Test accuracy (errors={e}, only note names)",
                acc, on_epoch=True, prog_bar=True,
            )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    total_epochs: int,
    pl_class: type = LitMusicModel,
    milestones: list[int] = [],
    gamma: float = 1,
    val_loader: Optional[DataLoader] = None,
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01*lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    plmodel = pl_class(model, optimizer, scheduler)
    trainer = pl.Trainer(max_epochs=total_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(plmodel, train_loader, val_loader)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    pl_class: type = LitMusicModel,
    allowed_errors: list[int] = [0],
) -> None:
    """
    Checks the percentage of frames that
    were fully correctly classified
    """
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    pl_model = pl_class(model, allowed_errors=allowed_errors)
    trainer.test(pl_model, test_loader)


def load(model: nn.Module, dev: bool = False):
    """Load weights from 'model_weights'."""
    weights = dev_model_weights if dev else model_weights
    model.load_state_dict(torch.load(weights))


def save(model: nn.Module):
    """
    Saves the model into 'model_weights'. This is special
    because this file is reserved for the best model so far.
    """
    torch.save(model.state_dict(), model_weights)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    from dataloaders import create_dataloader

    model = MusicModel(c=6, all_notes=True)
    model.load_state_dict(torch.load("parameters\\dev_model_weights.pth"))
    train_loader = create_dataloader(split="train", batch_size=100)
    val_loader = create_dataloader(split="test", batch_size=32)

    # train(
    #     model,
    #     train_loader,
    #     lr=0.004,
    #     total_epochs=10,
    #     pl_class=LitMusicModel,
    #     milestones=[5],
    #     gamma=0.4,
    #     val_loader=val_loader,
    # )

    test(
        model,
        val_loader,
        pl_class=LitMusicModel,
        allowed_errors=[0, 1, 2],
    )


# Experiments:

# Multiple convolution layers or with very large kernels doesn't work very well

# I got basically the best performance from a single (5, 9) convolution
#   by adding more channels and residual connection.
# I used 12 features in 0.3s along 2 notes. lr=0.004 and gamma=0.4 every epoch

# I also noticed that repeating the linear layer per channel doesn't loose much
#   performance, if after that we use a max over channels and residual connection

# I also noticed that the maxpool is not too important in convolutional layers

# The 3d convolution helps a lot for its size, but it's painfully slow

# The temporal part of the convolution doesn't seem to help as much


# POSSIBLE IMPROVEMENTS

# make the frequency window larger
# add the normal interpolated spectogram since features at high frequencies are bad


# 3 IMPORANT FINDINGS

# linear layers or long-range convolutions w.r.t. time are useless, but small ones help a bit

# convolutions w.r.t. frequency must include at most 1.5 notes above/below

# lr=0.04 is perfect for 1 epoch
