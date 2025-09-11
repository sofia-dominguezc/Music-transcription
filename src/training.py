from typing import Optional, Sequence
from math import log

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import lightning as pl

second = 22050 // 512


model_weights = "parameters\\model_weights.pth"
dev_model_weights = "parameters\\dev_model_weights.pth"


def positional_encoding(seq: int, dim: int):
    """Sinusoidal positional encoding of shape (seq, dim)"""
    pe = torch.zeros(seq, dim)
    position = torch.arange(0, seq).unsqueeze(1)  # (seq, 1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-log(100) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, qk_dim: int, n_heads: int):
        super().__init__()
        assert qk_dim % n_heads == 0, "Head dims must divide evenly"

        self.qk_dim = qk_dim
        self.v_dim = qk_dim
        self.n_heads = n_heads
        self.qk_head_dim = qk_dim // n_heads
        self.v_head_dim = qk_dim // n_heads

        self.q_proj = nn.Linear(input_dim, qk_dim)
        self.k_proj = nn.Linear(input_dim, qk_dim)
        self.v_proj = nn.Linear(input_dim, self.v_dim)
        self.out_proj = nn.Linear(self.v_dim, input_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (batch, seq, input_dim)
        mask: (batch, seq)
        Attention is applied over seq
        """
        batch, seq, _ = x.shape

        Q = self.q_proj(x)  # [batch, seq, qk_dim]
        K = self.k_proj(x)  # [batch, seq, qk_dim]
        V = self.v_proj(x)  # [batch, seq, v_dim]

        Q = Q.view(batch, seq, self.n_heads, self.qk_head_dim).transpose(-2, -3)  # [batch, n_heads, seq, qk_head_dim]
        K = K.view(batch, seq, self.n_heads, self.qk_head_dim).transpose(-2, -3)  # [batch, n_heads, seq, qk_head_dim]
        V = V.view(batch, seq, self.n_heads, self.v_head_dim).transpose(-2, -3)  # [batch, n_heads, seq, v_head_dim]

        scores: Tensor = Q @ K.transpose(-2, -1) / (self.qk_head_dim ** 0.5)  # [batch, n_heads, seq, seq]
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :], value=float('-inf'))

        attn_weights = nn.functional.softmax(scores, dim=-1)  # [batch, n_heads, seq, seq]
        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq, v_head_dim]
        attn_output = attn_output.transpose(-2, -3).contiguous().view(batch, seq, self.v_dim)  # [batch, seq, v_dim]

        out = self.out_proj(attn_output)  # [batch, seq, input_dim]
        return out


class Transpose(nn.Module):
    """Transposes the desired pairs of dimensions in order"""
    def __init__(self, *dims: tuple[int, int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        for dim0, dim1 in self.dims:
            x = x.transpose(dim0, dim1)
        return x


class MusicModel(nn.Module):
    def __init__(self, c: int, all_notes: bool):
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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: (B, T, F)
        """
        y = x.unsqueeze(-3)  # (*B, 1, T, F)
        y = y[..., ::4] + self.conv(y)
        y = torch.max(y, dim=-3)[0] + self.linear(y)
        return y


class MusicTransformer(nn.Module):
    """
    The architecture was inspired by a Vision Transformer,
    except that it uses criss-cross attention instead of full attention
    """
    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 4,
        head_dim: int = 32,
        c: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()
        freq = 12 * 8 * 4  # input frequency dimension
        output_dim = 12 * 8  # 96

        self.tokenizer = nn.Sequential(  # (batch, freq) -> (batch, embed_dim)
            nn.Unflatten(-1, (1, -1)),
            nn.Conv1d(1, c, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv1d(c, c**2, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Flatten(-2),
            nn.Linear(c**2 * ((freq+3)//4), embed_dim),
        )
        self.pos_enc = nn.Buffer(positional_encoding(second, embed_dim), persistent=False)
        # self.pos_enc = nn.Conv1d(  # (batch, embed_dim, time)
        #     embed_dim,
        #     embed_dim,
        #     kernel_size=5,
        #     padding=2,
        #     groups=embed_dim,
        # )
        self.decoder = nn.Linear(embed_dim, output_dim)

        self.attn = nn.ModuleList(
            (
                SelfAttention(input_dim=embed_dim, qk_dim=head_dim*n_heads, n_heads=n_heads)
                for _ in range(n_layers)
            ),
        )
        self.layer_norm = nn.ModuleList(
            (nn.LayerNorm(embed_dim) for _ in range(n_layers))
        )
        self.mlp = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Linear(embed_dim, 4*embed_dim),
                    nn.GELU(),
                    nn.Linear(4*embed_dim, embed_dim),
                ) for _ in range(n_layers)
            ),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (batch, second, freq)
        mask: (batch, second)
        Self-attention over time with CNN as positional encoding
        """
        batch = x.shape[0]
        x = self.tokenizer(x.flatten(0, 1)).unflatten(0, (batch, -1))  # (batch, time, embed_dim)
        # x = x + self.pos_enc(x.transpose(-2, -1)).transpose(-2, -1)
        x = x + self.pos_enc[None, :, :]

        for attn, mlp, ln in zip(self.attn, self.mlp, self.layer_norm):
            x = ln(x)
            x = x + attn(x, mask)
            x = x + mlp(x)

        x = self.decoder(x)  # (batch, time, output_dim)
        return x


class LitMusicModel(pl.LightningModule):
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_val_acc: float

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        allowed_errors: list[int] = [0],
        params_root: str = ".",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.allowed_errors = allowed_errors
        self.params_root = params_root
        self.best_train_loss = float('inf')

    def configure_optimizers(self):  # type: ignore
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _loss(self, X: Sequence[Tensor], y: Tensor, logits: Tensor) -> Tensor:
        if len(X) == 1:
            return torch.mean(self.loss_fn(logits, y))
        else:
            x, mask = X
            valid = ~mask[:, :, None]
            loss = self.loss_fn(logits, y) * valid
            return torch.sum(loss) / (valid.sum() * y.shape[-1])

    def training_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
    ) -> Tensor:
        *X, y = [el.to(self.device) for el in batch]
        logits = self.model(*X)
        loss = self._loss(X, y, logits)

        self.log("loss_step", 100 * loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", 100 * loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics['loss_epoch']
        print(f"\nEpoch {self.current_epoch} - train_loss: {loss:.4f}")
        if loss < self.best_train_loss:
            torch.save(self.model.state_dict(), f"{self.params_root}\\{dev_model_weights}")
            self.best_train_loss = loss.item()

    def _acc(self, X: Sequence[Tensor], y: Tensor, pred: Tensor, e: int = 0) -> float:
        """Find fraction of time steps that are fully correctly classified"""
        if len(X) == 1:
            correct = torch.sum(pred != y.bool(), dim=-1) <= e  # (batch, time)
            acc = torch.sum(correct) / correct.nelement()
            return 100 * acc.item()
        else:
            _, mask = X
            valid = ~mask
            correct = torch.sum(pred != y.bool(), dim=-1) <= e
            acc = torch.sum(correct & valid) / valid.sum()
            return 100 * acc.item()

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        *X, y = [el.to(self.device) for el in batch]
        logits = self.model(*X)  # (..., T, n_notes)
        loss = self._loss(X, y, logits)
        acc = self._acc(X, y, pred=(logits >= 0), e=0)
        self.log("val_loss", 100 * loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
    ) -> None:
        *X, y = [el.to(self.device) for el in batch]
        logits = self.model(*X)  # (batch, time, notes)
        # full case
        for e in self.allowed_errors:
            acc = self._acc(X, y, pred=(logits >= 0), e=e)
            self.log(
                f"Test accuracy (errors={e})",
                acc, on_epoch=True, prog_bar=True,
            )
        # only notes case
        label = y.unflatten(-1, (12, -1)).any(dim=-1)
        pred = (logits.unflatten(-1, (12, -1)) >= 0).any(dim=-1)
        for e in self.allowed_errors:
            acc = self._acc(X, label, pred, e=e)
            self.log(
                f"Test accuracy (errors={e}, only note names)",
                acc, on_epoch=True, prog_bar=True,
            )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    total_epochs: int,
    val_loader: Optional[DataLoader] = None,
    params_root: str = ".",
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=10*lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=0.01*lr)
    plmodel = LitMusicModel(model, optimizer, scheduler, params_root=params_root)
    trainer = pl.Trainer(max_epochs=total_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(plmodel, train_loader, val_loader)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    allowed_errors: list[int] = [0],
) -> None:
    """
    Checks the percentage of frames that
    were fully correctly classified
    """
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    pl_model = LitMusicModel(model, allowed_errors=allowed_errors)
    trainer.test(pl_model, test_loader)


def load(model: nn.Module, dev: bool = False):
    """Load weights from 'model_weights'."""
    weights = dev_model_weights if dev else model_weights
    model.load_state_dict(torch.load(weights))


def save(model: nn.Module):
    """
    Saves the model into 'model_weights'
    This file is reserved for the best model so far.
    """
    torch.save(model.state_dict(), model_weights)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    from dataloaders import create_lazy_dataloader

    model = MusicTransformer(n_layers=4, n_heads=4, head_dim=32, c=3, embed_dim=192)
    # model.load_state_dict(torch.load("parameters\\model_weights.pth"))
    train_loader = create_lazy_dataloader(split="train", batch_size=8, num_workers=8)
    val_loader = create_lazy_dataloader(split="test", batch_size=1, num_workers=0)

    train(
        model,
        train_loader,
        lr=5e-4,
        total_epochs=40,
        val_loader=val_loader,
    )

    test(
        model,
        val_loader,
        allowed_errors=[0, 1, 2],
    )

    # TODO: batch by second before doing self-attention so it doesn't take this long


# Experiments:

# Multiple convolution layers or with very large kernels doesn't work very well

# I also noticed that repeating the linear layer per channel doesn't loose much
#   performance, if after that we use a max over channels and residual connection

# I also noticed that the maxpool is not too important in convolutional layers

# The temporal part of the convolution doesn't seem to help as much


# 3 IMPORANT FINDINGS

# linear layers or long-range convolutions w.r.t. time are useless, but small ones help a bit

# convolutions w.r.t. frequency must include at most 1.5 notes above/below

# lr=0.04 is perfect for 1 epoch


# Normal transformer

# using n_layers=4, n_heads=4, head_dim=32, c=3, embed_dim=192:
# best lr: 5e-4, test acc = 40%, it overfits at 22/40 epochs (when losses are about 3.6)
