from math import log
from typing import Any
import lightning as pl
from einops import rearrange

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

model_weights = "parameters\\model_weights.pth"
dev_model_weights = "parameters\\dev_model_weights.pth"


class TransformerLayer(nn.Module):
    def __init__(self, dim=48, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        hidden_dim = int(dim * mlp_ratio)
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        batch, seq1, seq2, dim = x.shape
        x_norm = self.norm1(x)

        xf = rearrange(x_norm, "b t f d -> (b t) f d")
        attn_out, _ = self.attn(xf, xf, xf)
        attn_out = attn_out.unflatten(0, (batch, seq1))
        x = x + attn_out

        x_norm = self.norm2(x)
        x_norm = rearrange(x_norm, "b t f d -> (b f) d t")
        mlp_out = self.mlp(x_norm)
        mlp_out = rearrange(mlp_out.unflatten(0, (batch, seq2)), "b f d t -> b t f d")
        x = x + mlp_out

        return x


class MusicTranscription(nn.Module):
    """
    ViT in frequency + CNN in time
    """
    def __init__(self, dim=96, n_heads=3, depth=3):
        super().__init__()
        self.encoder = nn.Sequential(  # (time, freq) -> (time//12, freq//24)
            nn.Conv2d(1, dim//4, kernel_size=5, stride=3),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim//4, dim//2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim//2, dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
        )
        self.model = nn.Sequential(*(
            TransformerLayer(dim=dim, num_heads=n_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ))
        self.decoder = nn.ConvTranspose2d(dim, 1, kernel_size=(12, 3), stride=(12, 3))

        self.pos: torch.Tensor
        self.register_buffer("pos", self._make_pos_encoding(dim=dim, max_len=64))

    def _make_pos_encoding(self, dim, max_len):
        pe = torch.zeros(max_len, dim)
        t = torch.arange(0, max_len, dtype=torch.float)[:, None]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(t * div_term)
        pe[:, 1::2] = torch.cos(t * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, time, freq)
        """
        x = x.unsqueeze(1)
        x = self.encoder(x)
        batch, _, time, freq = x.shape

        x = rearrange(x, "b d t f -> b t f d")
        x = x + self.pos[None, None, :freq, :]
        x = self.model(x)

        x = rearrange(x, "b t f d -> b d t f")
        x = self.decoder(x)
        return x.squeeze(1)


class LitMusicModel(pl.LightningModule):
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    best_val_acc: float

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        allowed_errors: list[int] = [0],
        params_root: str = ".",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.allowed_errors = allowed_errors
        self.params_root = params_root
        self.best_val_acc = 0.0

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
        batch: tuple[Tensor, ...],
        batch_idx: int,
    ) -> Tensor:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        self.log("loss_step", loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics['loss_epoch']
        print(f"\nEpoch {self.current_epoch} - train_loss: {loss:.4f}")

    def _acc(
        self, label: Tensor, pred: Tensor, e: int = 0,
    ) -> float:
        """Find fraction of time steps that are classified with <= e errors"""
        correct = torch.sum(pred != label, dim=-1) <= e  # (batch, time)
        acc = torch.sum(correct) / correct.nelement()
        return 100 * acc.item()

    def validation_step(
        self, batch: tuple[Tensor, ...], batch_idx: int,
    ) -> None:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)  # (..., T, n_notes)
        loss = self.loss_fn(logits, y)
        acc = self._acc(label=y.bool(), pred=(logits >= 0), e=0)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.trainer.callback_metrics['val_acc']
        if acc > self.best_val_acc:
            torch.save(self.model.state_dict(), f"{self.params_root}\\{dev_model_weights}")
            self.best_val_acc = acc.item()

    def test_step(
        self, batch: tuple[Tensor, ...], batch_idx: int,
    ) -> None:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        # full case
        for e in self.allowed_errors:
            acc = self._acc(label=y.bool(), pred=(logits >= 0), e=e)
            self.log(
                f"Test accuracy (errors={e})",
                acc, on_epoch=True, prog_bar=True,
            )
        # only notes case
        label = y.unflatten(-1, (-1, 12)).any(dim=-2)
        pred = (logits.unflatten(-1, (-1, 12)) >= 0).any(dim=-2)
        for e in self.allowed_errors:
            acc = self._acc(label, pred, e=e)
            self.log(
                f"Test accuracy (errors={e}, only note names)",
                acc, on_epoch=True, prog_bar=True,
            )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    total_epochs: int,
    val_loader: DataLoader | None = None,
    params_root: str = ".",
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01*lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=0.03*lr)
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

    model = MusicTranscription(dim=192, n_heads=6, depth=6)
    model.load_state_dict(torch.load("parameters\\model_weights.pth"))
    train_loader = create_lazy_dataloader(split="train", batch_size=32, num_workers=8)
    val_loader = create_lazy_dataloader(split="test", batch_size=16, num_workers=0)

    train(
        model,
        train_loader,
        lr=3e-3,
        total_epochs=40,
        val_loader=val_loader,
    )

    test(
        model,
        val_loader,
        allowed_errors=[0, 1],
    )
