from math import log
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
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.pos: torch.Tensor
        self.register_buffer("pos", self._make_pos_encoding(dim=dim, max_len=2048))

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def _make_pos_encoding(self, dim, max_len):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch, seq, dim = x.shape
        x = x + self.pos[None, :seq, :]

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class MusicTranscription(nn.Module):
    """
    ViT with (deep) convolutional encoder-decoder
    NOTE: the time and frequency resolutions are hard-coded
        at hop_size=256, bins_per_note=8
    """
    def __init__(self, dim=48, n_heads=3, depth=4, n_octaves=8):
        super().__init__()
        self.tokenizer = nn.Sequential(  # (time, freq) -> (time//6, freq//24)
            nn.Conv2d(1, dim//4, kernel_size=5, stride=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(dim//4, dim//2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim//2, dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
        )
        self.model = nn.Sequential(*(
            TransformerLayer(dim=dim, num_heads=n_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ))
        self.decoder = nn.Sequential(  # (time, freq) -> (time*6, freq*3)
            nn.ConvTranspose2d(dim, dim//2, kernel_size=(2, 3), stride=(2, 3)),
            nn.GELU(),
            nn.ConvTranspose2d(dim//2, 1, kernel_size=(3, 1), stride=(3, 1)),
        )
        self.n_octaves = n_octaves

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, time, freq)
        """
        x = x.unsqueeze(1)
        x = self.tokenizer(x)

        b, c, t, f = x.shape
        x = rearrange(x, "b c t f -> b (t f) c")
        x = self.model(x)
        x = x.unflatten(1, (t, f))

        x = rearrange(x, "b t f c -> b c t f")
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
        if loss < self.best_train_loss:
            torch.save(self.model.state_dict(), f"{self.params_root}\\{dev_model_weights}")
            self.best_train_loss = loss.item()

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

    model = MusicTranscription(dim=96, n_heads=3, depth=6, n_octaves=8)
    # model.load_state_dict(torch.load("parameters\\model_weights.pth"))
    train_loader = create_lazy_dataloader(split="train", batch_size=32, num_workers=8)
    val_loader = create_lazy_dataloader(split="test", batch_size=8, num_workers=0)

    train(
        model,
        train_loader,
        lr=1e-3,
        total_epochs=30,
        val_loader=val_loader,
    )

    test(
        model,
        val_loader,
        allowed_errors=[0, 1, 2],
    )
