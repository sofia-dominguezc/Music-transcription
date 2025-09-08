from typing import Optional, Iterable, Callable
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import lightning as pl

model_weights = "parameters\\model_weights.pth"
dev_model_weights = "parameters\\dev_model_weights.pth"


# class PositionalEncoding(nn.Module):
#     def __init__(self, n_freq: int, n_time: int) -> None:
#         super().__init__()
#         pe = torch.zeros(n_time, n_freq)
#         position = torch.arange(0, n_time).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, n_freq, 2) * (-log(10000) / n_freq)).float()
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
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
        Q = self.q_proj(x)  # [batch, seq, qk_dim]
        K = self.k_proj(x)  # [batch, seq, qk_dim]
        V = self.v_proj(x)  # [batch, seq, v_dim]

        Q = Q.view(-1, self.n_heads, self.qk_head_dim).transpose(-2, -3)  # [batch, n_heads, seq, qk_head_dim]
        K = K.view(-1, self.n_heads, self.qk_head_dim).transpose(-2, -3)  # [batch, n_heads, seq, qk_head_dim]
        V = V.view(-1, self.n_heads, self.v_head_dim).transpose(-2, -3)  # [batch, n_heads, seq, v_head_dim]

        scores: Tensor = Q @ K.transpose(-2, -1) / (self.qk_head_dim ** 0.5)  # [batch, n_heads, seq, seq]
        if mask:
            scores = scores.masked_fill(mask[:, None, None, :], value=float('-inf'))

        attn_weights = nn.functional.softmax(scores, dim=-1)  # [batch, n_heads, seq, seq]
        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq, v_head_dim]
        attn_output = attn_output.transpose(-2, -3).contiguous().view(-1, self.v_dim)  # [batch, seq, v_dim]

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

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, F)
        """
        y = x.unsqueeze(-3)  # (*B, 1, T, F)
        y = y[..., ::4] + self.conv(y)
        y = torch.max(y, dim=-3)[0] + self.linear(y)
        return y

# ((x + 1) // 2 + 1) // 2
# 0, 1, 1, 1
# x // 4 + (x % 4 == 0)
# (x + 3) // 4

class MusicTransformer(nn.Module):
    """
    The architecture was inspired by a Vision Transformer,
    except that it uses criss-cross attention instead of full attention
    """
    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 4,
        head_dim: int = 64,
        c: int = 3,
    ):
        super().__init__()
        self.time = 22050 // 512  # time values in 1s
        notes = 12
        self.freq = notes * 4  # freq values in 1 octave
        output_dim = self.time * notes
        embed_dim = output_dim // 4

        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv2d(c, c**2, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Flatten(-3),
            nn.Linear(c**2 * (self.time+3)//4 * (self.freq+3)//4, self.time * notes),
        )
        self.pos_enc = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.decoder = nn.Linear(embed_dim, output_dim)

        self.t_attn = nn.ModuleList(
            (
                SelfAttention(input_dim=embed_dim, qk_dim=head_dim*n_heads, n_heads=n_heads)
                for _ in range(n_layers)
            ),
        )
        self.f_attn = nn.ModuleList(
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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: (batch, t_batch*time, f_batch*freq)
        mask: (batch, t_batch*time)
        Independent self-attentions over time and frequency
        """
        x = x.unflatten(-2, (-1, self.time)).unflatten(-1, (-1, self.freq))  # (B, tB, t, fB, f)
        x = x.transpose(-3, -2).unsqueeze(3)  # (batch, t_batch, f_batch, 1, time, freq)
        x = self.tokenizer(x.flatten(0, 2)).view(*x.shape[:3], -1)  # (batch, t_batch, f_batch, embed_dim)
        x = x + self.pos_enc(x.transpose(-1, -3)).transpose(-1, -3)

        batch, t_batch, f_batch = x.shape[:3]
        mask = mask.view(batch, t_batch, -1)  # (batch, t_batch, time)
        assert mask.max(dim=-1) == mask.min(dim=-1), "Mask is not constant over seconds"
        mask = mask[:, None, :, 0].repeat((f_batch, 1, 1)).flatten(0, 1)  # (batch*f_batch, t_batch)

        for t_attn, f_attn, mlp, ln in zip(self.t_attn, self.f_attn, self.mlp, self.layer_norm):
            x = ln(x)
            fx = f_attn(x.flatten(0, 1)).unflatten(0, (batch, t_batch))
            tx = t_attn(x.transpose(1, 2).flatten(0, 1), mask).unflatten(0, (batch, f_batch)).transpose(1, 2)
            x = x + tx + fx
            x = x + mlp(x)

        x = self.decoder(x)  # (batch, t_batch, f_batch, output_dim)
        x = x.unflatten(-1, (self.time, -1))  # (batch, t_batch, f_batch, time, notes)
        return x.transpose(-3, -2).flatten(-1).flatten(-2)  # (batch, t_batch*time, f_batch*notes)


class LitMusicModel(pl.LightningModule):
    loss_fn = nn.BCEWithLogitsLoss()
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
        batch: tuple[Tensor, ...],
        batch_idx: int,
    ) -> Tensor:
        *x, y = [el.to(self.device) for el in batch]
        logits = self.model(*x)  # (..., T, n_notes)
        loss = self.loss_fn(logits, y)

        self.log("loss_step", 100 * loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", 100 * loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get('loss_epoch')
        if loss is not None:
            print(f"\nEpoch {self.current_epoch} - train_loss: {loss:.4f}")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        *x, y = [el.to(self.device) for el in batch]
        logits = self.model(*x)  # (..., T, n_notes)
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
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
    ) -> None:
        *x, y = [el.to(self.device) for el in batch]
        logits = self.model(*x)  # (..., T, n_notes)
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
    milestones: list[int] = [],
    gamma: float = 1,
    val_loader: Optional[DataLoader] = None,
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01*lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    plmodel = LitMusicModel(model, optimizer, scheduler)
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
    from dataloaders import create_dataloader

    model = MusicTransformer(n_layers=2, n_heads=4, head_dim=32, c=3)
    # model.load_state_dict(torch.load("parameters\\dev_model_weights.pth"))
    train_loader = create_dataloader(split="train", batch_size=32, num_workers=8)
    val_loader = create_dataloader(split="test", batch_size=32, num_workers=8)

    train(
        model,
        train_loader,
        lr=0.004,
        total_epochs=1,
        milestones=[5],
        gamma=0.4,
        val_loader=val_loader,
    )

    test(
        model,
        val_loader,
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
