"""Causal TCN model for per-frame foot contact classification.

The model outputs raw **logits** (no sigmoid).  Apply ``torch.sigmoid``
at inference time and use ``BCEWithLogitsLoss`` during training.  This
is numerically more stable than a sigmoid output layer with ``BCELoss``.

Architecture
------------
input_proj  : Linear(n_features → channels)
blocks      : N × _CausalConvBlock(channels, kernel_size, dilation=2^i)
output_proj : Linear(channels → 1)

Each _CausalConvBlock uses left-only padding so the convolution is
strictly causal (no future leakage).  Residual connections keep
gradients healthy despite the increasing dilation.

Input  : [batch, time, n_features]
Output : [batch, time]  (raw logits)

Receptive field vs n_dilations (kernel_size=3)
----------------------------------------------
Increasing ``n_dilations`` extends how far back in time the model can
attend.  Each block adds ``(kernel_size-1) * 2^i`` frames to the RF.
Pick a value that covers at least one full stride cycle.

============  ==========  ==========================
n_dilations   RF (frames)  RF at 30 fps
============  ==========  ==========================
3 (default)   15           ~0.5 s
4             31           ~1.0 s
5             63           ~2.1 s
6             127          ~4.2 s
============  ==========  ==========================

To train with wider context::

    python -m myogait.training.train_contact_tcn --n-dilations 5 ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConvBlock(nn.Module):
    """Dilated causal convolution block with residual connection.

    Pads the left (past) side only so each output position only
    sees frames at or before the current time step.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self._causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation,
            padding=0,  # manual causal padding applied in forward()
        )
        # LayerNorm over the channel dimension; normalises per (batch, time) step
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        residual = x
        out = F.pad(x, (self._causal_pad, 0))  # pad left only
        out = self.conv(out)                     # [batch, channels, time]
        # LayerNorm expects [..., channels]; transpose, normalise, transpose back
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        return out + residual


class ContactTCN(nn.Module):
    """Lightweight causal TCN for per-frame foot contact estimation.

    Parameters
    ----------
    n_features : int
        Number of input features per frame (default 17).
    channels : int
        Internal channel width (default 32).
    kernel_size : int
        Temporal kernel size for all conv blocks (default 3).
    n_dilations : int
        Number of TCN blocks.  Block *i* uses dilation 2^i, so
        with n_dilations=3 the effective receptive field is
        ``(kernel_size - 1) * (2^0 + 2^1 + 2^2) + 1 = 15`` frames
        (for kernel_size=3).

    Forward
    -------
    x : torch.Tensor, shape [batch, time, n_features]

    Returns
    -------
    logits : torch.Tensor, shape [batch, time]
        Raw pre-sigmoid contact logits.  Apply ``torch.sigmoid`` for
        probabilities.  Use ``BCEWithLogitsLoss`` for training.
    """

    def __init__(
        self,
        n_features: int = 17,
        channels: int = 32,
        kernel_size: int = 3,
        n_dilations: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_dilations = n_dilations

        self.input_proj = nn.Linear(n_features, channels)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(channels, kernel_size, dilation=2 ** i)
            for i in range(n_dilations)
        ])
        self.output_proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, n_features]
        out = self.input_proj(x)         # [batch, time, channels]
        out = out.transpose(1, 2)        # [batch, channels, time]
        for block in self.blocks:
            out = block(out)             # [batch, channels, time]
        out = out.transpose(1, 2)        # [batch, time, channels]
        logits = self.output_proj(out)   # [batch, time, 1]
        return logits.squeeze(-1)        # [batch, time]

    # ------------------------------------------------------------------ #
    # Convenience                                                           #
    # ------------------------------------------------------------------ #

    @property
    def receptive_field(self) -> int:
        """Effective temporal receptive field in frames."""
        return sum(
            (self.kernel_size - 1) * (2 ** i)
            for i in range(self.n_dilations)
        ) + 1

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
