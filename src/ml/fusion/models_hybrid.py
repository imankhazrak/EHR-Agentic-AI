"""Conv1d + Transformer encoder (exactly 4 attention heads) for dense fusion vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionHybridCNNTransformer(nn.Module):
    """Project flat features to a short token sequence, local Conv1d, TransformerEncoder (nhead=4), pool, classify."""

    def __init__(
        self,
        in_dim: int,
        n_tokens: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        if nhead != 4:
            raise ValueError("This fusion hybrid requires exactly nhead=4 per experiment spec.")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        flat_out = n_tokens * d_model
        self.proj = nn.Linear(in_dim, flat_out)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        b = x.shape[0]
        h = self.proj(x).view(b, self.n_tokens, self.d_model)
        # Conv1d on (B, C, L)
        z = h.transpose(1, 2)
        z = torch.relu(self.conv(z))
        z = z.transpose(1, 2)
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)
