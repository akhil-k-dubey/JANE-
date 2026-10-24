"""Transformer Encoder: Language -> 256D Latent Vector Space."""

import torch
import torch.nn as nn
from typing import Optional

class TextToManifoldEncoder(nn.Module):
    """Encodes linguistic tokens into Jane's 256-dimensional continuous latent space."""

    def __init__(
        self,
        vocab_size: int = 5000,
        encoder_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, encoder_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, encoder_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len]
            mask: Optional boolean padding mask [batch_size, seq_len]
        Returns:
            latent_vectors: [batch_size, seq_len, latent_dim]
        """
        seq_len = token_ids.size(1)
        x = self.embedding(token_ids) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x, src_key_padding_mask=mask)
        latent = self.to_latent(x)
        return latent

    def pool_state(self, latent_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pools sequence vectors into a single 256D concept anchor for the manifold."""
        if mask is not None:
            weights = (~mask).unsqueeze(-1).float()
            return (latent_vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return latent_vectors.mean(dim=1)
