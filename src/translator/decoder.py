"""Transformer Decoder: Proven/Settled Latent Vectors -> English Text."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LatentToTextDecoder(nn.Module):
    """Translates Jane's verified, settled latent thoughts back into tokenized English."""

    def __init__(
        self,
        vocab_size: int = 5000,
        latent_dim: int = 256,
        decoder_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, decoder_dim) * 0.02)
        
        self.from_latent = nn.Linear(latent_dim, decoder_dim) if latent_dim != decoder_dim else nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_projection = nn.Linear(decoder_dim, vocab_size)

    def forward(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            target_ids: [batch, tgt_len] (teacher-forced target tokens)
            memory: [batch, mem_len, latent_dim] (settled manifold/scratchpad vectors)
        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        batch, tgt_len = target_ids.size()
        x = self.embedding(target_ids) + self.pos_embedding[:, :tgt_len, :]

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=target_ids.device)

        if memory.dim() == 2:
            memory = memory.unsqueeze(1)

        mem = self.from_latent(memory)
        out = self.transformer(tgt=x, memory=mem, tgt_mask=tgt_mask)
        return self.out_projection(out)

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_steps: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Verbalizes settled latent memory into token sequences using temperature-scaled
        top-p (nucleus) sampling or greedy argmax when temperature == 0.0.
        """
        batch = memory.size(0)
        generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=memory.device)

        for _ in range(max_steps):
            logits = self.forward(generated, memory)
            last_logits = logits[:, -1, :]
            # Anti-NaN firewall: protect against upstream NALU/memory math explosions
            last_logits = torch.nan_to_num(last_logits, nan=0.0, posinf=1e4, neginf=-1e4)

            if temperature > 0.0:
                scaled_logits = last_logits / max(temperature, 1e-4)
                probs = F.softmax(scaled_logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Mask tokens outside nucleus
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                    prob_sum = torch.clamp(sorted_probs.sum(dim=-1, keepdim=True), min=1e-8)
                    sorted_probs = torch.nan_to_num(sorted_probs / prob_sum, nan=0.0)
                    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, sampled_idx)
                else:
                    probs = torch.nan_to_num(probs, nan=0.0)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = last_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return generated
