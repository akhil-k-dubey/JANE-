"""Differentiable Neural Computer (DNC): Internal memory tape for multi-step latent reasoning."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DNCMemoryTape(nn.Module):
    """
    Differentiable scratchpad / Turing-machine memory tape.
    Allows multi-step reasoning by writing variables to tape, executing intermediate steps,
    reading accumulated values, and updating memory slots differentiably.
    """

    def __init__(
        self,
        controller_dim: int = 256,
        memory_slots: int = 128,
        memory_dim: int = 64,
        read_heads: int = 2
    ):
        super().__init__()
        self.controller_dim = controller_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.read_heads = read_heads

        # Write interface
        self.write_key = nn.Linear(controller_dim, memory_dim)
        self.write_strength = nn.Linear(controller_dim, 1)
        self.erase_vector = nn.Linear(controller_dim, memory_dim)
        self.add_vector = nn.Linear(controller_dim, memory_dim)
        self.write_gate = nn.Linear(controller_dim, 1)

        # Read interface (multiple heads to read multiple variables simultaneously)
        self.read_keys = nn.Linear(controller_dim, read_heads * memory_dim)
        self.read_strengths = nn.Linear(controller_dim, read_heads)

        # Projection back to latent dimension
        self.read_to_latent = nn.Linear(read_heads * memory_dim, controller_dim)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        slots: Optional[int] = None
    ) -> torch.Tensor:
        """Initializes empty memory tape matrix [batch, memory_slots, memory_dim]. Supports dynamic slot scaling."""
        num_slots = slots or self.memory_slots
        return torch.zeros(batch_size, num_slots, self.memory_dim, device=device, dtype=dtype)

    def _address_content(self, memory: torch.Tensor, key: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        """Content-based addressing using cosine similarity weighted by key strength."""
        norm_mem = F.normalize(memory, p=2, dim=-1, eps=1e-6)
        norm_key = F.normalize(key, p=2, dim=-1, eps=1e-6).unsqueeze(1)
        similarity = torch.sum(norm_mem * norm_key, dim=-1)
        logits = similarity * F.softplus(strength)
        weights = F.softmax(logits, dim=-1)
        return weights

    def write(
        self,
        memory: torch.Tensor,
        controller: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiably erases and writes information to addressed memory slots.
        """
        key = self.write_key(controller)
        beta = self.write_strength(controller)
        write_weights = self._address_content(memory, key, beta).unsqueeze(-1)  # [batch, slots, 1]

        erase = torch.sigmoid(self.erase_vector(controller)).unsqueeze(1)       # [batch, 1, mem_dim]
        add = torch.tanh(self.add_vector(controller)).unsqueeze(1)              # [batch, 1, mem_dim]
        gate = torch.sigmoid(self.write_gate(controller)).unsqueeze(-1)         # [batch, 1, 1]

        # Memory update: M_new = M * (1 - w * e) + w * a
        erased_mem = memory * (1.0 - gate * (write_weights * erase))
        new_memory = erased_mem + gate * (write_weights * add)
        return new_memory, write_weights.squeeze(-1)

    def read(
        self,
        memory: torch.Tensor,
        controller: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads variables from memory tape using read heads.
        """
        batch = controller.size(0)
        keys = self.read_keys(controller).view(batch, self.read_heads, self.memory_dim)
        strengths = self.read_strengths(controller)

        read_vectors = []
        read_weights_list = []
        for h in range(self.read_heads):
            head_key = keys[:, h, :]
            head_str = strengths[:, h:h+1]
            w = self._address_content(memory, head_key, head_str)
            head_read = torch.sum(w.unsqueeze(-1) * memory, dim=1)
            read_vectors.append(head_read)
            read_weights_list.append(w)

        combined_reads = torch.cat(read_vectors, dim=-1)  # [batch, read_heads * mem_dim]
        read_latent = self.read_to_latent(combined_reads)
        return read_latent, torch.stack(read_weights_list, dim=1)

    def step(
        self,
        controller: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full Turing machine step: Read current tape -> Write step update -> Return read latent and updated tape.
        """
        if memory is None:
            memory = self.init_memory(controller.size(0), controller.device, controller.dtype)

        # First read intermediate values needed for this step
        read_latent, _ = self.read(memory, controller)

        # Next write new variable/calculation to tape
        new_memory, _ = self.write(memory, controller)

        return read_latent, new_memory
