"""Energy-Based Constraints with Dual-Head Concept Bottlenecks (Magnitude + SI Dimension)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.symbolic_verifier.contracts import Dimension, PhysicalQuantity, DimensionalQuantity


SI_DIMENSIONS = [
    "MASS",           # [M] (kg)
    "LENGTH",         # [L] (m)
    "TIME",           # [T] (s)
    "CURRENT",        # [I] (A)
    "TEMPERATURE",    # [Θ] (K)
    "VELOCITY",       # [L]/[T] (m/s)
    "ENERGY",         # [M][L]^2/[T]^2 (J)
    "DIMENSIONLESS"   # [1]
]


class DualHeadConceptBottleneck(nn.Module):
    """
    Concept Bottleneck Architecture:
    The neural network cannot output raw numbers.
    It is partitioned into two parallel heads:
    1. Magnitude Head: Emits continuous scalar float values.
    2. Dimension Head: Emits categorical logits over SI physical dimensions.
    """

    def __init__(self, latent_dim: int = 256, num_channels: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_dimensions = len(SI_DIMENSIONS)

        # 1. Magnitude Head
        self.magnitude_head = nn.Linear(latent_dim, num_channels)

        # 2. Dimension Head (Categorical probability distribution over SI dimensions)
        self.dimension_head = nn.Linear(latent_dim, num_channels * self.num_dimensions)

        # Channel name tags for inspectability
        self.channel_names = [
            "entity_mass",
            "position_x",
            "position_y",
            "system_energy_in",
            "system_energy_out",
            "kinematic_velocity"
        ]

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            magnitudes: [batch, num_channels]
            dimension_probs: [batch, num_channels, num_dimensions]
        """
        batch = latent.size(0)
        mags = self.magnitude_head(latent)
        dim_logits = self.dimension_head(latent).view(batch, self.num_channels, self.num_dimensions)
        dim_probs = F.softmax(dim_logits, dim=-1)
        return mags, dim_probs

    def extract_quantities(self, latent: torch.Tensor) -> List[PhysicalQuantity]:
        """
        Extracts formal PhysicalQuantity objects for single instance.
        Uses Dimension Head to assign formal Dimension Enum and explicit direction.
        No raw dictionaries or guessing via substring matching.
        """
        mags, dim_probs = self.forward(latent[:1])
        quantities = []

        channel_configs = [
            (Dimension.MASS, False, False),       # 0: entity_mass
            (Dimension.LENGTH, False, False),     # 1: position_x
            (Dimension.LENGTH, False, False),     # 2: position_y
            (Dimension.ENERGY, True, False),      # 3: system_energy_in (explicit input)
            (Dimension.ENERGY, False, True),      # 4: system_energy_out (explicit output)
            (Dimension.VELOCITY, False, False),   # 5: kinematic_velocity
        ]

        for i, (expected_dim, is_in, is_out) in enumerate(channel_configs):
            val = mags[0, i].item()
            conf, dim_idx = torch.max(dim_probs[0, i], dim=-1)
            classified_dim = Dimension[SI_DIMENSIONS[dim_idx.item()]]

            quantities.append(PhysicalQuantity(
                magnitude=val,
                dimension=classified_dim,
                confidence=conf.item(),
                is_input=is_in,
                is_output=is_out
            ))

        return quantities


class PhysicsBarrier(nn.Module):
    """
    Scale-invariant physics barrier evaluated on continuous variables
    and dimensional constraints.
    """

    def __init__(self, barrier_weight: float = 100.0, min_confidence: float = 0.85):
        super().__init__()
        self.barrier_weight = barrier_weight
        self.min_confidence = min_confidence

    def forward(self, mags: torch.Tensor, dim_probs: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates penalty energy.
        Penalizes both physical violations (e.g. negative mass, broken conservation)
        AND dimensional ambiguity (if network is $<85\%$ sure about physical dimensions).
        """
        device = mags.device
        penalty = torch.tensor(0.0, device=device)

        # 1. Dimensional Ambiguity Penalty:
        # Forces network to develop sharp, confident physical concepts
        max_confs, _ = torch.max(dim_probs, dim=-1)
        ambiguity_viol = F.relu(self.min_confidence - max_confs) ** 2
        penalty = penalty + self.barrier_weight * torch.sum(ambiguity_viol)

        # 2. Invariant Physical Penalties
        # Channel 0: entity_mass (must be >= 0)
        mass_val = mags[:, 0]
        penalty = penalty + self.barrier_weight * torch.sum(F.relu(-mass_val) ** 2)

        # Channels 1 & 2: continuous normalized position in [-1, 1]
        pos_val = mags[:, 1:3]
        pos_viol = F.relu(-1.0 - pos_val) ** 2 + F.relu(pos_val - 1.0) ** 2
        penalty = penalty + self.barrier_weight * torch.sum(pos_viol)

        # Channels 3 & 4: Energy conservation (E_out <= E_in)
        e_in = mags[:, 3]
        e_out = mags[:, 4]
        e_viol = F.relu(e_out - e_in) ** 2
        penalty = penalty + self.barrier_weight * torch.sum(e_viol)

        return penalty


class EnergyConstraintLandscape(nn.Module):
    """
    Energy landscape coupling learned neural dynamics with the Dual-Head Concept Bottleneck.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        barrier_weight: float = 100.0,
        settling_steps: int = 5,
        step_size: float = 0.05
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.barrier_weight = barrier_weight
        self.settling_steps = settling_steps
        self.step_size = step_size

        # Learned neural coherence energy
        self.learned_energy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Dual-Head Concept Bottleneck
        self.concept_bottleneck = DualHeadConceptBottleneck(latent_dim=latent_dim)
        # Physics Invariant Barrier
        self.physics_barrier = PhysicsBarrier(barrier_weight=barrier_weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        learned_e = self.learned_energy_net(latent).squeeze(-1)
        mags, dim_probs = self.concept_bottleneck(latent)
        barrier_e = self.physics_barrier(mags, dim_probs)
        return learned_e + barrier_e

    def ranking_loss(self, feasible_latent: torch.Tensor, infeasible_latent: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        e_feasible = self.forward(feasible_latent)
        e_infeasible = self.forward(infeasible_latent)
        return F.relu(margin + e_feasible - e_infeasible).mean()

    def settle_equilibrium(
        self,
        latent: torch.Tensor,
        steps: Optional[int] = None,
        step_size: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_steps = steps or self.settling_steps
        lr = step_size or self.step_size

        z = latent.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            for _ in range(n_steps):
                energy = self.forward(z).sum()
                grad = torch.autograd.grad(energy, z, create_graph=False)[0]
                z = (z - lr * grad).detach().requires_grad_(True)

        final_energy = self.forward(z.detach())
        return z.detach(), final_energy.detach()
