"""Training Stage 2: Train the Energy-Based Boundaries and Physics Brick Walls."""

import os
import sys
import torch
from torch.optim import AdamW

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning_engine.energy_constraints import EnergyConstraintLandscape


def train_energy_landscape(epochs: int = 10, batch_size: int = 16, lr: float = 1e-3, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2] Training Energy Landscape on device: {device}")

    energy_model = EnergyConstraintLandscape(latent_dim=256, hidden_dim=512, barrier_weight=100.0).to(device)
    optimizer = AdamW(energy_model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Synthesize feasible physical states
        feasible_latents = torch.randn(batch_size, 256, device=device) * 0.5
        with torch.no_grad():
            phys = energy_model.concept_decoder(feasible_latents)
            phys["mass"].clamp_min_(0.1)
            # Continuous normalized coordinates in [-1, 1]
            phys["position"].clamp_(-0.9, 0.9)
            phys["energy_in"].clamp_min_(100.0)
            phys["energy_out"] = phys["energy_in"] * 0.7  # 70% efficiency

        # 2. Synthesize corrupted infeasible states
        infeasible_latents = feasible_latents.clone() + torch.randn_like(feasible_latents) * 0.2
        with torch.no_grad():
            infeasible_phys = energy_model.concept_decoder(infeasible_latents)
            infeasible_phys["mass"].fill_(-5.0)  # Impossible negative mass
            infeasible_phys["energy_out"] = infeasible_phys["energy_in"] + 50.0  # Impossible free energy

        # Ranking loss: E(feasible) + margin < E(infeasible)
        loss = energy_model.ranking_loss(feasible_latents, infeasible_latents, margin=2.0)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            e_feas = energy_model(feasible_latents).mean().item()
            e_infeas = energy_model(infeasible_latents).mean().item()
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} | E(Feasible): {e_feas:.2f} | E(Violating): {e_infeas:.2f}")

    os.makedirs("logs/training_runs", exist_ok=True)
    torch.save(energy_model.state_dict(), "logs/training_runs/stage2_energy.pt")
    print("[Stage 2] Checkpoint saved to logs/training_runs/stage2_energy.pt")


if __name__ == "__main__":
    train_energy_landscape()
