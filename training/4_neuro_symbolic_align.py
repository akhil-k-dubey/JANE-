"""Training Stage 4: Neuro-Symbolic Alignment (Train Neural Net to output Z3-readable logic)."""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning_engine.nalu_arithmetic import NALUCluster
from src.symbolic_verifier.contracts import MathHypothesis
from src.symbolic_verifier.z3_interface import Z3LogicVerifier


def train_neuro_symbolic_alignment(epochs: int = 15, batch_size: int = 16, lr: float = 2e-3, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Stage 4] Neuro-Symbolic Alignment on device: {device}")

    nalu = NALUCluster(latent_dim=256, math_dim=32, num_layers=2).to(device)
    verifier = Z3LogicVerifier()
    optimizer = AdamW(nalu.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Batch of arithmetic tasks: a + b
        a = torch.randint(1, 20, (batch_size, 1), device=device).float()
        b = torch.randint(1, 20, (batch_size, 1), device=device).float()
        expected = a + b

        numbers = torch.cat([a, b], dim=-1)
        latent_context = torch.randn(batch_size, 256, device=device)

        math_anchor, raw_math = nalu(latent_context, explicit_numbers=numbers)
        pred_val = raw_math[:, 0:1]

        loss = loss_fn(pred_val, expected)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 3 == 0 or epoch == 0:
            sample_a = a[0].item()
            sample_b = b[0].item()
            sample_pred = pred_val[0].item()

            # Safe AST-based Z3 verification (Zero eval)
            hyp = MathHypothesis(lhs=f"{sample_a} + {sample_b}", rhs=f"{sample_pred:.2f}", tolerance=0.1)
            proof = verifier.verify_math(hyp)

            print(f"Epoch {epoch + 1}/{epochs} - Alignment Loss: {loss.item():.4f} | Proof: {proof.proof_log} | Valid: {proof.is_valid}")

    os.makedirs("logs/training_runs", exist_ok=True)
    torch.save(nalu.state_dict(), "logs/training_runs/stage4_nalu_align.pt")
    print("[Stage 4] Checkpoint saved to logs/training_runs/stage4_nalu_align.pt")


if __name__ == "__main__":
    train_neuro_symbolic_alignment()
