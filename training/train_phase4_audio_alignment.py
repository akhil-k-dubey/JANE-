"""Phase 4: Direct Audio Alignment Training (Waveform -> 256D Manifold).

Trains DirectAudioTransformer to map raw acoustic waveforms directly
into Jane's 256D Relational Manifold:
- Zero Speech-to-Text: Bypasses text transcription completely.
- Intent Alignment: Aligns audio commands with canonical AST action vectors in 256D.
- Noise Gating: Ensures non-speech noise (coughing, ambient white noise) fails the
  confidence threshold and is safely dropped without hallucination.
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.audio_encoder import DirectAudioTransformer
from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator


def train_audio_manifold_alignment(
    epochs: int = 120,
    lr: float = 1.5e-3,
    device_str: str = "cuda"
) -> float:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("PROJECT JANE — PHASE 4: DIRECT AUDIO-TO-MANIFOLD ALIGNMENT")
    print("Training Direct Waveform-to-256D Vector Mapping (Zero STT / No Whisper)")
    print(f"Device: {device}")
    print("=" * 70)

    audio_transformer = DirectAudioTransformer(
        sample_rate=16000,
        conv_channels=128,
        audio_dim=192,
        latent_dim=256,
        checkpoint_path=None  # Start training from scratch
    ).to(device)

    generator = SyntheticAcousticGenerator(sample_rate=16000)
    optimizer = AdamW(audio_transformer.parameters(), lr=lr, weight_decay=1e-4)

    commands = ["EMERGENCY_HALT", "VENT_GAS", "CHECK_PRESSURE", "NAVIGATE_REACTOR"]
    intent_keys = list(audio_transformer.canonical_intents.keys())
    anchor_matrix = torch.stack([audio_transformer.canonical_intents[k] for k in intent_keys], dim=0)

    final_loss = 0.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for cmd in commands:
            # 1. Synthesize command audio
            waveform = generator.generate_command_audio(cmd, duration=1.0).unsqueeze(0).to(device)

            # 2. Forward pass through DirectAudioTransformer
            z_audio = audio_transformer(waveform)
            norm_z = F.normalize(z_audio, p=2, dim=-1)
            target_anchor = audio_transformer.canonical_intents[cmd].unsqueeze(0)

            # 3. Supervised InfoNCE Classification Loss
            logits = torch.matmul(norm_z, anchor_matrix.T) / 0.1
            target_idx = intent_keys.index(cmd)
            ce_loss = F.cross_entropy(logits, torch.tensor([target_idx], device=device))

            # 4. Direct Cosine Alignment Loss
            cos_loss = 1.0 - F.cosine_similarity(norm_z, target_anchor, dim=-1)
            total_loss += ce_loss + 2.0 * cos_loss

        # 5. Non-speech noise suppression: ensure noise has minimal similarity with any intent
        for noise_type in ["cough", "white", "hum"]:
            n_wave = generator.generate_noise_audio(noise_type, duration=1.0).unsqueeze(0).to(device)
            z_n = audio_transformer(n_wave)
            norm_zn = F.normalize(z_n, p=2, dim=-1)
            sims = torch.matmul(norm_zn, anchor_matrix.T)
            total_loss += 3.0 * F.relu(sims.max() - 0.1).pow(2)

        total_loss.backward()
        optimizer.step()

        avg_loss = total_loss.item() / (len(commands) + 3)
        final_loss = avg_loss

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs:03d} | Alignment Loss: {avg_loss:.4f}")

    # Save Phase 4 Audio Checkpoint
    os.makedirs("logs/training_runs", exist_ok=True)
    ckpt_path = "logs/training_runs/phase4_audio_manifold.pt"
    torch.save({
        "audio_transformer": audio_transformer.state_dict()
    }, ckpt_path)
    print(f"\n[Phase 4 Audio Complete] Checkpoint saved successfully to {ckpt_path}")
    return final_loss


if __name__ == "__main__":
    train_audio_manifold_alignment(epochs=120)
