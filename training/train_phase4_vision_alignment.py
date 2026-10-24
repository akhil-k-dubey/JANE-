"""Phase 4: Multimodal Vision Alignment Training (Sight -> 256D Manifold).

Wired directly into Jane's 256D Relational Manifold:
- Vision Encoder: VisionPatchTransformer maps visual image crops to 256D
- Text Reference: PretrainedLinguisticEar provides target text concept vectors
- Objective: Contrastive Cosine Alignment so visual objects ('reactor_key', 'reactor_core')
  fire the exact same 256D manifold dots as their text equivalents.
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Tuple

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.spatial_graph.environment import SpatialEnvironmentGraph
from environments.spatial_graph.visual_renderer import FacilityVisualRenderer
from src.perception.vision_encoder import VisionPatchTransformer
from src.translator.pretrained_translator import PretrainedLinguisticEar
from src.reasoning_engine.manifold import RelationalManifold


def train_vision_manifold_alignment(
    epochs: int = 25,
    lr: float = 1e-3,
    device_str: str = "cuda",
    checkpoint_phase3: str = "logs/training_runs/phase3_world_manifold_rl.pt"
) -> float:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("PROJECT JANE — PHASE 4: MULTIMODAL VISION ALIGNMENT")
    print("Aligning Visual Transformer Patches with 256D Manifold Knowledge Dots")
    print(f"Device: {device}")
    print("=" * 70)

    # 1. Models
    vision_encoder = VisionPatchTransformer(
        img_size=128,
        patch_size=16,
        vision_dim=192,
        latent_dim=256
    ).to(device)

    ear = PretrainedLinguisticEar(latent_dim=256, device=device_str)
    manifold = RelationalManifold(latent_dim=256).to(device)
    renderer = FacilityVisualRenderer(img_size=128)

    # Load Phase 3 manifold checkpoint if available
    if os.path.exists(checkpoint_phase3):
        ckpt = torch.load(checkpoint_phase3, map_location=device)
        if "manifold" in ckpt:
            manifold.load_state_dict(ckpt["manifold"])
        if "ear_projection" in ckpt:
            ear.projection.load_state_dict(ckpt["ear_projection"])
        print(f"[Phase 4 Vision] Loaded base manifold checkpoint: {checkpoint_phase3}")

    manifold.eval()

    # 2. Concept Catalog: Multimodal Concept Pairs (Text Concept -> Visual Target Region)
    concept_targets = [
        ("reactor_key", "lab", (0.0, 0.0)),
        ("reactor_core", "reactor_core", (10.0, 5.0)),
        ("reactor_airlock", "reactor_airlock", (5.0, 5.0)),
        ("lab", "lab", (0.0, 0.0)),
        ("hallway", "hallway", (5.0, 0.0)),
        ("central_pillar", "central_pillar", (2.75, 3.0))
    ]

    # Precompute 256D text anchor vectors
    text_anchors: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, _, _ in concept_targets:
            z_text = ear(name)
            text_anchors[name] = z_text.squeeze(0)  # [256]

    optimizer = AdamW(vision_encoder.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = 0.0

    env = SpatialEnvironmentGraph.create_facility_env()

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(concept_targets)

        for concept_name, room_id, center_coord in concept_targets:
            optimizer.zero_grad()

            # Render live environment
            img = renderer.render_environment(env).unsqueeze(0).to(device)  # [1, 3, 128, 128]

            # Add gentle data augmentation (subtle jitter / noise)
            noise = torch.randn_like(img) * 0.01
            img_aug = (img + noise).clamp(0.0, 1.0)

            # Vision Transformer encodes image directly into 256D
            z_visual = vision_encoder(img_aug)  # [1, 256]

            # Positive text target
            z_text_pos = text_anchors[concept_name].unsqueeze(0)  # [1, 256]

            # Contrastive cosine loss with negative concepts
            neg_concepts = [c for c, _, _ in concept_targets if c != concept_name]
            neg_name = random.choice(neg_concepts)
            z_text_neg = text_anchors[neg_name].unsqueeze(0)

            cos_pos = F.cosine_similarity(z_visual, z_text_pos, dim=-1)
            cos_neg = F.cosine_similarity(z_visual, z_text_neg, dim=-1)

            # Margin triplet loss: pull positive text near, push negative text away
            margin = 0.4
            loss = (1.0 - cos_pos) + F.relu(cos_neg - cos_pos + margin)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(concept_targets)
        final_loss = avg_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Multimodal Vision Loss: {avg_loss:.4f} | Cos(Pos): {cos_pos.item():.3f} | Cos(Neg): {cos_neg.item():.3f}")

    # Save Phase 4 Vision Checkpoint
    os.makedirs("logs/training_runs", exist_ok=True)
    ckpt_path = "logs/training_runs/phase4_vision_manifold.pt"
    torch.save({
        "vision_encoder": vision_encoder.state_dict(),
        "manifold": manifold.state_dict(),
        "ear_projection": ear.projection.state_dict()
    }, ckpt_path)
    print(f"\n[Phase 4 Vision Complete] Checkpoint saved successfully to {ckpt_path}")
    return final_loss


if __name__ == "__main__":
    train_vision_manifold_alignment(epochs=20)
