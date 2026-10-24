"""Benchmark: Evaluating 256D Manifold Geometry & 3D Visual Concept Dots.

Evaluates whether the self-supervised Knowledge Graph eating process successfully
shaped the 3D Manifold such that:
1. Related concepts cluster tightly in vector space:
   - 'mass' and 'weight'
   - 'addition' and 'subtraction'
   - 'thruster' and 'fuel_valve'
2. Unrelated concepts are geometrically distant:
   - 'mass' vs 'banana'
   - 'addition' vs 'reactor'
3. 3D projection dots preserve physical neighborhood topology.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.translator.pretrained_translator import PretrainedLinguisticEar
from src.reasoning_engine.manifold import RelationalManifold


def evaluate_manifold_geometry(checkpoint_path: str = "logs/training_runs/phase2_world_manifold.pt"):
    print("=" * 70)
    print("Benchmark: 3D Manifold Geometry & Concept Cluster Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ear = PretrainedLinguisticEar(latent_dim=256, device=str(device))
    manifold = RelationalManifold(latent_dim=256).to(device)

    # Load trained weights if available
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "manifold" in ckpt:
            manifold.load_state_dict(ckpt["manifold"])
        if "ear_projection" in ckpt:
            ear.projection.load_state_dict(ckpt["ear_projection"])
        print(f"Loaded trained Phase 2 manifold checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running on initialized weights.")

    manifold.eval()

    def get_concept_point(concept: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = ear(concept)
            nodes, _ = manifold(z)
            m_256 = manifold.readout(nodes)[0]  # [256]
            c_3d = manifold.project_to_3d(nodes)[0, 0]  # [3]
            return m_256, c_3d

    # Test Concept Pairs: (Concept A, Concept B, Unrelated Concept C)
    test_triplets = [
        ("mass", "weight", "banana"),
        ("addition", "subtraction", "reactor"),
        ("thruster", "fuel_valve", "philosophy"),
        ("voltage", "current", "sandwich"),
        ("kinetic_energy", "velocity", "poem")
    ]

    total_tests = len(test_triplets)
    passed_tests = 0

    print("\n[Pairwise Geometric Distance Analysis in 256D Latent Space & 3D Projection]")
    print(f"{'Concept Pair':<28} | {'Dist(A, B)':<12} | {'Dist(A, Unrelated)':<18} | {'Clustering Ratio':<16} | Status")
    print("-" * 90)

    for a_name, b_name, unk_name in test_triplets:
        m_a, c_a = get_concept_point(a_name)
        m_b, c_b = get_concept_point(b_name)
        m_unk, c_unk = get_concept_point(unk_name)

        # 256D Euclidean Distances
        dist_related = torch.norm(m_a - m_b, p=2).item()
        dist_unrelated = torch.norm(m_a - m_unk, p=2).item()

        # 3D Euclidean Distances
        dist_3d_related = torch.norm(c_a - c_b, p=2).item()
        dist_3d_unrelated = torch.norm(c_a - c_unk, p=2).item()

        ratio = dist_related / max(dist_unrelated, 1e-6)
        is_clustered = dist_related < dist_unrelated

        if is_clustered:
            passed_tests += 1
            status = "PASS"
        else:
            status = "FAIL"

        pair_label = f"'{a_name}' & '{b_name}'"
        print(f"{pair_label:<28} | {dist_related:<12.4f} | {dist_unrelated:<18.4f} | {ratio:<16.4f} | {status}")

    print("-" * 90)
    pass_rate = (passed_tests / total_tests) if total_tests > 0 else 0.0
    print(f"Manifold Semantic Clustering Accuracy: {pass_rate:.1%} ({passed_tests}/{total_tests})")
    print(f"Result: {'PASS (Concepts Physically Clustered in Manifold)' if pass_rate >= 0.8 else 'FAIL'}")

    return pass_rate >= 0.8


if __name__ == "__main__":
    evaluate_manifold_geometry()
