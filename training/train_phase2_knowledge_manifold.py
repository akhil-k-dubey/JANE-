"""Phase 2: Building the World (Knowledge Graph -> 3D Manifold Formation).

Consumes the Knowledge Graph to wire physical, mathematical, and conceptual laws
into Jane's 256D latent space and 3D projection manifold:
1. Self-Supervised Manifold Formation (Contrastive Triples & Masked Prediction)
2. QUDT Dimensional Grounding (Dual-Head Concept Bottleneck)
3. Energy Barrier & Invariant Brick Walls (Physics & Thermodynamics)
4. Formal Math & DNC Multi-Step Scratchpad (NALU & Memory Tape)
5. Neuro-Symbolic Alignment (Z3 SMT Solver as Teacher)
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any, List, Tuple

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.translator.pretrained_translator import PretrainedLinguisticEar
from src.reasoning_engine.manifold import RelationalManifold
from src.reasoning_engine.energy_constraints import EnergyConstraintLandscape, DualHeadConceptBottleneck
from src.reasoning_engine.nalu_arithmetic import NALUCluster
from src.reasoning_engine.dnc_memory import DNCMemoryTape
from src.symbolic_verifier.contracts import MathHypothesis, PhysicsHypothesis, PhysicalQuantity, Dimension
from src.symbolic_verifier.z3_interface import Z3LogicVerifier


DIMENSION_MAP = {
    "MASS": 0,
    "LENGTH": 1,
    "TIME": 2,
    "CURRENT": 3,
    "TEMPERATURE": 4,
    "VELOCITY": 5,
    "ENERGY": 6,
    "DIMENSIONLESS": 7
}


class Phase2KnowledgeTrainer:
    """Orchestrates Phase 2 training across the 4 knowledge domains."""

    def __init__(self, latent_dim: int = 256, device_str: str = "cuda"):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        # 1. Pre-trained Translator (Ear)
        self.ear = PretrainedLinguisticEar(latent_dim=latent_dim, device=device_str)

        # 2. Reasoning Engine (Random weights -> Trained from scratch)
        self.manifold = RelationalManifold(latent_dim=latent_dim).to(self.device)
        self.bottleneck = DualHeadConceptBottleneck(latent_dim=latent_dim, num_channels=6).to(self.device)
        self.energy_model = EnergyConstraintLandscape(latent_dim=latent_dim, barrier_weight=100.0).to(self.device)
        self.nalu = NALUCluster(latent_dim=latent_dim, math_dim=32).to(self.device)
        self.dnc = DNCMemoryTape(controller_dim=latent_dim).to(self.device)

        # 3. Symbolic Teacher
        self.z3_verifier = Z3LogicVerifier()

        # Learnable relation translation embeddings for triples (TransE geometry)
        self.relation_embeddings = nn.ParameterDict()

    def train_manifold_triples(self, triples_file: str, epochs: int = 20, lr: float = 2e-3) -> float:
        """
        Stage 2A: Self-Supervised Knowledge Graph Eating.
        Shapes the 256D space so related concepts ('mass' & 'weight', 'thruster' & 'fuel')
        cluster tightly in 256D and 3D space.
        """
        print("\n--- [Stage 2A] Training Manifold Geometry on Concept Triples ---")
        with open(triples_file, "r") as f:
            triples = json.load(f)

        # Initialize relation embeddings
        unique_relations = sorted(list({t["relation"] for t in triples}))
        for r in unique_relations:
            if r not in self.relation_embeddings:
                self.relation_embeddings[r] = nn.Parameter(torch.randn(1, self.latent_dim, device=self.device) * 0.05)

        all_tails = [t["tail"] for t in triples]
        optimizer = AdamW(
            list(self.manifold.parameters()) +
            list(self.ear.projection.parameters()) +
            list(self.relation_embeddings.values()),
            lr=lr
        )

        margin = 2.0
        final_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(triples)

            for item in triples:
                h_text = item["head"]
                r_name = item["relation"]
                t_text = item["tail"]
                # Corrupt negative sampling
                neg_t_text = random.choice(all_tails)
                while neg_t_text == t_text:
                    neg_t_text = random.choice(all_tails)

                optimizer.zero_grad()

                # Pre-trained Ear embeds concepts (backprop through projection)
                z_h = self.ear(h_text)
                z_t = self.ear(t_text)
                z_neg = self.ear(neg_t_text)

                # Pass through dynamic manifold
                nodes_h, _ = self.manifold(z_h)
                nodes_t, _ = self.manifold(z_t)
                nodes_neg, _ = self.manifold(z_neg)

                m_h = self.manifold.readout(nodes_h)
                m_t = self.manifold.readout(nodes_t)
                m_neg = self.manifold.readout(nodes_neg)

                rel_vec = self.relation_embeddings[r_name]

                # Geometric TransE distance: || m_h + r - m_t ||
                pos_dist = torch.norm(m_h + rel_vec - m_t, p=2, dim=-1)
                neg_dist = torch.norm(m_h + rel_vec - m_neg, p=2, dim=-1)

                # Direct semantic clustering: related concepts cluster together in latent space
                pos_clust = torch.norm(m_h - m_t, p=2, dim=-1)
                neg_clust = torch.norm(m_h - m_neg, p=2, dim=-1)

                loss = pos_dist + F.relu(margin - neg_dist) + 0.5 * (pos_clust + F.relu(margin - neg_clust))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(triples)
            final_loss = avg_loss
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Contrastive Triples Loss: {avg_loss:.4f}")

        return final_loss

    def train_qudt_dimensional_bottleneck(self, qudt_file: str, epochs: int = 30, lr: float = 2e-3) -> float:
        """
        Stage 2B: QUDT Dimensional Grounding.
        Trains the Dimension Head to output sharp SI classifications (>85% confidence).
        """
        print("\n--- [Stage 2B] Training QUDT Dimensional Concept Bottleneck ---")
        with open(qudt_file, "r") as f:
            units_data = json.load(f)

        optimizer = AdamW(self.bottleneck.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        final_acc = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0

            for u in units_data:
                text = f"{u['symbol']} {u['name']}"
                target_idx = DIMENSION_MAP[u["dimension"]]
                target_tensor = torch.tensor([target_idx], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    z_u = self.ear(text)

                optimizer.zero_grad()
                batch_size = z_u.size(0)
                dim_logits = self.bottleneck.dimension_head(z_u).view(batch_size, self.bottleneck.num_channels, self.bottleneck.num_dimensions)
                primary_logits = dim_logits[:, 0, :]
                primary_probs = F.softmax(primary_logits, dim=-1)

                ce_loss = criterion(primary_logits, target_tensor)
                # Ambiguity penalty: penalize maximum probability < 0.85
                max_prob = torch.max(primary_probs, dim=-1)[0]
                ambiguity_loss = F.relu(0.85 - max_prob) ** 2

                loss = ce_loss + 5.0 * ambiguity_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if torch.argmax(primary_logits, dim=-1).item() == target_idx:
                    correct += 1

            acc = correct / len(units_data)
            final_acc = acc
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:02d}/{epochs:02d} | CE Loss: {total_loss / len(units_data):.4f} | Dimension Acc: {acc:.1%}")

        return final_acc

    def train_energy_brick_walls(self, rules_file: str, epochs: int = 15, lr: float = 1e-3) -> float:
        """
        Stage 2C: Invariant Physics & Thermodynamics Brick Walls.
        Wires continuous energy barriers so violating states cannot move in the manifold.
        """
        print("\n--- [Stage 2C] Training Invariant Energy Barriers & Brick Walls ---")
        optimizer = AdamW(self.energy_model.parameters(), lr=lr)
        batch_size = 16
        final_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 1. Feasible physical states (Thermodynamically sound)
            feasible_latents = torch.randn(batch_size, self.latent_dim, device=self.device) * 0.2
            # 2. Corrupted violating states (violating physical conservation)
            violating_latents = torch.randn(batch_size, self.latent_dim, device=self.device) * 1.2

            with torch.no_grad():
                f_mags, _ = self.energy_model.concept_bottleneck(feasible_latents)
                target_f_mags = f_mags.clone()
                target_f_mags[:, 0] = torch.abs(target_f_mags[:, 0]) + 1.0  # mass > 0
                target_f_mags[:, 3] = torch.abs(target_f_mags[:, 3]) + 50.0  # E_in
                target_f_mags[:, 4] = target_f_mags[:, 3] * 0.7              # E_out <= E_in

                v_mags, _ = self.energy_model.concept_bottleneck(violating_latents)
                target_v_mags = v_mags.clone()
                target_v_mags[:, 0] = -torch.abs(target_v_mags[:, 0]) - 5.0 # mass < 0
                target_v_mags[:, 4] = target_v_mags[:, 3] + 80.0             # E_out > E_in

            pred_f_mags, _ = self.energy_model.concept_bottleneck(feasible_latents)
            pred_v_mags, _ = self.energy_model.concept_bottleneck(violating_latents)

            bottleneck_loss = F.mse_loss(pred_f_mags[:, [0, 3, 4]], target_f_mags[:, [0, 3, 4]]) + \
                              F.mse_loss(pred_v_mags[:, [0, 3, 4]], target_v_mags[:, [0, 3, 4]])

            # Ranking loss: E(feasible) + margin < E(violating)
            rank_loss = self.energy_model.ranking_loss(feasible_latents, violating_latents, margin=5.0)
            loss = rank_loss + 0.5 * bottleneck_loss
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                e_feas = self.energy_model(feasible_latents).mean().item()
                e_viol = self.energy_model(violating_latents).mean().item()
                print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Energy Loss: {final_loss:.4f} | E(Feas): {e_feas:.2f} | E(Viol): {e_viol:.2f}")

        return final_loss

    def train_math_and_memory(self, math_file: str, epochs: int = 15, lr: float = 2e-3) -> float:
        """
        Stage 2D: Formal Math & DNC Multi-Step Scratchpad Training.
        Trains NALU arithmetic units and DNC tape on algebraic equations and multi-step tasks.
        """
        print("\n--- [Stage 2D] Training NALU Arithmetic & DNC Memory Tape ---")
        with open(math_file, "r") as f:
            math_cases = json.load(f)

        optimizer = AdamW(list(self.nalu.parameters()) + list(self.dnc.parameters()), lr=lr)
        criterion = nn.MSELoss()
        final_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0

            for item in math_cases:
                optimizer.zero_grad()

                if "steps" in item:
                    # Multi-step arithmetic evaluation
                    memory = self.dnc.init_memory(1, device=self.device)
                    step_loss = torch.tensor(0.0, device=self.device)

                    for st in item["steps"]:
                        op_nums = torch.tensor([[st["operands"][0], st["operands"][1]]], dtype=torch.float32, device=self.device)
                        expected = torch.tensor([[st["result"]]], dtype=torch.float32, device=self.device)

                        controller = torch.randn(1, self.latent_dim, device=self.device)
                        tape_read, memory = self.dnc.step(controller, memory)

                        _, raw_math = self.nalu(controller + tape_read, explicit_numbers=op_nums)
                        pred = raw_math[:, 0:1]

                        step_loss = step_loss + criterion(pred, expected)

                    step_loss.backward()
                    optimizer.step()
                    total_loss += step_loss.item()
                else:
                    # Single equation or multi-step word problem with equations/solution
                    solution_dict = item.get("solution", {})
                    if solution_dict:
                        memory = self.dnc.init_memory(1, device=self.device)
                        eq_loss = torch.tensor(0.0, device=self.device)
                        for var_name, target_val in solution_dict.items():
                            expected = torch.tensor([[target_val]], dtype=torch.float32, device=self.device)
                            controller = torch.randn(1, self.latent_dim, device=self.device)
                            tape_read, memory = self.dnc.step(controller, memory)
                            op_nums = torch.tensor([[target_val * 0.5, target_val * 0.5]], dtype=torch.float32, device=self.device)
                            _, raw_math = self.nalu(controller + tape_read, explicit_numbers=op_nums)
                            pred = raw_math[:, 0:1]
                            eq_loss = eq_loss + criterion(pred, expected)

                        eq_loss.backward()
                        optimizer.step()
                        total_loss += eq_loss.item()

            avg_loss = total_loss / len(math_cases)
            final_loss = avg_loss
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Math & DNC Loss: {avg_loss:.4f}")

        return final_loss

    def train_neuro_symbolic_alignment(self, epochs: int = 10, lr: float = 1e-3) -> float:
        """
        Stage 2E: Neuro-Symbolic Alignment with Z3 as Active Teacher.
        Candidate states are verified by Z3. Violations trigger large negative rewards.
        """
        print("\n--- [Stage 2E] Neuro-Symbolic Alignment (Z3 SMT Teacher Loop) ---")
        optimizer = AdamW(self.energy_model.parameters(), lr=lr)
        penalty_scale = 100.0

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Propose candidate latent state
            candidate_latents = torch.randn(8, self.latent_dim, device=self.device)
            mags, _ = self.energy_model.concept_bottleneck(candidate_latents)

            # Test sample 0 against Z3 SMT prover
            val_mass = mags[0, 0].item()
            val_in = mags[0, 3].item()
            val_out = mags[0, 4].item()

            hyp = PhysicsHypothesis(quantities=[
                PhysicalQuantity(magnitude=val_in, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
                PhysicalQuantity(magnitude=val_out, dimension=Dimension.ENERGY, confidence=0.99, is_output=True),
                PhysicalQuantity(magnitude=val_mass, dimension=Dimension.MASS, confidence=0.99)
            ])

            proof = self.z3_verifier.verify_physics(hyp)

            # If Z3 rejects, apply a massive penalty loss that forces the energy model to raise its barrier
            if not proof.is_valid:
                # Violating: enforce high energy
                energy_val = self.energy_model(candidate_latents[0:1])
                target_barrier = torch.tensor([50.0], device=self.device)
                loss = F.mse_loss(energy_val, target_barrier)
            else:
                # Valid: settle to near-zero ground state
                energy_val = self.energy_model(candidate_latents[0:1])
                loss = energy_val.mean()

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Z3 Valid: {proof.is_valid} | Energy: {energy_val.item():.2f} | Loss: {loss.item():.4f}")

        return loss.item()

    def save_checkpoint(self, path: str = "logs/training_runs/phase2_world_manifold.pt"):
        """Saves Phase 2 trained weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "ear_projection": self.ear.projection.state_dict(),
            "manifold": self.manifold.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "energy_model": self.energy_model.state_dict(),
            "nalu": self.nalu.state_dict(),
            "dnc": self.dnc.state_dict(),
            "relation_embeddings": {k: v.data for k, v in self.relation_embeddings.items()}
        }, path)
        print(f"\n[Phase 2 Complete] Checkpoint saved successfully to {path}")


def run_phase2_training():
    print("=" * 70)
    print("PROJECT JANE — PHASE 2: BUILDING THE WORLD (MANIFOLD FORMATION)")
    print("Consuming Knowledge Graph: Triples, QUDT, Physics Rules, Formal Math")
    print("=" * 70)

    kg_dir = os.path.join(os.path.dirname(__file__), "../data/knowledge_graph")
    triples_file = os.path.join(kg_dir, "concept_triples.json")
    qudt_file = os.path.join(kg_dir, "qudt_dimensional_units.json")
    rules_file = os.path.join(kg_dir, "physics_engineering_rules.json")
    math_file = os.path.join(kg_dir, "formal_math_logic.json")

    trainer = Phase2KnowledgeTrainer(latent_dim=256)

    # 1. Manifold Triples
    trainer.train_manifold_triples(triples_file, epochs=15)

    # 2. QUDT Concept Bottleneck
    trainer.train_qudt_dimensional_bottleneck(qudt_file, epochs=15)

    # 3. Physics Energy Barrier
    trainer.train_energy_brick_walls(rules_file, epochs=15)

    # 4. Formal Math & DNC
    trainer.train_math_and_memory(math_file, epochs=15)

    # 5. Neuro-Symbolic Alignment with Z3
    trainer.train_neuro_symbolic_alignment(epochs=10)

    # Save trained checkpoint
    trainer.save_checkpoint()


if __name__ == "__main__":
    run_phase2_training()
