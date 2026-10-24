"""Unit & System Tests for Project Jane Phase 2: Building the World (Manifold Formation)."""

import os
import json
import pytest
import torch

from src.translator.pretrained_translator import PretrainedLinguisticEar, PretrainedLinguisticMouth
from src.reasoning_engine.manifold import RelationalManifold
from src.reasoning_engine.energy_constraints import EnergyConstraintLandscape, DualHeadConceptBottleneck
from src.reasoning_engine.nalu_arithmetic import NALUCluster
from src.reasoning_engine.dnc_memory import DNCMemoryTape
from src.symbolic_verifier.contracts import PhysicsHypothesis, MathHypothesis, PhysicalQuantity, Dimension
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.core.pipeline import JanePipeline


def test_knowledge_graph_data_integrity():
    kg_dir = "data/knowledge_graph"

    # 1. Concept Triples
    with open(os.path.join(kg_dir, "concept_triples.json"), "r") as f:
        triples = json.load(f)
    assert len(triples) >= 30
    for t in triples:
        assert "head" in t and "relation" in t and "tail" in t
        assert len(t["head"]) > 0 and len(t["relation"]) > 0 and len(t["tail"]) > 0

    # 2. QUDT Units
    with open(os.path.join(kg_dir, "qudt_dimensional_units.json"), "r") as f:
        qudt = json.load(f)
    assert len(qudt) >= 25
    valid_dims = {"MASS", "LENGTH", "TIME", "CURRENT", "TEMPERATURE", "VELOCITY", "ENERGY", "DIMENSIONLESS"}
    for u in qudt:
        assert u["dimension"] in valid_dims
        assert len(u["vector"]) == 8
        assert sum(u["vector"]) == 1

    # 3. Physics & Engineering Rules
    with open(os.path.join(kg_dir, "physics_engineering_rules.json"), "r") as f:
        rules = json.load(f)
    assert len(rules) >= 5
    rule_ids = {r["id"] for r in rules}
    assert "phys_speed_of_light" in rule_ids
    assert "phys_energy_conservation" in rule_ids
    assert "phys_mass_positivity" in rule_ids

    # 4. Formal Math & Logic
    with open(os.path.join(kg_dir, "formal_math_logic.json"), "r") as f:
        math_data = json.load(f)
    assert len(math_data) >= 5


def test_pretrained_linguistic_ear_and_mouth():
    ear = PretrainedLinguisticEar(latent_dim=256, device="cpu")
    mouth = PretrainedLinguisticMouth()

    # 1. Ear encodes text into 256D latent space
    z = ear("thruster fuel valve")
    assert z.shape == (1, 256)
    assert torch.isfinite(z).all()

    # Batch encoding
    z_batch = ear(["mass", "acceleration", "force"])
    assert z_batch.shape == (3, 256)

    # 2. Mouth generates natural verbalization
    response = mouth.verbalize(
        user_prompt="Calibrate reactor output to 50%",
        reasoned_concepts=["reactor", "power_output"],
        proof_log="Formally proven by Z3 SMT solver.",
        is_valid=True
    )
    assert isinstance(response, str)
    assert len(response) > 10


def test_qudt_concept_bottleneck_dimension_classification():
    bottleneck = DualHeadConceptBottleneck(latent_dim=256, num_channels=6)
    z = torch.randn(2, 256)

    mags, dim_probs = bottleneck(z)
    assert mags.shape == (2, 6)
    assert dim_probs.shape == (2, 6, 8)

    # Probabilities must sum to 1 over the 8 dimensions
    prob_sums = dim_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_energy_brick_wall_barrier_separation():
    energy_model = EnergyConstraintLandscape(latent_dim=256, barrier_weight=100.0)

    # Feasible state
    z_feas = torch.randn(4, 256) * 0.1
    # Violating state
    z_viol = torch.randn(4, 256) * 2.0

    e_feas = energy_model(z_feas)
    assert torch.isfinite(e_feas).all()

    # Ranking loss calculation
    loss = energy_model.ranking_loss(z_feas, z_viol, margin=5.0)
    assert loss.item() >= 0.0


def test_neuro_symbolic_z3_verification_feedback():
    verifier = Z3LogicVerifier()

    # 1. Valid state: E_in >= E_out
    hyp_valid = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=100.0, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
        PhysicalQuantity(magnitude=80.0, dimension=Dimension.ENERGY, confidence=0.99, is_output=True)
    ])
    res_valid = verifier.verify_physics(hyp_valid)
    assert res_valid.is_valid is True

    # 2. Violating state: E_out > E_in (Thermodynamic violation)
    hyp_invalid = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=100.0, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
        PhysicalQuantity(magnitude=150.0, dimension=Dimension.ENERGY, confidence=0.99, is_output=True)
    ])
    res_invalid = verifier.verify_physics(hyp_invalid)
    assert res_invalid.is_valid is False
    assert "Thermodynamic violation" in res_invalid.violation_details


def test_full_phase2_pipeline_with_pretrained_translator():
    pipeline = JanePipeline(
        latent_dim=256,
        max_reasoning_steps=5,
        use_pretrained_translator=True,
        device="cpu"
    )

    hyp = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=50.0, dimension=Dimension.MASS, confidence=0.99),
        PhysicalQuantity(magnitude=200.0, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
        PhysicalQuantity(magnitude=160.0, dimension=Dimension.ENERGY, confidence=0.99, is_output=True)
    ])

    result = pipeline.process(
        text="Adjust reactor fuel to maintain 160 MW power output",
        hypothesis=hyp
    )

    assert "response_text" in result
    assert len(result["response_text"]) > 0
    assert result["verification"].is_valid is True
    assert "dots_3d" in result
    assert len(result["dots_3d"][0]) == 32  # 32 manifold dots in 3D
