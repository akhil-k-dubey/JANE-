"""Unit & Integration Tests for Active Learning Interface (Teaching Jane Natively)."""

import pytest
import torch
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.self_model import SelfModelEngine


def test_active_learner_learn_valid_concept(tmp_path):
    manifold_path = str(tmp_path / "test_manifold.pt")
    self_model_path = str(tmp_path / "self_model.json")
    triples_path = str(tmp_path / "concept_triples.json")

    # Create dummy initial files
    with open(self_model_path, "w") as f:
        f.write('{"knowledge_base": {"core_concepts": 36, "knowledge_graph_entries": 101, "domains": ["Physics"]}}')
    with open(triples_path, "w") as f:
        f.write('[]')

    manifold = HebbianConceptManifold(latent_dim=256, save_path=manifold_path)
    verifier = Z3LogicVerifier()
    self_model = SelfModelEngine(model_path=self_model_path)

    learner = ActiveLearner(
        manifold=manifold,
        z3_verifier=verifier,
        hebbian_plasticity=manifold,
        self_model=self_model,
        triples_path=triples_path
    )

    initial_count = manifold.num_concepts
    res = learner.learn_new_concept(
        concept_name="superconductivity",
        related_concepts=["temperature", "resistance"],
        definition_ast="superconductivity(temperature, resistance) -> zero_resistance_state"
    )

    # 1. Output verification
    assert "Understood" in res
    assert "superconductivity" in res

    # 2. Node created in 256D space
    assert "superconductivity" in manifold.name_to_idx
    assert manifold.num_concepts >= initial_count + 1
    sc_idx = manifold.name_to_idx["superconductivity"]
    assert manifold.concept_nodes.shape[0] == manifold.num_concepts

    # 3. Permanent Hebbian wiring
    assert "temperature" in manifold.name_to_idx
    t_idx = manifold.name_to_idx["temperature"]
    assert manifold.adjacency[sc_idx, t_idx].item() >= 0.40
    assert manifold.adjacency[t_idx, sc_idx].item() >= 0.40


def test_active_learner_rejects_physical_violation(tmp_path):
    manifold = HebbianConceptManifold(latent_dim=256, save_path=str(tmp_path / "manifold.pt"))
    verifier = Z3LogicVerifier()
    learner = ActiveLearner(manifold=manifold, z3_verifier=verifier)

    initial_count = manifold.num_concepts
    res = learner.learn_new_concept(
        concept_name="perpetual_energy_source",
        related_concepts=["energy", "power"],
        definition_ast="energy_out > energy_in"
    )

    # Must reject the violation
    assert "I cannot learn this, sir" in res
    assert "violates existing mathematical axioms" in res.lower()
    # Must NOT add invalid node
    assert manifold.num_concepts == initial_count
    assert "perpetual_energy_source" not in manifold.name_to_idx


def test_parse_teaching_utterance():
    learner = ActiveLearner()

    utterance = "Jane, a 'quantum entanglement' is when two particles share a wave function, so measuring one instantly affects the other."
    parsed = learner.parse_teaching_utterance(utterance)

    assert parsed is not None
    assert parsed["concept_name"] == "quantum_entanglement"
    assert any("particle" in r for r in parsed["related_concepts"])
    assert "quantum_entanglement" in parsed["definition_ast"]

    # Also test unquoted format
    utterance_unquoted = "Jane, dark_matter is when matter does not interact with electromagnetic radiation."
    parsed_unquoted = learner.parse_teaching_utterance(utterance_unquoted)
    assert parsed_unquoted is not None
    assert parsed_unquoted["concept_name"] == "dark_matter"


def test_active_learner_dynamic_fallback_related_concepts(tmp_path):
    manifold = HebbianConceptManifold(latent_dim=256, save_path=str(tmp_path / "manifold.pt"))
    verifier = Z3LogicVerifier()
    learner = ActiveLearner(
        manifold=manifold,
        z3_verifier=verifier,
        triples_path=str(tmp_path / "triples.json")
    )

    # Learn a valid concept with NO explicit related_concepts passed
    res = learner.learn_new_concept(
        concept_name="gravitational_potential",
        related_concepts=[],
        definition_ast="gravitational_potential(gravity, mass) -> energy_state"
    )

    assert "Understood" in res
    assert "gravitational_potential" in manifold.name_to_idx
    gp_idx = manifold.name_to_idx["gravitational_potential"]

    # Must be wired to at least one concept via Hebbian plasticity
    connected_edges = (manifold.adjacency[gp_idx] > 0.1).nonzero().squeeze().tolist()
    if isinstance(connected_edges, int):
        connected_edges = [connected_edges]
    assert len(connected_edges) >= 2  # self-loop + at least 1 neighbor
