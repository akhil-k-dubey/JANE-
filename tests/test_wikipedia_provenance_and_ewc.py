"""Unit and Integration Tests for:
1. Wikipedia Scientific Ingestion with Provenance (WikipediaIngestor)
2. Episodic Memory Ledger & Dynamic Provenance Breakdown (EpisodicLedger)
3. Elastic Weight Consolidation (ElasticWeightConsolidator)
4. Sub-linear Inverted Index Retrieval in ManifoldRetriever
5. Dynamic Episodic Reflection in PretrainedLinguisticMouth (No Canned String Matching)
"""

import os
import pytest
import torch
import torch.nn as nn

from src.core.wikipedia_ingestor import WikipediaIngestor, WIKIPEDIA_FOUNDATIONAL_ARTICLES
from src.reasoning_engine.episodic_ledger import EpisodicLedger, EpisodicEvent
from src.reasoning_engine.ewc_consolidation import ElasticWeightConsolidator
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.manifold_retriever import ManifoldRetriever
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic
from src.translator.pretrained_translator import PretrainedLinguisticMouth
from src.core.pipeline import JanePipeline


def test_episodic_ledger_recording_and_stats(tmp_path):
    """Verifies that EpisodicLedger records events with source provenance and computes statistics."""
    log_file = str(tmp_path / "test_ledger.jsonl")
    ledger = EpisodicLedger(log_path=log_file)

    # 1. Record an arXiv ingestion
    ledger.record_event(
        action_type="INGEST_ARXIV",
        concept_name="josephson_supercurrent",
        source_name="arXiv:quant-ph/0306072",
        formula_or_axiom="current == critical_current * sin(phase)",
        status="COMMITTED"
    )

    # 2. Record a Wikipedia ingestion
    ledger.record_event(
        action_type="INGEST_WIKIPEDIA",
        concept_name="bekenstein_hawking_entropy",
        source_name="Wikipedia:Black_hole_thermodynamics",
        formula_or_axiom="entropy == (k_b * area) / (4 * planck_length**2)",
        status="COMMITTED"
    )

    # 3. Record an autonomous curiosity loop discovery
    ledger.record_event(
        action_type="DISCOVERY",
        concept_name="force_work_bridge",
        source_name="CuriosityLoop",
        formula_or_axiom="work == force * displacement",
        status="COMMITTED"
    )

    # 4. Record user teaching
    ledger.record_event(
        action_type="TEACH_USER",
        concept_name="warp_drive_metric",
        source_name="Akhil",
        formula_or_axiom="alcubierre_tensor == energy_momentum",
        status="COMMITTED"
    )

    # Check stats breakdown
    stats = ledger.get_provenance_stats()
    assert stats["arxiv"] == 1
    assert stats["wikipedia"] == 1
    assert stats["curiosity_loop"] == 1
    assert stats["user_teaching"] == 1

    # Check formatted context
    fmt = ledger.format_recent_ledger_for_context(limit=4)
    assert "arXiv:quant-ph/0306072" in fmt
    assert "Wikipedia:Black_hole_thermodynamics" in fmt
    assert "CuriosityLoop" in fmt
    assert "force_work_bridge" in fmt


def test_wikipedia_ingestor_curated_and_z3_verification(tmp_path):
    """Verifies that WikipediaIngestor extracts concepts, proves with Z3, and logs provenance."""
    log_file = str(tmp_path / "wiki_test_ledger.jsonl")
    ledger = EpisodicLedger(log_path=log_file)
    manifold = HebbianConceptManifold(latent_dim=256)
    z3_verifier = Z3LogicVerifier()
    self_model = SelfModelEngine()
    learner = ActiveLearner(manifold=manifold, z3_verifier=z3_verifier, self_model=self_model)

    ingestor = WikipediaIngestor(
        active_learner=learner,
        manifold=manifold,
        z3_verifier=z3_verifier,
        episodic_ledger=ledger
    )

    report = ingestor.ingest_article("Superconductivity")
    assert report.articles_processed == 1
    assert len(report.accepted_concepts) > 0
    assert "meissner_effect" in report.accepted_concepts or "london_penetration_depth" in report.accepted_concepts

    # Check that provenance was committed to episodic ledger
    recent = ledger.get_recent_events(limit=3)
    assert any("Wikipedia:Superconductivity" in e.source_name for e in recent)
    assert any(e.status == "COMMITTED" for e in recent)


def test_elastic_weight_consolidation_penalty_and_fisher():
    """Verifies that EWC penalizes weight drift and protects foundational weights."""
    # Simple toy linear network representing foundational arithmetic/fusion weights
    net = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    ewc = ElasticWeightConsolidator(model=net, ewc_lambda=100.0)

    # Initial penalty should be exactly 0.0 because parameters have not drifted
    initial_penalty = ewc.compute_ewc_penalty().item()
    assert initial_penalty == pytest.approx(0.0, abs=1e-6)

    # Simulate catastrophic forgetting / drift by perturbing weights
    with torch.no_grad():
        for param in net.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    # EWC penalty must now be strictly positive, reflecting resistance to parameter drift
    drift_penalty = ewc.compute_ewc_penalty().item()
    assert drift_penalty > 0.01

    # Verify Fisher estimation
    dummy_inputs = [torch.randn(2, 16) for _ in range(3)]
    loss_fn = lambda out: out.sum()
    ewc.estimate_fisher(dummy_inputs, loss_fn)
    for name, f_diag in ewc.fisher.items():
        assert (f_diag > 0).all()


def test_sublinear_inverted_index_manifold_retriever():
    """Verifies that ManifoldRetriever builds inverted indices and retrieves in O(1)."""
    retriever = ManifoldRetriever(
        concept_names=["quantum_entanglement", "general_relativity", "kinetic_energy"]
    )

    # Verify indices exist
    assert hasattr(retriever, "_concepts_index")
    assert hasattr(retriever, "_triples_index")
    assert hasattr(retriever, "_principles_index")

    # Inverted index lookup for 'quantum'
    ctx = retriever.retrieve("tell me about quantum entanglement")
    assert "quantum_entanglement" in ctx.matched_concepts
    assert ctx.is_known is True

    # Retrieve physical principle
    ctx_energy = retriever.retrieve("what is kinetic energy")
    assert any("kinetic_energy" in c for c in ctx_energy.matched_concepts)


def test_dynamic_episodic_reflection_no_canned_deflection(tmp_path):
    """Verifies that when asked 'have you added this discovered concept', Jane reflects her live ledger."""
    log_file = str(tmp_path / "reflection_ledger.jsonl")
    ledger = EpisodicLedger(log_path=log_file)

    # Simulate discovery event
    ledger.record_event(
        action_type="DISCOVERY",
        concept_name="force_work_bridge",
        source_name="CuriosityLoop",
        formula_or_axiom="work == force * displacement",
        status="COMMITTED",
        details={"node_a": "force", "node_b": "work"}
    )

    self_model = SelfModelEngine()
    retriever = ManifoldRetriever(concept_names=["force_work_bridge", "force", "work"])
    critic = AntiHallucinationCritic(self_model=self_model)

    mouth = PretrainedLinguisticMouth(
        self_model=self_model,
        retriever=retriever,
        critic=critic,
        episodic_ledger=ledger
    )

    # Ask the exact question that previously produced the deflection
    reply = mouth.verbalize(
        user_prompt="have you added this discovered concept in your knowledge graph",
        reasoned_concepts=["added", "discovered", "concept"],
        proof_log=None,
        is_valid=True
    )

    # Assert it reflects the actual discovery rather than robotic deflection
    assert "force_work_bridge" in reply
    assert "CuriosityLoop" in reply
    assert "work == force * displacement" in reply
    assert "I am analyzing the relationships surrounding" not in reply


def test_end_to_end_wikipedia_pipeline_integration():
    """Verifies that JanePipeline.search_and_ingest_wikipedia wires concepts and updates provenance."""
    pipeline = JanePipeline(use_pretrained_translator=True)
    initial_count = pipeline.hebbian_manifold.num_concepts

    report = pipeline.search_and_ingest_wikipedia("Quantum_entanglement")
    assert report.articles_processed == 1
    assert len(report.accepted_concepts) > 0
    assert pipeline.hebbian_manifold.num_concepts > initial_count

    # Check that episodic ledger recorded the event
    stats = pipeline.episodic_ledger.get_provenance_stats()
    assert stats["wikipedia"] >= 1
