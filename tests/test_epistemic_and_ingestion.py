"""Unit & Integration Tests for Epistemic Discovery Engine & Knowledge Ingestion:
1. Topological Void Detection in 256D Manifold
2. Karl Popper Falsification via Z3 SMT Solver (Sound vs. Contradictory Invariants)
3. Step Curiosity Cycle (Autonomous Discovery & Hebbian Plasticity)
4. Knowledge Ingestor Symbolic Quality Gate (Acceptance of Physics, Rejection of Violations)
5. Curated Seed Corpus Ingestion
6. Conversational Fallback Cleanup (Scrubbing of 'what, do, you' Stopword Artifacts)
"""

import pytest
import torch
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.nalu_arithmetic import NALUCluster
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.epistemic_engine import EpistemicEngine
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.core.knowledge_ingestor import KnowledgeIngestor
from src.translator.pretrained_translator import PretrainedLinguisticMouth
from src.reasoning_engine.manifold_retriever import ManifoldRetriever
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic


@pytest.fixture
def epistemic_stack():
    """Builds a fresh, isolated neuro-symbolic discovery stack."""
    manifold = HebbianConceptManifold(latent_dim=256)
    nalu = NALUCluster(latent_dim=256)
    z3_verifier = Z3LogicVerifier()
    self_model = SelfModelEngine()
    learner = ActiveLearner(
        manifold=manifold,
        z3_verifier=z3_verifier,
        self_model=self_model
    )
    engine = EpistemicEngine(
        manifold=manifold,
        nalu=nalu,
        z3_verifier=z3_verifier,
        active_learner=learner,
        self_model=self_model
    )
    ingestor = KnowledgeIngestor(
        active_learner=learner,
        manifold=manifold,
        z3_verifier=z3_verifier
    )
    return {
        "manifold": manifold,
        "nalu": nalu,
        "z3_verifier": z3_verifier,
        "self_model": self_model,
        "learner": learner,
        "engine": engine,
        "ingestor": ingestor
    }


def test_epistemic_void_detection(epistemic_stack):
    """Verifies that EpistemicEngine discovers unlinked conceptual voids in the manifold."""
    engine = epistemic_stack["engine"]
    gaps = engine.find_unlinked_clusters(threshold=0.35, top_k=5)

    assert len(gaps) > 0
    # Each gap is (node_a, node_b, priority)
    node_a, node_b, priority = gaps[0]
    assert isinstance(node_a, str)
    assert isinstance(node_b, str)
    assert node_a != node_b
    assert priority > 0.0


def test_popperian_falsification_sound_invariant(epistemic_stack):
    """Verifies that mathematically and physically sound formulas survive Popperian falsification."""
    engine = epistemic_stack["engine"]

    # Newton's second law
    is_falsified, log = engine.falsify_hypothesis("force == mass * acceleration")
    assert is_falsified is False
    assert "passed" in log.lower() or "sound" in log.lower() or "consistent" in log.lower()

    # Planck-Einstein relation
    is_falsified_qe, log_qe = engine.falsify_hypothesis("energy == planck_constant * frequency")
    assert is_falsified_qe is False

    # Power rate
    is_falsified_p, log_p = engine.falsify_hypothesis("power == work / delta_time")
    assert is_falsified_p is False


def test_popperian_falsification_rejects_violations(epistemic_stack):
    """Verifies that thermodynamic violations and arithmetic contradictions are actively disproven/falsified."""
    engine = epistemic_stack["engine"]

    # Arithmetic contradiction: 10 == 25
    is_falsified_math, log_math = engine.falsify_hypothesis("10 == 25")
    assert is_falsified_math is True
    assert "falsified" in log_math.lower()

    # Over-unity thermodynamic energy creation
    violation_formula = "energy_out == 500 and energy_in == 100 and energy_out > energy_in"
    is_falsified_phys, log_phys = engine.falsify_hypothesis(violation_formula)
    assert is_falsified_phys is True
    assert "falsified" in log_phys.lower()


def test_step_curiosity_cycle_discovery(epistemic_stack):
    """Verifies that an autonomous curiosity cycle discovers an invariant and wires it into the manifold."""
    engine = epistemic_stack["engine"]
    manifold = epistemic_stack["manifold"]

    initial_nodes = manifold.num_concepts
    result = engine.step_curiosity_cycle()

    assert result is not None
    assert result["status"] in ("DISCOVERED", "FALSIFIED")
    if result["status"] == "DISCOVERED":
        assert "concept_name" in result
        assert "formula" in result
        assert result["new_edge_weight"] > 0
        # Manifold should have gained a new node or reinforced edge
        assert manifold.num_concepts >= initial_nodes


def test_knowledge_ingestor_symbolic_quality_gate(epistemic_stack):
    """Verifies that KnowledgeIngestor permits valid science and blocks pseudoscientific axioms."""
    ingestor = epistemic_stack["ingestor"]

    # 1. Valid relativistic concept
    valid_spec = {
        "name": "relativistic_momentum",
        "related": ["momentum", "velocity"],
        "definition": "momentum == (mass * velocity) / sqrt(1 - (velocity * velocity) / (c * c))"
    }
    accepted, log_acc = ingestor.ingest_concept_spec(
        concept_name=valid_spec["name"],
        related_concepts=valid_spec["related"],
        definition_ast=valid_spec["definition"]
    )
    assert accepted is True
    assert "ACCEPTED" in log_acc

    # 2. Thermodynamic violation (free energy generation)
    invalid_spec = {
        "name": "perpetual_free_energy",
        "related": ["energy", "power"],
        "definition": "energy_output == 2000 and energy_input == 500 and energy_output > energy_input"
    }
    rejected, log_rej = ingestor.ingest_concept_spec(
        concept_name=invalid_spec["name"],
        related_concepts=invalid_spec["related"],
        definition_ast=invalid_spec["definition"]
    )
    assert rejected is False
    assert "REJECTED" in log_rej


def test_curated_seed_corpus_ingestion(epistemic_stack):
    """Verifies that ingest_curated_seed_corpus scales knowledge through the Z3 Quality Gate."""
    ingestor = epistemic_stack["ingestor"]
    manifold = epistemic_stack["manifold"]

    initial_count = manifold.num_concepts
    report = ingestor.ingest_curated_seed_corpus()

    assert report.total_processed >= 6
    assert report.accepted_count > 0
    assert report.rejected_count == 0
    assert manifold.num_concepts > initial_count


def test_conversational_fallback_stopword_filter():
    """Verifies that offline conversational fallback handles self-queries, coding, hyperbolic math,
    and never outputs robotic stopword artifacts ('Regarding your query related to what, do, you:')."""
    self_model = SelfModelEngine()
    retriever = ManifoldRetriever(concept_names=["mass", "velocity", "force"])
    critic = AntiHallucinationCritic(self_model=self_model)
    mouth = PretrainedLinguisticMouth(
        self_model=self_model,
        retriever=retriever,
        critic=critic
    )

    # 1. Self-reflection query
    resp_self = mouth.verbalize(
        user_prompt="what do you think about yourself",
        reasoned_concepts=["what", "do", "you"],
        proof_log=None,
        is_valid=True
    )
    assert "related to what, do, you" not in resp_self
    assert "manifold" in resp_self.lower() or "neuro-symbolic" in resp_self.lower() or "departure" in resp_self.lower()

    # 2. Hyperbolic calculations query
    resp_hyp = mouth.verbalize(
        user_prompt="can you calculate hyperbolic calculatiions",
        reasoned_concepts=["can", "you", "calculate"],
        proof_log=None,
        is_valid=True
    )
    assert "related to what, do, you" not in resp_hyp
    assert "related to can, you" not in resp_hyp
    assert "hyperbolic" in resp_hyp.lower() or "lorentzian" in resp_hyp.lower() or "poincar" in resp_hyp.lower()

    # 3. Coding query
    resp_code = mouth.verbalize(
        user_prompt="do you know coding",
        reasoned_concepts=["do", "you", "know"],
        proof_log=None,
        is_valid=True
    )
    assert "related to do, you" not in resp_code
    assert "python" in resp_code.lower() or "code" in resp_code.lower() or "pytorch" in resp_code.lower()
