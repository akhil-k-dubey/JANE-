"""Tests for Domain Topology, Continuous Domain Classification, and Grounded Python Introspection."""

import pytest
from src.core.pipeline import JanePipeline
from src.reasoning_engine.domain_topology import DomainTopologyEngine


def test_domain_topology_classification():
    """Verify that 256D continuous domain topology classifies concepts accurately."""
    engine = DomainTopologyEngine.get_instance()

    d_py, s_py = engine.classify_domain("python programming language and script synthesis")
    assert d_py == "Computer Science & Coding"

    d_phys, s_phys = engine.classify_domain("superconductivity and macroscopic quantum Meissner effect")
    assert d_phys == "Physics & Energy"

    d_math, s_math = engine.classify_domain("calculus derivative and non-euclidean hyperbolic manifolds")
    assert d_math == "Mathematics & Logic"

    d_cog, s_cog = engine.classify_domain("theory of mind perspective tracking and popperian falsification")
    assert d_cog == "Cognitive & Epistemic"


def test_domain_density_monitoring():
    """Verify that domain density balancer monitors distribution and detects saturation."""
    engine = DomainTopologyEngine.get_instance()
    counts = {
        "Computer Science & Coding": 10,
        "Physics & Energy": 45,
        "Mathematics & Logic": 15,
        "Cognitive & Epistemic": 8,
        "Speech & Audio Modality": 5
    }
    report = engine.monitor_density(counts, density_ceiling=40)
    assert report["Physics & Energy"]["is_dense"] is True
    assert report["Computer Science & Coding"]["is_dense"] is False
    assert report["Physics & Energy"]["status"] == "SATURATED"
    assert report["Computer Science & Coding"]["status"] == "BALANCED"


def test_python_grounded_in_computer_science():
    """Verify that Python is attributed to Computer Science & Coding with verified triples."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    ctx = pipeline.retriever.retrieve("how much knowledge you currently have about python")
    assert "Computer Science & Coding" in ctx.matched_domains
    assert "python" in ctx.matched_concepts
    assert any(t["head"] == "python" for t in ctx.triples)


def test_multi_turn_python_intent_correction():
    """Verify the exact user sequence: general inquiry -> negative clarification."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    # Turn 1
    res1 = pipeline.process("how much do you know about python")
    assert res1["route_type"] == "CONVERSATIONAL"
    assert "Python" in res1["response_text"]
    assert "0 knowledge entries about Python from Wikipedia" not in res1["response_text"]

    # Turn 2: User clarifies
    res2 = pipeline.process("no not that, i meant how much knowledge you currently have about python")
    assert res2["route_type"] == "INTROSPECTIVE_SELF"
    assert "Computer Science & Coding" in res2["response_text"]
    assert "python" in res2["response_text"].lower()
    assert "0 knowledge entries about Python from Wikipedia" not in res2["response_text"]
    assert "relational" in res2["response_text"].lower() or "manifold" in res2["response_text"].lower()
