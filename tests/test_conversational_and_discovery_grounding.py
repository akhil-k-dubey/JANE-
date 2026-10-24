"""Integration tests for grounded discovery, intent correction, and capability challenges."""

import pytest
from src.core.pipeline import JanePipeline
from src.core.cognitive_frame_parser import CognitiveFrameParser


def test_discovery_cycle_deduplication():
    """Verify that consecutive discovery cycles do not endlessly repeat force_work_bridge."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=False, device="cpu")

    # Run 3 discovery cycles
    res1 = pipeline.epistemic_engine.step_curiosity_cycle()
    assert res1 is not None
    assert res1.get("status") in ("DISCOVERED", "FALSIFIED")
    c1 = res1.get("concept_name")

    res2 = pipeline.epistemic_engine.step_curiosity_cycle()
    assert res2 is not None
    c2 = res2.get("concept_name")

    # Second discovery must NOT be identical to first discovery!
    assert c1 != c2, f"Discovery repeated identical concept: {c1} == {c2}"


def test_can_you_discover_something_else_routes_to_discovery():
    """Verify that asking 'can you discover something else too' executes real discovery."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=False, device="cpu")

    res = pipeline.process("can you discover something else too")
    assert res["route_type"] == "CAPABILITY_CHALLENGE"
    assert "Yes, I can formulate and test my own hypotheses" in res["response_text"] or "candidate bridging hypothesis" in res["response_text"]
    assert "Candidate Hypothesis:" in res["response_text"] or "Candidate Axiom:" in res["response_text"]
    # Must NOT be the old passive ledger recital!
    assert "In my episodic memory ledger, the latest verified discovery is" not in res["response_text"]


def test_how_much_do_you_know_about_python_grounded():
    """Verify that asking about python routes to conversational domain inquiry, not self-model architecture dump."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    res = pipeline.process("how much do you know about python")
    assert res["route_type"] == "CONVERSATIONAL"
    assert "Python" in res["response_text"]
    # Must NOT dump internal codebase module counts or self-model bio
    assert "source modules in `src/`" not in res["response_text"]
    assert "I am Jane, an embodied neuro-symbolic" not in res["response_text"]


def test_intent_correction_adaptive_recovery():
    """Verify that negative feedback refocuses naturally via conversation history without canned apologies."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    # First turn: ask about python
    pipeline.process("how much do you know about python")

    # Second turn: user corrects Jane
    res_corr = pipeline.process("that's not what i asked")
    assert res_corr["route_type"] == "CONVERSATIONAL"
    # Must NOT issue canned corporate apologies
    assert "I apologize" not in res_corr["response_text"]
    # Must NOT have the old silly manifold fallback about 'not, asked'!
    assert "surrounding not, asked" not in res_corr["response_text"]
    # Must refocus on the prior inquiry
    assert "how much do you know about python" in res_corr["response_text"]
