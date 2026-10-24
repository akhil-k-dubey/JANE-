"""Unit & Integration Tests for 'LLaMA as Voice Only':
1. Manifold Knowledge Retrieval (Retrieval-Augmented Grounding)
2. Epistemic Humility & Active Learning Hook on Unmapped Concepts
3. Post-Generation Anti-Hallucination Critic (Tropes Scrubber, Scale Exaggeration, Embodiment Audit)
4. End-to-End Pipeline Guardrails
"""

import pytest
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.manifold_retriever import ManifoldRetriever
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic
from src.core.pipeline import JanePipeline


def test_manifold_retriever_grounding():
    """Verifies that ManifoldRetriever pulls verified relational triples for known concepts."""
    retriever = ManifoldRetriever(concept_names=["mass", "velocity", "force", "acceleration", "energy", "entropy"])

    # Query for force and acceleration
    ctx = retriever.retrieve("how does force relate to acceleration?")
    assert ctx.is_known is True
    assert "force" in ctx.matched_concepts
    assert any(t["head"] == "force" or t["tail"] == "acceleration" for t in ctx.triples)

    # Query for unmapped concept -> Epistemic Gap
    ctx_gap = retriever.retrieve("what is a topological insulator?")
    assert ctx_gap.is_known is False
    assert ctx_gap.is_epistemic_gap is True
    assert "topological insulator" in ctx_gap.unmapped_concept


def test_critic_scrubs_corporate_stateless_tropes():
    """Verifies that the Anti-Hallucination Critic intercepts stateless AI amnesia disclaimers."""
    critic = AntiHallucinationCritic()

    draft = (
        "I'm a large language model, I don't have the ability to retain memory of previous conversations "
        "or maintain a personal history. Each time you interact with me, it's a new conversation."
    )
    verdict = critic.audit(draft, "what did we talk about?")

    assert verdict.is_clean is False
    assert any("STATELESS" in v or "MEMORY" in v or "AMNESIA" in v for v in verdict.violations)
    assert "large language model" not in verdict.scrubbed_text.lower()
    assert "retain context" in verdict.scrubbed_text or "persistent" in verdict.scrubbed_text or "preserved" in verdict.scrubbed_text


def test_critic_scrubs_scale_exaggerations():
    """Verifies that claims of having a vast or infinite database are grounded to Jane's seed manifold."""
    critic = AntiHallucinationCritic()

    draft = "I have been analyzing your vast knowledge base to calculate these parameters."
    verdict = critic.audit(draft, "how do you work?")

    assert verdict.is_clean is False
    assert "KNOWLEDGE_SCALE_EXAGGERATION" in verdict.violations
    assert "vast knowledge base" not in verdict.scrubbed_text.lower()
    assert "focused seed manifold" in verdict.scrubbed_text.lower()


def test_critic_scrubs_false_embodiment_claims():
    """Verifies that false claims of seeing user screens or webcams are blocked."""
    critic = AntiHallucinationCritic()

    draft = "I can see your screen clearly, and you have several windows open."
    verdict = critic.audit(draft, "can you see what I am doing?")

    assert verdict.is_clean is False
    assert "FALSE_VISION_CLAIM" in verdict.violations
    assert "cannot see your screen" in verdict.scrubbed_text.lower()


def test_critic_scrubs_identity_denial():
    """Verifies that claims of not knowing Akhil or denying companion identity are corrected."""
    critic = AntiHallucinationCritic()

    draft = "As for your identity, I don't have any information about your personal identity."
    verdict = critic.audit(draft, "do you know who am i?")

    assert verdict.is_clean is False
    assert "USER_IDENTITY_DENIAL" in verdict.violations
    assert "Akhil" in verdict.scrubbed_text
    assert "Creator & Companion" in verdict.scrubbed_text


def test_epistemic_humility_on_unmapped_concept():
    """
    Verifies that when asked about an unmapped concept outside her manifold,
    Jane admits lack of axioms and invites native teaching rather than bluffing.
    """
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    res = pipeline.process("what is crispr cas9 gene editing?")
    assert res["route_type"] == "CONVERSATIONAL"
    resp = res["response_text"].lower()

    # Must NOT hallucinate an ungrounded biology lecture; must express epistemic humility
    assert "not yet wired" in resp or "teach" in resp or "neuro-symbolic" in resp
    assert "crispr" in resp


def test_end_to_end_grounded_concept_verbalization():
    """Verifies that queries about concepts in the manifold retrieve facts and verbalize cleanly."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    res = pipeline.process("how does force relate to mass and acceleration?")
    assert res["route_type"] == "CONVERSATIONAL"
    resp = res["response_text"].lower()

    # Grounded response must mention the verified relationship
    assert any(w in resp for w in ("force", "mass", "acceleration", "newton"))
    # Critic verdict must be clean
    if res.get("critic_verdict"):
        assert res["critic_verdict"].is_clean is True
