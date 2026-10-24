"""Unit Tests for:
1. JaneConfig Centralized Architecture & Hyperparameter Configuration
2. Single-Source-of-Truth IntentRouter.infer_hypothesis
3. Enhanced Teaching Utterance Parsing without Brittle String Patterns
"""

import pytest
import threading
import torch
from src.core.config import (
    JaneConfig,
    ManifoldConfig,
    TheoryOfMindConfig,
    AudioConfig,
    EmbodimentConfig,
    PipelineConfig,
    DEFAULT_CONFIG
)
from src.core.intent_router import IntentRouter, RouteType
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.symbolic_verifier.contracts import MathHypothesis, PhysicsHypothesis


def test_config_defaults():
    cfg = DEFAULT_CONFIG
    assert cfg.manifold.latent_dim == 256
    assert cfg.manifold.hebbian_weight_boost == 0.45
    assert cfg.tom.dissonance_critical_threshold == 0.70
    assert cfg.tom.dissonance_elevated_threshold == 0.40
    assert cfg.embodiment.similarity_gate == 0.15
    assert cfg.embodiment.step_duration == 0.05
    assert cfg.pipeline.latent_steps_cap == 20


def test_custom_config_injection(tmp_path):
    custom_cfg = JaneConfig(
        manifold=ManifoldConfig(latent_dim=256, decay_rate=0.005, hebbian_weight_boost=0.55),
        tom=TheoryOfMindConfig(dissonance_critical_threshold=0.85)
    )

    m = HebbianConceptManifold(config=custom_cfg.manifold, save_path=str(tmp_path / "m.pt"))
    assert m.decay_rate == 0.005

    learner = ActiveLearner(manifold=m, config=custom_cfg, triples_path=str(tmp_path / "triples.json"))
    assert learner.config.manifold.hebbian_weight_boost == 0.55


def test_intent_router_infer_hypothesis_math():
    router = IntentRouter()

    # 1. Calculation queries
    h1 = router.infer_hypothesis("calculate 12 * 12")
    assert isinstance(h1, MathHypothesis)
    assert h1.lhs == "12 * 12"
    assert h1.rhs == "144"

    h2 = router.infer_hypothesis("what is 100 / 4")
    assert isinstance(h2, MathHypothesis)
    assert h2.lhs == "100 / 4"
    assert h2.rhs == "25.0"

    # 2. Equation assertions
    h3 = router.infer_hypothesis("does 15 + 15 equal 30")
    assert isinstance(h3, MathHypothesis)
    assert h3.lhs == "15 + 15"
    assert h3.rhs == "30"

    h4 = router.infer_hypothesis("is 7 * 7 == 49")
    assert isinstance(h4, MathHypothesis)
    assert h4.lhs == "7 * 7"
    assert h4.rhs == "49"

    # 3. Conversational non-math input should return None
    h_conv = router.infer_hypothesis("how do you solve differential equations?")
    assert h_conv is None


def test_intent_router_infer_hypothesis_physics():
    router = IntentRouter()

    h_phys = router.infer_hypothesis("verify first law conservation of energy: 150 W input vs 120 W output")
    assert isinstance(h_phys, PhysicsHypothesis)
    assert len(h_phys.quantities) == 2
    assert h_phys.quantities[0].magnitude == 150.0
    assert h_phys.quantities[1].magnitude == 120.0


def test_enhanced_teaching_utterance_parsing():
    learner = ActiveLearner()

    # Pattern with 'define' verb
    u1 = "Jane, define dark_energy as the cosmological force causing accelerating expansion."
    parsed1 = learner.parse_teaching_utterance(u1)
    assert parsed1 is not None
    assert parsed1["concept_name"] == "dark_energy"
    assert "dark_energy" in parsed1["definition_ast"]

    # Pattern with quoted concept
    u2 = "Jane, a 'cosmic_string' is a 1-dimensional topological defect in spacetime."
    parsed2 = learner.parse_teaching_utterance(u2)
    assert parsed2 is not None
    assert parsed2["concept_name"] == "cosmic_string"
