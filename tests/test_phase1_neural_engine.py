"""Comprehensive Unit & System Tests for Project Jane Phase 1 Neural Engine (Zero Hardcode, Zero Eval, Zero Regex)."""

import pytest
import torch

from src.translator.tokenization import SemanticTokenizer, LogicSyntaxParser
from src.reasoning_engine.nalu_arithmetic import NALUCluster, NeuralArithmeticLogicUnit
from src.reasoning_engine.dnc_memory import DNCMemoryTape
from src.reasoning_engine.energy_constraints import EnergyConstraintLandscape, PhysicsBarrier, DualHeadConceptBottleneck
from src.reasoning_engine.manifold import RelationalManifold
from src.symbolic_verifier.contracts import (
    MathHypothesis,
    PhysicsHypothesis,
    SpatialLogicHypothesis,
    KinematicTrajectoryHypothesis,
    PhysicalQuantity,
    Dimension
)
from src.symbolic_verifier.safe_ast import SafeMathCompiler, SecurityException
from src.symbolic_verifier.verifier_orchestrator import SymbolicVerifierOrchestrator
from src.core.pipeline import JanePipeline


def test_safe_ast_rejects_eval_injection():
    compiler = SafeMathCompiler()

    # 1. RCE / Arbitrary code execution attempt
    malicious_payloads = [
        "__import__('os').system('ls')",
        "eval('2 + 2')",
        "open('/etc/passwd').read()",
        "lambda x: x + 1",
        "[x for x in range(10)]",
        "os.remove('test.txt')"
    ]

    for payload in malicious_payloads:
        with pytest.raises((SecurityException, ValueError)):
            compiler.parse_to_ast(payload)

    # 2. Legitimate math is accepted
    legit_tree = compiler.parse_to_ast("2 * x + 3 * y - 10 / 2")
    symbols = compiler.extract_symbols(legit_tree)
    assert symbols == {"x", "y"}


def test_transcendental_interval_arithmetic_hard_rejection():
    orchestrator = SymbolicVerifierOrchestrator()

    # A. Bounded transcendental evaluated via rigorous Interval Arithmetic (mpmath.iv)
    # sin(pi/6) == 0.5. For x in [0.5235, 0.5236] (approx pi/6), sin(x) in [0.4999, 0.5001]
    hyp_bounded = MathHypothesis(
        lhs="sin(x)",
        rhs="0.5",
        tolerance=0.01,
        var_bounds={"x": (0.52, 0.53)}
    )
    result_bounded = orchestrator.verify(hyp_bounded)
    assert result_bounded.is_valid is True
    assert result_bounded.prover_name == "Interval_Arithmetic"

    # B. Unbounded transcendental math MUST be strictly rejected as UNKNOWN (No false Taylor proofs)
    hyp_unbounded = MathHypothesis(lhs="sin(x)", rhs="0.5")
    result_unbounded = orchestrator.verify(hyp_unbounded)
    assert result_unbounded.is_valid is False
    assert "UNKNOWN" in result_unbounded.proof_log or "rejected" in result_unbounded.proof_log

    # C. Variable named 'sin_wave' or 'cosine_factor' must NOT be flagged as transcendental (AST Call node check)
    hyp_var_named_sin = MathHypothesis(
        lhs="(sin_wave + cosine_factor) * 2",
        rhs="2 * sin_wave + 2 * cosine_factor"
    )
    result_var_named_sin = orchestrator.verify(hyp_var_named_sin)
    # The algebraic solver can solve this with SMT, not failing on transcendental check
    assert result_var_named_sin.prover_name == "Z3_SMT"
    assert result_var_named_sin.is_valid is True


def test_dual_head_concept_bottleneck_and_dimensional_analysis():
    orchestrator = SymbolicVerifierOrchestrator()

    # 1. Dual-Head Concept Bottleneck Architecture
    bottleneck = DualHeadConceptBottleneck(latent_dim=256, num_channels=6)
    latent = torch.randn(1, 256)
    mags, dim_probs = bottleneck(latent)
    assert mags.shape == (1, 6)
    assert dim_probs.shape == (1, 6, 8)  # 8 SI base dimensions

    # 2. Formal Dimensional Analysis: Dimensionally sound configuration
    hyp_sound = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=15.0, dimension=Dimension.MASS, confidence=0.98),
        PhysicalQuantity(magnitude=500.0, dimension=Dimension.VELOCITY, confidence=0.95),
        PhysicalQuantity(magnitude=1000.0, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
        PhysicalQuantity(magnitude=800.0, dimension=Dimension.ENERGY, confidence=0.99, is_output=True)
    ])
    res_sound = orchestrator.verify(hyp_sound)
    assert res_sound.is_valid is True

    # 3. Dimensional Analysis Violation: Negative Mass
    hyp_bad_mass = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=-10.0, dimension=Dimension.MASS, confidence=0.99)
    ])
    res_bad_mass = orchestrator.verify(hyp_bad_mass)
    assert res_bad_mass.is_valid is False
    assert "[MASS]" in res_bad_mass.violation_details

    # 4. Dimensional Analysis Violation: Speed of light exceeded
    hyp_faster_than_light = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=3.5e8, dimension=Dimension.VELOCITY, confidence=0.99)
    ])
    res_ftl = orchestrator.verify(hyp_faster_than_light)
    assert res_ftl.is_valid is False
    assert "relativistic speed limit" in res_ftl.violation_details

    # 5. Concept Bottleneck Confidence Gate: Ambiguous dimension (<85%) is rejected
    hyp_uncertain = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=50.0, dimension=Dimension.MASS, confidence=0.60)
    ])
    res_uncertain = orchestrator.verify(hyp_uncertain)
    assert res_uncertain.is_valid is False
    assert "uncertain" in res_uncertain.violation_details.lower()

    # 6. Strict Type Rejection: Raw dicts or invalid quantity types rejected
    hyp_raw_dict = PhysicsHypothesis(properties={"mass": 10.0})
    res_raw_dict = orchestrator.verify(hyp_raw_dict)
    assert res_raw_dict.is_valid is False
    assert "Type Error" in res_raw_dict.proof_log


def test_tiered_spatial_verification_logic_and_kinematics():
    orchestrator = SymbolicVerifierOrchestrator()

    # Tier 1: Clingo ASP Discrete Logic & Permission Verifier (Room/Zone Graph)
    logic_hyp_valid = SpatialLogicHypothesis(
        start_zone="bridge",
        target_zone="reactor_room",
        zone_sequence=["bridge", "corridor_alpha", "reactor_room"],
        allowed_transitions={("bridge", "corridor_alpha"), ("corridor_alpha", "reactor_room")},
        locked_doors={"corridor_alpha"},
        agent_keys={"corridor_alpha"}
    )
    proof_logic_valid = orchestrator.verify(logic_hyp_valid)
    assert proof_logic_valid.is_valid is True
    assert proof_logic_valid.prover_name == "Clingo_Logical_ASP"

    # Tier 1 Violation: Moving through locked zone without key
    logic_hyp_invalid = SpatialLogicHypothesis(
        start_zone="bridge",
        target_zone="armory",
        zone_sequence=["bridge", "armory"],
        allowed_transitions={("bridge", "armory")},
        locked_doors={"armory"},
        agent_keys=set()  # No key
    )
    proof_logic_invalid = orchestrator.verify(logic_hyp_invalid)
    assert proof_logic_invalid.is_valid is False

    # Tier 2: Continuous Kinematic Trajectory Solver (No state space explosion)
    kinematic_hyp_valid = KinematicTrajectoryHypothesis(
        start_pos=(0.0, 0.0),
        target_pos=(5.0, 5.0),
        waypoints=[(2.0, 0.0), (2.0, 5.0)],
        obstacles=[(2.5, 1.0, 4.0, 3.0)]  # Box obstacle away from path
    )
    proof_kin_valid = orchestrator.verify(kinematic_hyp_valid)
    assert proof_kin_valid.is_valid is True
    assert proof_kin_valid.prover_name == "Continuous_Kinematics"

    # Tier 2 Collision: Line segment cuts directly through obstacle box
    kinematic_hyp_colliding = KinematicTrajectoryHypothesis(
        start_pos=(0.0, 2.0),
        target_pos=(5.0, 2.0),
        waypoints=[],  # Straight line (0, 2) -> (5, 2)
        obstacles=[(1.0, 1.0, 3.0, 3.0)]  # Directly blocks x in [1, 3] at y=2
    )
    proof_kin_colliding = orchestrator.verify(kinematic_hyp_colliding)
    assert proof_kin_colliding.is_valid is False
    assert "intersects obstacle" in proof_kin_colliding.proof_log


def test_zero_regex_tokenizer_and_ilr_parser():
    tok = SemanticTokenizer()
    text = "Adjust reactor power to 40% and thruster to +2.5 MW"

    # 1. Zero-regex character lexing
    semantic_tokens = tok.encode(text)
    pct_tok = [t for t in semantic_tokens if t.token_type == "PERCENTAGE"][0]
    scalar_tok = [t for t in semantic_tokens if t.token_type == "SCALAR"][0]

    assert pct_tok.raw_value == 40.0
    assert scalar_tok.raw_value == 2.5

    # 2. Logic AST parser (ILR verification)
    ilr_str = "action(adjust, target(reactor_power), value(40, percent))"
    ast_root = LogicSyntaxParser.parse_ilr(ilr_str)
    assert ast_root.predicate == "action"
    assert len(ast_root.args) == 3
    assert ast_root.args[0].predicate == "adjust"
    assert ast_root.args[1].predicate == "target"
    assert ast_root.args[1].args[0].predicate == "reactor_power"


def test_end_to_end_jane_pipeline():
    pipeline = JanePipeline(latent_dim=256, vocab_size=1000, max_reasoning_steps=10, device="cpu")
    hyp = PhysicsHypothesis(quantities=[
        PhysicalQuantity(magnitude=100.0, dimension=Dimension.MASS, confidence=0.99),
        PhysicalQuantity(magnitude=500.0, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
        PhysicalQuantity(magnitude=400.0, dimension=Dimension.ENERGY, confidence=0.99, is_output=True)
    ])

    result = pipeline.process(
        text="Adjust reactor power to 40%",
        hypothesis=hyp
    )

    assert "response_text" in result
    assert result["verification"].is_valid is True
    assert result["converged_at_step"] <= 10
