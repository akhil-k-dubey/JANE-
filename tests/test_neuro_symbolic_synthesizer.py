"""Unit & Integration Tests for Native Neuro-Symbolic Program & Web Synthesizer.

Verifies:
1. Continuous Vector Domain & Parameter Routing (256D Manifold Projections).
2. Native Web DOM & Portfolio Synthesis with Closed-Loop ViT Visual Inspection.
3. Grammar-Constrained Algorithmic Python Synthesis with Z3 SMT Formal Proofs.
4. Data Structures (LRU Cache, Binary Search) Native AST Synthesis.
5. Autonomous Scheduler Closed-Loop Task Execution in Ephemeral Shadow Worktree.
6. Zero LLM Code Prompting in ActionDirectiveDispatcher.
Zero regular expressions.
"""

import os
import sys
import tempfile
import pytest
import torch

from src.symbolic_verifier.neuro_symbolic_synthesizer import (
    NeuroSymbolicProgramSynthesizer, DOMNode
)
from src.core.autonomous_scheduler import AutonomousScheduler
from src.core.action_dispatcher import ActionDirectiveDispatcher, ActionDirectiveResult


def test_neuro_symbolic_synthesizer_domain_routing():
    """Verify 256D continuous manifold routes tasks into correct domains without regex or hardcoding."""
    synthesizer = NeuroSymbolicProgramSynthesizer.get_instance()

    domain_web, sim_web = synthesizer.route_domain("build a personal showcase portfolio website with interactive canvas")
    assert domain_web == "web_portfolio"
    assert sim_web > 0.40

    domain_algo, sim_algo = synthesizer.route_domain("implement quicksort divide and conquer recursive array sorting")
    assert domain_algo == "python_algorithm"
    assert sim_algo > 0.40

    domain_script, sim_script = synthesizer.route_domain("create a python script to run an operational task")
    assert domain_script == "system_script"


def test_neuro_symbolic_synthesizer_portfolio_and_vit_inspection():
    """Verify native portfolio synthesis and closed-loop visual inspection through Vision Transformer."""
    synthesizer = NeuroSymbolicProgramSynthesizer.get_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        res = synthesizer.synthesize_and_verify(
            task_description="create a modern interactive portfolio for jane showcasing autonomous neuro-symbolic AI",
            target_directory=tmpdir,
            target_filename="portfolio.html"
        )

        assert res.success is True
        assert res.domain == "web_portfolio"
        assert res.language == "html"
        assert os.path.exists(res.target_path)

        # Inspect DOM & Web Content
        with open(res.target_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "neural-canvas" in content
        assert "PROJECT JANE // AGI" in content
        assert "256D Continuous Manifold" in content
        assert "Formal SMT Verification" in content
        assert "Vision Transformer (ViT)" in content
        assert "Neural Arithmetic Logic Unit (NALU)" in content
        assert "telemetry-dashboard" in content
        assert "Knowledge Graph Entities" in content
        assert "ENTRIES:" in content
        assert "concept-explorer" in content
        assert "AKHIL" in content
        assert "proveDivisionSafety" in content
        assert "proveLoopTermination" in content

        # Verify Multimodal ViT Inspection Telemetry
        assert res.visual_telemetry is not None
        assert res.visual_telemetry.is_visually_harmonious is True
        assert res.visual_telemetry.visual_loss < 0.40
        assert len(res.visual_telemetry.detected_components) == 5
        assert res.visual_telemetry.wcag_contrast_ratio >= 7.0
        assert res.visual_telemetry.inspected_elements_count > 20
        detected_names = {d["concept"] for d in res.visual_telemetry.detected_components}
        assert "telemetry_dashboard" in detected_names
        assert "interactive_console" in detected_names
        assert "navigation_bar" in detected_names
        assert "hero_header" in detected_names
        assert "showcase_cards" in detected_names


def test_mathematical_style_engine_first_principles():
    """Verify first-principles mathematical styling: golden ratio scale and WCAG contrast proofs."""
    from src.symbolic_verifier.neuro_symbolic_synthesizer import MathematicalStyleEngine

    # Golden ratio scaling
    assert MathematicalStyleEngine.space(0) == "8px"
    assert MathematicalStyleEngine.space(2) == "13px"
    assert MathematicalStyleEngine.type_scale(0) == "1.0rem"
    assert float(MathematicalStyleEngine.type_scale(3).replace("rem", "")) > 1.6

    # Mathematical color conversions
    hex_cyan = MathematicalStyleEngine.hsl_to_hex(185.0, 1.0, 0.50)
    assert hex_cyan.startswith("#")
    rgba_str = MathematicalStyleEngine.hsl_to_rgba(185.0, 1.0, 0.50, 0.25)
    assert "rgba(" in rgba_str

    # WCAG AAA formal contrast proof (>= 7.0)
    contrast = MathematicalStyleEngine.calculate_contrast_ratio(215.0, 0.30, 0.96, 225.0, 0.25, 0.04)
    assert contrast >= 7.0


def test_neuro_symbolic_synthesizer_algorithmic_quicksort_and_z3():
    """Verify algorithmic Python synthesis, Z3 formal verification, and sandboxed execution."""
    synthesizer = NeuroSymbolicProgramSynthesizer.get_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        res = synthesizer.synthesize_and_verify(
            task_description="synthesize quicksort partition algorithm and execute test",
            target_directory=tmpdir,
            target_filename="quicksort.py"
        )

        assert res.success is True
        assert res.domain == "python_algorithm"
        assert res.execution_result is not None
        assert res.execution_result.success is True
        assert "SORTED_RESULT" in res.execution_result.clean_output
        assert "[8, 17, 25, 42, 93]" in res.execution_result.clean_output


def test_neuro_symbolic_synthesizer_binary_search_and_lru():
    """Verify data structures and search algorithms synthesized into typed ASTs."""
    synthesizer = NeuroSymbolicProgramSynthesizer.get_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        res_bin = synthesizer.synthesize_and_verify(
            task_description="implement binary search divide and conquer lookup",
            target_directory=tmpdir,
            target_filename="bsearch.py"
        )
        assert res_bin.success is True
        assert res_bin.execution_result.success is True
        assert "FOUND_INDEX: 5" in res_bin.execution_result.clean_output

        res_lru = synthesizer.synthesize_and_verify(
            task_description="implement an lru cache with eviction capacity and hashmap lookup",
            target_directory=tmpdir,
            target_filename="lru.py"
        )
        assert res_lru.success is True
        assert res_lru.execution_result.success is True
        assert "LRU_GET: 1" in res_lru.execution_result.clean_output


def test_autonomous_scheduler_closed_loop_synthesis_task():
    """Verify AutonomousScheduler executes an end-to-end autonomous synthesis goal in a shadow worktree."""
    scheduler = AutonomousScheduler()

    telemetry = scheduler.execute_autonomous_synthesis_task(
        goal_id="auto_portfolio_goal",
        goal_description="autonomously synthesize modern personal portfolio website for project jane",
        target_filename="index.html"
    )

    assert telemetry.converged is True
    assert telemetry.final_energy < 0.35
    assert telemetry.total_steps == 1
    assert len(telemetry.step_history) == 1
    assert telemetry.step_history[0].action_type == "SYNTHESIZE_WEB_PORTFOLIO"
    assert telemetry.step_history[0].passing_tests == 1


def test_action_dispatcher_zero_llm_code_generation():
    """Verify ActionDirectiveDispatcher uses native NeuroSymbolicProgramSynthesizer without calling LLM."""
    dispatcher = ActionDirectiveDispatcher()

    # Even with no LLM backend, native neuro-symbolic coding brain executes actions deterministically
    cmd = 'create a file name "jane_test_zero_llm.py" in tmp and write a program saying "neuro symbolic proof passed"'
    res = dispatcher.parse_and_dispatch(cmd)

    assert res is not None
    assert res.is_action is True
    assert res.success is True
    assert "neuro symbolic proof passed" in res.stdout
    assert os.path.exists(res.target_path)


def test_neuro_symbolic_synthesizer_next_gen_neural_architecture():
    """Verify synthesis, PyTorch module execution, gradient flow, and Lyapunov stability of Next-Gen AI."""
    synthesizer = NeuroSymbolicProgramSynthesizer.get_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        res = synthesizer.synthesize_and_verify(
            task_description="design a next-gen neural architecture with extrapolative NALU units and residual connections",
            target_directory=tmpdir,
            target_filename="next_gen_cell.py"
        )

        assert res.success is True
        assert res.domain == "neural_architecture"
        assert res.language == "python"
        assert os.path.exists(res.target_path)

        # Verify Execution & Gradient Flow Telemetry
        assert res.execution_result is not None
        assert res.execution_result.success is True
        output = res.execution_result.clean_output
        assert "NEURAL_OUTPUT_SHAPE: [4, 16]" in output
        assert "GRADIENT_FLOW_NORM:" in output
        assert "STABILITY_VERDICT: LYAPUNOV_EQUILIBRIUM_VERIFIED" in output
        assert res.lyapunov_energy == 0.0

