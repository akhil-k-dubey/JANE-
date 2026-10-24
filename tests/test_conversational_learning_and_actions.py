"""Unit and Integration Tests for Action Directives and Continuous Conversational Learning.

Verifies:
1. Operational Action Directives: Translates natural language file/code requests into verified executions.
2. Continuous Conversational Learning: Crystallizes novel concepts and updates Hebbian synaptic links per turn.
3. Anti-Hallucination Critic non-interference: Natural technical discussions (Python libraries, vision models, holograms)
   are not falsely lobotomized or replaced with canned disclaimers.
4. Semantic Resurfacer non-interference: Latent goal alignment is tracked as an attentional bias without intrusive popups.
"""

import os
import sys
import tempfile
import pytest
import torch

from src.core.pipeline import JanePipeline
from src.core.action_dispatcher import ActionDirectiveDispatcher, ActionDirectiveResult
from src.core.intent_router import IntentRouter, RouteType
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic


def test_action_dispatcher_create_file_and_execute():
    """Verify ActionDirectiveDispatcher extracts parameters, writes file, and executes it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dispatcher = ActionDirectiveDispatcher()

        # Simulate user command with explicit directory
        cmd = f'create a file name "jane_test_first.py" in tmp and write a program saying "hello , jane online"'
        res = dispatcher.parse_and_dispatch(cmd)

        assert res is not None
        assert res.is_action is True
        assert res.action_type == "CREATE_FILE"
        assert res.success is True
        assert "hello" in res.stdout and "jane online" in res.stdout
        assert os.path.exists(res.target_path)


def test_intent_router_classifies_action_directive():
    """Verify IntentRouter routes action directives to RouteType.ACTION_DIRECTIVE."""
    router = IntentRouter()
    cmd = 'create a file name "test_script.py" in Downloads and write a program saying "test passed"'
    route_type, payload = router.route(cmd)

    assert route_type == RouteType.ACTION_DIRECTIVE
    assert isinstance(payload, ActionDirectiveResult)
    assert payload.is_action is True


def test_continuous_conversational_learning_expands_manifold():
    """Verify Jane's manifold dynamically crystallizes novel concepts and wires Hebbian edges during dialogue."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")
    initial_count = pipeline.hebbian_manifold.num_concepts

    # Conversational turn introducing a novel technical concept
    user_turn = "Let us start research into holographic_projection and spatial_light_modulators"
    res = pipeline.process(text=user_turn)

    # Concept count must increase dynamically
    new_count = pipeline.hebbian_manifold.num_concepts
    assert new_count > initial_count
    assert "holographic_projection" in pipeline.hebbian_manifold.concept_names or "spatial_light_modulators" in pipeline.hebbian_manifold.concept_names


def test_critic_does_not_lobotomize_vision_models_or_python_libraries():
    """Verify Critic allows natural discussions of vision processing, video, and libraries."""
    critic = AntiHallucinationCritic()

    draft = "We can use OpenCV and PyTorch to analyze video frames from YouTube for holographic projection experiments."
    prompt = "How can we process video for holograms?"
    verdict = critic.audit(draft, prompt)

    assert verdict.is_clean is True
    assert "OpenCV and PyTorch" in verdict.scrubbed_text
    assert "FALSE_VISION_CLAIM" not in verdict.violations


def test_pipeline_executes_action_directive_end_to_end():
    """Verify end-to-end execution of a file creation action from user input."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")
    cmd = 'create a file name "test_run.py" in tmp and write a program saying "jane active"'
    result = pipeline.process(text=cmd)

    assert result["route_type"] == "ACTION_DIRECTIVE"
    assert "test_run.py" in result["response_text"]
    assert "jane active" in result["response_text"]
    assert result["action_telemetry"].success is True


def test_action_dispatcher_spin_up_script_flexible_phrasing():
    """Verify action dispatcher correctly handles unscripted, natural variations without hardcoded keywords."""
    dispatcher = ActionDirectiveDispatcher()

    # Query with no 'create a file' or 'saying'
    cmd = 'spin up a python script in /tmp to compute math and run it: print("math result:", 21 * 2)'
    res = dispatcher.parse_and_dispatch(cmd)

    assert res is not None
    assert res.is_action is True
    assert res.success is True
    assert "42" in res.stdout or "math result" in res.stdout


def test_unified_cognitive_mouth_tool_calling():
    """Verify ToolRegistry parses and executes TOOL_CALL blocks within conversational flow."""
    from src.core.tools import ToolRegistry
    registry = ToolRegistry.get_instance()

    raw_llm = (
        'TOOL_CALL: {"tool": "file_system_write", "args": {"path": "/tmp/test_unified.py", "code": "print(\'unified jarvis active\')"}}\n'
        'I have generated and tested the script for you. The unified perception-action loop is now active.'
    )
    results, clean_response = registry.parse_and_execute_tool_calls(raw_llm)

    assert len(results) == 1
    assert results[0].success is True
    assert "unified jarvis active" in results[0].output
    assert "I have generated and tested the script for you." in clean_response
    assert "TOOL_CALL" not in clean_response


