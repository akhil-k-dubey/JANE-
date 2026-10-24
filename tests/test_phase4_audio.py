"""Unit & Integration Tests for Phase 4: Auditory Embodiment & 3-Thread Decoupled Architecture."""

import time
import pytest
import torch

from src.perception.audio_encoder import DirectAudioTransformer, AudioIntent
from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator
from environments.spatial_graph.environment import SpatialEnvironmentGraph
from src.reasoning_engine.system2_planner import PlanStep
from src.core.async_embodiment import (
    DecoupledAsynchronousEmbodiment,
    SharedDNCPlanState
)


def test_direct_audio_transformer_dimensions():
    transformer = DirectAudioTransformer(
        sample_rate=16000,
        conv_channels=64,
        audio_dim=128,
        latent_dim=256,
        num_layers=2,
        num_heads=4
    )

    # 1 second of 16kHz audio: 16000 samples
    dummy_wave = torch.randn(2, 1, 16000)

    # Forward pass maps directly into 256D manifold space
    z_audio = transformer(dummy_wave)
    assert z_audio.shape == (2, 256)
    assert torch.isfinite(z_audio).all()


def test_synthetic_acoustic_generator():
    generator = SyntheticAcousticGenerator(sample_rate=16000)

    # Command audio
    cmd_audio = generator.generate_command_audio("VENT_GAS", duration=1.0)
    assert cmd_audio.shape == (1, 16000)
    assert torch.max(torch.abs(cmd_audio)).item() <= 1.05

    # Cough noise
    cough_audio = generator.generate_noise_audio("cough", duration=1.0)
    assert cough_audio.shape == (1, 16000)


def test_noise_gating_anti_hallucination():
    """
    Core Property:
    Non-speech noise (coughing, ambient noise) MUST be dropped by the confidence gate
    without hallucinating random words or actions.
    """
    transformer = DirectAudioTransformer(confidence_threshold=0.50)
    generator = SyntheticAcousticGenerator(sample_rate=16000)

    # 1. Cough audio
    cough = generator.generate_noise_audio("cough", duration=1.0)
    intent_cough = transformer.parse_audio_intent(cough)
    assert intent_cough is None  # Dropped without hallucination!

    # 2. White noise
    noise = generator.generate_noise_audio("white", duration=1.0)
    intent_noise = transformer.parse_audio_intent(noise)
    assert intent_noise is None  # Dropped without hallucination!


def test_asynchronous_plan_merging_suggestion():
    """
    Thread 1 (Body) executes steps continuously.
    Thread 2 (Ears) receives audio suggestion.
    Thread 3 (Brain) verifies and inserts into DNC tape right after current step.
    Executor never halts; seamlessly finishes and executes the merged step.
    """
    env = SpatialEnvironmentGraph.create_facility_env()
    transformer = DirectAudioTransformer()
    generator = SyntheticAcousticGenerator()

    initial_steps = [
        PlanStep(step_index=1, action_type="NAVIGATE", target="hallway"),
        PlanStep(step_index=2, action_type="NAVIGATE", target="armory")
    ]

    embodiment = DecoupledAsynchronousEmbodiment(
        initial_steps=initial_steps,
        env=env,
        audio_transformer=transformer,
        step_duration=0.06
    )

    embodiment.start()
    time.sleep(0.03)

    # Suggestion audio injected mid-step
    suggestion_audio = generator.generate_command_audio("CHECK_PRESSURE", duration=1.0)
    embodiment.feed_audio_waveform(suggestion_audio)

    # Allow time to execute
    t_start = time.time()
    while time.time() - t_start < 1.0:
        if any("CHECKED_PRESSURE" in h for h in embodiment.body_thread.executed_history):
            break
        time.sleep(0.05)
    embodiment.stop()

    # The action executor history should contain both navigation and the merged pressure check
    history = embodiment.body_thread.executed_history
    assert any("NAVIGATE" in h for h in history)
    assert any("CHECKED_PRESSURE" in h for h in history)


def test_emergency_override_boundary():
    """
    High-energy EMERGENCY_HALT audio triggers an immediate hardware interrupt,
    halting the Action Executor and freezing DNC tape execution.
    """
    env = SpatialEnvironmentGraph.create_facility_env()
    transformer = DirectAudioTransformer()
    generator = SyntheticAcousticGenerator()

    long_plan = [
        PlanStep(step_index=1, action_type="NAVIGATE", target="hallway"),
        PlanStep(step_index=2, action_type="NAVIGATE", target="armory"),
        PlanStep(step_index=3, action_type="NAVIGATE", target="reactor_airlock"),
        PlanStep(step_index=4, action_type="NAVIGATE", target="reactor_core")
    ]

    embodiment = DecoupledAsynchronousEmbodiment(
        initial_steps=long_plan,
        env=env,
        audio_transformer=transformer,
        step_duration=0.08
    )

    embodiment.start()
    time.sleep(0.02)

    # Emergency halt shouted
    emergency_audio = generator.generate_command_audio("EMERGENCY_HALT", duration=0.8)
    embodiment.feed_audio_waveform(emergency_audio)

    time.sleep(0.30)
    embodiment.stop()

    history = embodiment.body_thread.executed_history
    # System must halt and NOT execute deep remaining steps
    assert embodiment.shared_plan.is_halted or "EXECUTED_EMERGENCY_HALT" in history
    assert "NAVIGATE_TO_reactor_core" not in history
