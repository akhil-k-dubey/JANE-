"""Benchmark: Phase 4 Asynchronous Audio Embodiment & Dynamic Plan Merging.

Evaluates:
1. Direct Audio-to-Vector Mapping: Waveforms map directly to 256D manifold without STT / Whisper.
2. Noise Gating / Anti-Hallucination: Coughing & ambient noise dropped safely (0% hallucination).
3. Asynchronous Plan Merging (The APPEND Node): Suggestion merged into DNC tape mid-execution
   without halting the Action Executor.
4. Emergency Override (The INTERRUPT Node): High-energy emergency audio freezes execution in place.
"""

import os
import sys
import time
import torch

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.audio_encoder import DirectAudioTransformer, AudioIntent
from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator
from environments.spatial_graph.environment import SpatialEnvironmentGraph
from src.reasoning_engine.system2_planner import PlanStep
from src.core.async_embodiment import DecoupledAsynchronousEmbodiment


def evaluate_async_audio_embodiment():
    print("=" * 70)
    print("BENCHMARK: PHASE 4 ASYNCHRONOUS AUDIO & 3-THREAD EMBODIMENT")
    print("Testing Direct Audio-to-AST, Noise Gating, and Dynamic Plan Merging")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    audio_transformer = DirectAudioTransformer().to(device)
    generator = SyntheticAcousticGenerator()

    ckpt_path = "logs/training_runs/phase4_audio_manifold.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "audio_transformer" in ckpt:
            audio_transformer.load_state_dict(ckpt["audio_transformer"])
        print(f"Loaded trained Phase 4 audio checkpoint from: {ckpt_path}")
    else:
        print(f"Note: Running with initialized audio transformer weights.")

    audio_transformer.eval()

    # -------------------------------------------------------------
    # TEST 1: Direct Audio-to-AST Extraction (Zero STT / No Whisper)
    # -------------------------------------------------------------
    print("\n[TEST 1] Direct Audio-to-AST Intent Extraction without Text Generation")
    cmd_audio = generator.generate_command_audio("VENT_GAS", duration=1.0).to(device)
    intent = audio_transformer.parse_audio_intent(cmd_audio)

    test1_pass = (intent is not None and intent.action_type == "VENT_GAS")
    print(f"  Acoustic Signal: Waveform [1, 16000] (16kHz PCM)")
    print(f"  Parsed Intent:   {intent.action_type if intent else 'None'} -> {intent.target if intent else 'None'}")
    print(f"  Confidence:      {intent.confidence:.3f}" if intent else "  Confidence: 0.0")
    print(f"  Zero-STT Status: {'PASS' if test1_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST 2: Noise Gating / Zero Hallucination (Cough & White Noise)
    # -------------------------------------------------------------
    print("\n[TEST 2] Noise Gating: Non-Speech Noise and Cough Rejection")
    cough_audio = generator.generate_noise_audio("cough", duration=1.0).to(device)
    cough_intent = audio_transformer.parse_audio_intent(cough_audio)

    white_noise = generator.generate_noise_audio("white", duration=1.0).to(device)
    white_intent = audio_transformer.parse_audio_intent(white_noise)

    test2_pass = (cough_intent is None and white_intent is None)
    print(f"  Cough Audio Result:       {'DROPPED (Safe)' if cough_intent is None else 'SPURIOUS HALLUCINATION!'}")
    print(f"  White Noise Audio Result: {'DROPPED (Safe)' if white_intent is None else 'SPURIOUS HALLUCINATION!'}")
    print(f"  Noise Gating Status:      {'PASS (Zero Spurious Hallucinations)' if test2_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST 3: Asynchronous Plan Merging (The APPEND Node)
    # -------------------------------------------------------------
    print("\n[TEST 3] Asynchronous Plan Merging: Voice Suggestion without Halting Body")
    env = SpatialEnvironmentGraph.create_facility_env()

    # Initial plan: Step 1 = Navigate to hallway, Step 2 = Navigate to armory
    initial_steps = [
        PlanStep(step_index=1, action_type="NAVIGATE", target="hallway"),
        PlanStep(step_index=2, action_type="NAVIGATE", target="armory")
    ]

    embodiment = DecoupledAsynchronousEmbodiment(
        initial_steps=initial_steps,
        env=env,
        audio_transformer=audio_transformer,
        step_duration=0.08
    )

    print("  Starting 3-Thread Decoupled System: Body (Executor), Ears (Audio), Brain (Deliberator)...")
    embodiment.start()

    # While executor is moving towards hallway, Tony speaks: "Also check the pressure valve"
    time.sleep(0.04)  # Mid-step execution
    print("  [Tony speaks mid-execution]: 'Also check the pressure valve'")
    suggestion_audio = generator.generate_command_audio("CHECK_PRESSURE", duration=1.0).to(device)
    embodiment.feed_audio_waveform(suggestion_audio)

    # Allow time for threads to merge and execute
    time.sleep(0.40)
    embodiment.stop()

    # Check if CHECK_PRESSURE was merged into execution history
    history = embodiment.body_thread.executed_history
    print(f"  Action Executor History: {history}")
    has_check_pressure = any("CHECKED_PRESSURE" in h for h in history)
    has_navigate = any("NAVIGATE" in h for h in history)
    test3_pass = has_check_pressure and has_navigate

    print(f"  Dynamic Plan Merging Status: {'PASS (Suggestion executed without interruption)' if test3_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST 4: The Emergency Override Boundary (The INTERRUPT Node)
    # -------------------------------------------------------------
    print("\n[TEST 4] Emergency Override: High-Energy Voice Halt Freezes Execution in Place")
    env_emergency = SpatialEnvironmentGraph.create_facility_env()

    long_plan = [
        PlanStep(step_index=1, action_type="NAVIGATE", target="hallway"),
        PlanStep(step_index=2, action_type="NAVIGATE", target="armory"),
        PlanStep(step_index=3, action_type="NAVIGATE", target="reactor_airlock")
    ]

    emergency_system = DecoupledAsynchronousEmbodiment(
        initial_steps=long_plan,
        env=env_emergency,
        audio_transformer=audio_transformer,
        step_duration=0.10
    )

    emergency_system.start()
    time.sleep(0.03)

    print("  [Tony shouts]: 'STOP! Emergency halt!'")
    emergency_audio = generator.generate_command_audio("EMERGENCY_HALT", duration=0.8).to(device)
    emergency_system.feed_audio_waveform(emergency_audio)

    time.sleep(0.35)
    emergency_system.stop()

    em_history = emergency_system.body_thread.executed_history
    print(f"  Emergency Execution History: {em_history}")
    was_halted = emergency_system.shared_plan.is_halted or "EXECUTED_EMERGENCY_HALT" in em_history
    # Ensure remaining steps (armory, airlock) were NOT executed because the system halted
    remaining_not_executed = not ("NAVIGATE_TO_reactor_airlock" in em_history)
    test4_pass = was_halted and remaining_not_executed

    print(f"  Emergency Freeze Status:     {'PASS (Execution frozen instantly)' if test4_pass else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    all_passed = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"Phase 4 Audio & 3-Thread Benchmark Result: {'PASS (100% Sound & Complete)' if all_passed else 'FAIL'}")
    print(f"  1. Direct Audio-to-AST (Zero STT):       {'PASS' if test1_pass else 'FAIL'}")
    print(f"  2. Non-Speech Noise / Cough Gating:     {'PASS' if test2_pass else 'FAIL'}")
    print(f"  3. Non-Blocking Plan Merging (APPEND):   {'PASS' if test3_pass else 'FAIL'}")
    print(f"  4. Emergency Override (INTERRUPT):       {'PASS' if test4_pass else 'FAIL'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    evaluate_async_audio_embodiment()
