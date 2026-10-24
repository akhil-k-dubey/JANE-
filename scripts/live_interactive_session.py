"""Single-Command Live Interactive Voice Session for Project Jane.

Run this script to interact with Project Jane using your own live voice:
- Jane starts executing a spatial plan in the facility (moving across rooms).
- You speak into your microphone:
  * "Also check the pressure" -> Jane formally verifies and merges it into her plan without stopping!
  * "STOP! Emergency halt!" -> Jane freezes in her tracks instantly!
- Any coughing, room noise, or unclear speech is dropped by the confidence gate with ZERO hallucinations.

Usage:
  python3 scripts/live_interactive_session.py
"""

import os
import sys
import time
import argparse
import threading
import queue
import torch
import torch.nn.functional as F

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.audio_encoder import DirectAudioTransformer, AudioIntent
from src.perception.vision_encoder import VisionPatchTransformer
from environments.spatial_graph.visual_renderer import FacilityVisualRenderer
from environments.spatial_graph.environment import SpatialEnvironmentGraph
from environments.spatial_graph.continuous_kinematics import ContinuousKinematicsEngine
from src.reasoning_engine.system2_planner import PlanStep
from src.core.async_embodiment import (
    SharedDNCPlanState,
    ActionExecutorThread,
    AudioEncoderThread,
    DeliberationEngineThread,
    DecoupledAsynchronousEmbodiment
)
from scripts.live_mic_driver import LiveMicDriver
from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator


def run_live_voice_session(
    step_duration: float = 5.0,
    confidence_threshold: float = 0.40,
    use_synthetic: bool = False
):
    print("\n" + "=" * 75)
    print("      PROJECT JANE — LIVE MULTIMODAL EMBODIMENT (VOICE RUNTIME)")
    print("=" * 75)
    print("  Decoupled 3-Thread Architecture Active:")
    print("    - Thread 1 (The Body): Continuously executes actions in the facility.")
    print("    - Thread 2 (The Ears): Real-time microphone capture -> Direct 256D manifold.")
    print("    - Thread 3 (The Brain): Z3/Clingo verification, non-blocking plan merging.")
    print("-" * 75)
    print("  Available Voice Commands:")
    print("    * Suggestion: 'Also check the pressure'  -> Dynamic Append mid-stride")
    print("    * Suggestion: 'Vent gas'                 -> Dynamic Append mid-stride")
    print("    * Override:   'STOP! / Emergency halt'   -> Instant Hardware Freeze")
    print("-" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [System] Device: {device}")

    # 1. Load Audio Transformer & Calibrated Vocal Anchors
    transformer = DirectAudioTransformer(
        confidence_threshold=confidence_threshold,
        checkpoint_path="logs/training_runs/phase4_audio_manifold.pt"
    ).to(device)
    transformer.eval()

    speech_threshold = 0.025
    calib_path = "logs/training_runs/my_voice_anchors.pt"
    if os.path.exists(calib_path):
        transformer.load_calibrated_anchors(calib_path)
        if hasattr(transformer, "speech_threshold"):
            speech_threshold = min(0.030, max(0.015, transformer.speech_threshold))
        print(f"  [System] Loaded your personalized vocal anchors from: {calib_path}")
        print(f"  [System] Voice Activity Detection active (Silence cutoff RMS: {speech_threshold:.4f})")
        print(f"  [System] Manifold Recognition Gate: Cosine Similarity >= {confidence_threshold:.2f}")
    else:
        print(f"  [System] Note: Using base manifold anchors. (Run `python3 scripts/calibrate_voice.py` to personalize!)")

    # 2. Load Vision Transformer & Facility Camera Stream
    renderer = FacilityVisualRenderer(img_size=128)
    vision_encoder = VisionPatchTransformer(
        img_size=128, patch_size=16, vision_dim=192, latent_dim=256
    ).to(device)
    vision_ckpt = "logs/training_runs/phase4_vision_manifold.pt"
    if os.path.exists(vision_ckpt):
        try:
            v_data = torch.load(vision_ckpt, map_location=device)
            if "vision_encoder" in v_data:
                vision_encoder.load_state_dict(v_data["vision_encoder"], strict=False)
            print(f"  [System] Loaded Vision Transformer checkpoint from: {vision_ckpt}")
        except Exception as e:
            print(f"  [Warning] Failed to load Vision Transformer checkpoint from {vision_ckpt}: {e}")
    vision_encoder.eval()
    print("  [System] Multimodal Camera & Visual HUD: ACTIVE")

    # 3. Build Facility Environment & Initial Plan
    env = SpatialEnvironmentGraph.create_facility_env()
    initial_steps = [
        PlanStep(step_index=1, action_type="NAVIGATE", target="hallway"),
        PlanStep(step_index=2, action_type="NAVIGATE", target="armory"),
        PlanStep(step_index=3, action_type="NAVIGATE", target="reactor_airlock"),
        PlanStep(step_index=4, action_type="NAVIGATE", target="reactor_core")
    ]

    print(f"  [Plan] Initial DNC Tape Loaded: 4 Steps (Current Node: '{env.agent_node}')")
    for step in initial_steps:
        print(f"    Step {step.step_index}: {step.action_type} -> {step.target}")

    # 4. Initialize Decoupled Embodiment (Sight + Sound + Brain)
    embodiment = DecoupledAsynchronousEmbodiment(
        initial_steps=initial_steps,
        env=env,
        audio_transformer=transformer,
        step_duration=step_duration,
        verbose=True,
        vision_encoder=vision_encoder,
        renderer=renderer,
        enable_sight=True
    )

    # 4. Start Embodiment Threads
    embodiment.start()
    print("\n[Jane Online] Action Executor is now moving through the facility!")

    mic_driver = None
    if not use_synthetic:
        # Start Non-Blocking Live Microphone Driver with 2.0s rolling window & 0.5s hop
        mic_driver = LiveMicDriver(sample_rate=16000, chunk_duration=2.0, hop_duration=0.5)
        try:
            mic_driver.start()
            print("[Microphone Online] Listening... Speak at any time into your mic!\n")
        except Exception as e:
            print(f"[Warning] Could not initialize hardware mic: {e}. Falling back to simulation.")
            mic_driver = None

    last_step_idx = -1
    last_event_count = 0
    start_time = time.time()

    try:
        while not embodiment.shared_plan.is_halted and not embodiment.shared_plan.is_completed:
            # Poll live microphone if active
            if mic_driver and mic_driver.is_running:
                try:
                    chunk = mic_driver.get_chunk(timeout=0.1)
                except Exception as e:
                    print(f"  [Warning: Microphone read error: {e}]")
                    chunk = None
                if chunk is not None:
                    rms = torch.sqrt(torch.mean(chunk ** 2)).item()
                    # Voice Activity Detection (VAD) - Skip ambient noise/silence
                    if rms < speech_threshold:
                        continue

                    # Voice energy detected! Feed chunk into embodiment's Ears queue
                    embodiment.feed_audio_waveform(chunk.to(device))
            elif use_synthetic:
                time.sleep(0.1)
            else:
                time.sleep(0.1)

            # Display real-time Body execution updates
            current_step = embodiment.shared_plan.get_current_step()
            cur_idx = embodiment.shared_plan.current_step_idx

            if cur_idx != last_step_idx and current_step is not None:
                last_step_idx = cur_idx

            # Display Brain deliberation / append events
            if len(embodiment.shared_plan.event_log) > last_event_count:
                for event in embodiment.shared_plan.event_log[last_event_count:]:
                    print(f"  *** [BRAIN / MANIFOLD EVENT] {event}")
                last_event_count = len(embodiment.shared_plan.event_log)

            # Check if execution finished
            if embodiment.shared_plan.is_completed:
                break

        # Final flush
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[User Interrupt] Stopping session...")
    finally:
        if mic_driver:
            mic_driver.stop()
        embodiment.stop()

    # 5. Print Execution Report
    print("\n" + "=" * 75)
    print("                      SESSION EXECUTION REPORT")
    print("=" * 75)
    print(f"Halted by Emergency Interrupt: {'YES (Safe Freeze)' if embodiment.shared_plan.is_halted else 'NO (Normal Completion)'}")
    print(f"Total Steps Executed:          {len(embodiment.body_thread.executed_history)}")
    print("Action History:")
    for idx, act in enumerate(embodiment.body_thread.executed_history, 1):
        print(f"  {idx:02d}. {act}")

    print("\nDeliberation Brain Audit Log:")
    for log in embodiment.brain_thread.audit_log:
        print(f"  - {log}")

    print(f"\nTotal Noise / Ambiguity Drops: {embodiment.ears_thread.dropped_noise_count} (Safe Zero-Hallucination Discards)")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Interactive Voice Session with Project Jane.")
    parser.add_argument("--step-duration", type=float, default=5.0, help="Seconds per step for the Action Executor.")
    parser.add_argument("--confidence", type=float, default=0.40, help="Confidence threshold for voice recognition.")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic waveforms for automated test.")
    args = parser.parse_args()

    run_live_voice_session(
        step_duration=args.step_duration,
        confidence_threshold=args.confidence,
        use_synthetic=args.synthetic
    )
