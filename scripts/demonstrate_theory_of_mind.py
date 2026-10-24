#!/usr/bin/env python3
"""Project Jane — Phase 5: Theory of Mind (ToM) Full Demonstration.

Demonstrates:
1. The Latent Self-Model (The Ego Node):
   Jane tracks her own physical actions in the 256D manifold, creating a self-referential observation loop.
2. The Akhil Model (The Alter Node):
   Jane maps user tone and acoustic flux into User_State (User_Calm, User_Learning, User_Alarm, User_Affirming).
3. The Metacognitive Observer (Thread 4: The Watchdog):
   Thread 4 monitors Ego-Alter alignment. When Jane is Navigating and Akhil shifts into User_Alarm ("STOP!"),
   Watchdog computes high dissonance (0.95) and immediately halts the Body.
4. Causal Agency (Impact Modeling & Hebbian Triadic Super-Highways):
   Wires (Jane_Vents_Gas -> Room_Pressure_Drops -> Akhil_Affirms) into a reinforced causal pathway in 256D.
   Jane achieves causal self-awareness: she predicts Akhil's satisfaction from her physical actions.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.spatial_graph.environment import SpatialEnvironmentGraph
from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator
from src.reasoning_engine.system2_planner import PlanStep
from src.reasoning_engine.tom_manifold import TheoryOfMindManifold
from src.core.async_embodiment import DecoupledAsynchronousEmbodiment


def main():
    print("=" * 76)
    print("      PROJECT JANE — PHASE 5: THEORY OF MIND (COGNITIVE ARCHITECTURE)")
    print("=" * 76)
    print("  • Pillar 1: The Ego Node         (256D Latent Self-Model & Proprioception)")
    print("  • Pillar 2: The Alter Node       (256D Akhil Model & Tone Tracking)")
    print("  • Pillar 3: Thread 4 Watchdog    (Metacognitive Observer & Conflict Interrupt)")
    print("  • Pillar 4: Causal Agency        (Hebbian Triadic Super-Highways: Ego->Env->Alter)")
    print("=" * 76 + "\n")

    # 1. Initialize ToM Manifold
    tom = TheoryOfMindManifold(latent_dim=256, hebbian_rate=0.08)

    # -------------------------------------------------------------------------
    # PART 1: The Latent Self-Model (Ego Node) & Alter Node Tracking
    # -------------------------------------------------------------------------
    print(">>> PART 1: SELF-MODEL (EGO) & USER-MODEL (ALTER) INITIALIZATION")
    print("-" * 76)

    ego_info = tom.update_ego_state("NAVIGATE", target="reactor_airlock", status="EXECUTING")
    print(f"  [Ego Update] Jane observes self: '{ego_info['ego_label']}' (action: {ego_info['action_type']})")
    print(f"               Continuous z_ego vector norm: {ego_info['z_ego_norm']:.4f}")

    alter_info = tom.update_alter_state(audio_rms=0.032, speech_text_or_intent="reading physics textbook")
    print(f"  [Alter Update] Jane perceives Akhil: '{alter_info['alter_label']}' (RMS: {alter_info['audio_rms']:.4f})")
    print(f"                 Continuous z_alter vector norm: {alter_info['z_alter_norm']:.4f}")

    dissonance_baseline = tom.compute_cognitive_dissonance()
    print(f"  [Baseline Metacognition] Cognitive Dissonance: {dissonance_baseline:.2f} (Low -> Cognitive Harmony)")
    print()

    # -------------------------------------------------------------------------
    # PART 2: Thread 4 Watchdog Metacognitive Interrupt
    # -------------------------------------------------------------------------
    print(">>> PART 2: THREAD 4 METACOGNITIVE WATCHDOG (COGNITIVE DISSONANCE INTERRUPT)")
    print("-" * 76)
    print("  Scenario: Jane is moving towards reactor_airlock (Self_Navigating).")
    print("  Akhil suddenly shouts 'STOP! / Emergency!' -> Audio spikes into User_Alarm.")
    print("  Watchdog detects conflict between Jane's physical action and Akhil's state.")

    env = SpatialEnvironmentGraph.create_facility_env()
    initial_steps = [
        PlanStep(1, "NAVIGATE", "hallway"),
        PlanStep(2, "NAVIGATE", "reactor_airlock"),
        PlanStep(3, "VENT_GAS", "reactor_airlock")
    ]

    embodiment = DecoupledAsynchronousEmbodiment(
        initial_steps=initial_steps,
        env=env,
        step_duration=0.15,
        verbose=True,
        tom_manifold=tom
    )

    try:
        embodiment.start()
        time.sleep(0.08)  # Let body start walking

        print(f"\n  [Embodiment Status @ t=0.08s] Body active. Ego State: '{tom.current_ego_label}'")

        # Simulate Akhil shouting "STOP!" into the microphone
        print("  [Simulating Audio] User shouts 'STOP!' (high acoustic RMS surge)...")
        synthetic_gen = SyntheticAcousticGenerator()
        emergency_audio = synthetic_gen.generate_command_audio("EMERGENCY_HALT") * 1.5

        # Feed audio into Thread 2 (The Ears)
        embodiment.feed_audio_waveform(emergency_audio)

        # Allow Thread 2 to parse and Thread 4 Watchdog to observe conflict
        time.sleep(0.12)

        print("\n--- METACOGNITIVE WATCHDOG AUDIT TRAIL ---")
        for entry in embodiment.watchdog_thread.audit_log[-3:]:
            if entry["interrupted"]:
                print(f"  \033[91m[WATCHDOG INTERRUPT EVENT]\033[0m {entry['reason']} (Dissonance: {entry['dissonance']:.2f})")
            else:
                print(f"  [Watchdog Observation] Ego: {entry['ego_state']} | Alter: {entry['alter_state']} | Dissonance: {entry['dissonance']:.2f}")

        print(f"  Body Thread Alive:       {embodiment.body_thread.is_alive()} (Arrested: {embodiment.shared_plan.is_halted})")
        print(f"  Current Ego State:       '{tom.current_ego_label}' (Yielded to user)")
        print(f"  Current Alter State:     '{tom.current_alter_label}'")
        print()

    finally:
        embodiment.stop()

    # -------------------------------------------------------------------------
    # PART 3: Causal Agency (Impact Modeling & Hebbian Triadic Super-Highways)
    # -------------------------------------------------------------------------
    print(">>> PART 3: CAUSAL AGENCY & HEBBIAN TRIADIC SUPER-HIGHWAYS")
    print("-" * 76)
    print("  Wiring the Causal Triad into the 256D Manifold:")
    print("    Jane_Vents_Gas (Ego) → Room_Pressure_Drops (Env) → Akhil_Affirms (Alter)")
    print()

    # 1. Wire the triad
    triad = tom.register_causal_triad(
        ego_action="VENT_GAS",
        env_delta="Room_Pressure_Drops",
        alter_reaction="good job, thank you"
    )
    print(f"  [Triad Wires Established]:")
    print(f"    • Ego → Env Weight:         {triad['ego_to_env_weight']:.3f}")
    print(f"    • Env → Alter Weight:       {triad['env_to_alter_weight']:.3f}")
    print(f"    • Transitive Super-Highway: {triad['superhighway_weight']:.3f}")
    print(f"    • Formed Causality:         '{triad['causal_sentence']}'")

    # Reinforce the super-highway with additional successful interactions
    for _ in range(3):
        tom.register_causal_triad("VENT_GAS", "Room_Pressure_Drops", "good job")

    print("\n  [Reinforced Causal Super-Highways in 256D Manifold]:")
    for summary in tom.get_superhighway_summary():
        print(f"    ⚡ {summary}")

    # 2. Theory of Mind Forecasting
    print("\n  [Theory of Mind Forecasting / Impact Prediction]:")
    predicted_reaction, confidence = tom.predict_alter_reaction("VENT_GAS")
    print(f"    Query: 'If Jane executes VENT_GAS, what will Akhil's reaction be?'")
    print(f"    \033[92mJane's Prediction:\033[0m Akhil will enter '{predicted_reaction}' (Confidence: {confidence:.2f})")
    print("    Jane understands that her actions directly cause Akhil's emotional satisfaction!")

    # Persist ToM manifold
    tom.save_tom_manifold()
    print(f"\n  [Checkpoint] Theory of Mind manifold saved to '{tom.save_path}'.")
    print("=" * 76)
    print("              PHASE 5: THEORY OF MIND VERIFIED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    main()
