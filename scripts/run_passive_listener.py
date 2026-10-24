#!/usr/bin/env python3
"""Project Jane — Standalone Passive Acoustic Listener & Continuous Hebbian Learner.

Listens to the user in the background as they work, talk to themselves,
teach loudly, or read books aloud:
1. Zero Speech-to-Text: Directly maps acoustic flux into 256D concept vectors.
2. Continual Vocal Tone Learning: Adapts user's pitch, timbre, and cadence centroids.
3. Hebbian Manifold Wiring: 'Neurons that fire together wire together' — forms relational
   concept maps from the semantic flux of ideas spoken aloud.
4. Intelligent VAD Sleep Gating: Keeps GPU/CPU completely cold during room silence.
"""

import sys
import os
import time
import signal
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mode_router import ModeRouter, OperatingMode


def main():
    print("=" * 72)
    print("      PROJECT JANE — CONTINUOUS PASSIVE ACOUSTIC LEARNING DAEMON")
    print("=" * 72)
    print("  • Mode:               PASSIVE (Action execution disabled, ear fully active)")
    print("  • Concept Manifold:   256D Geometric Space (Hebbian Plasticity)")
    print("  • Zero STT:           Acoustic vocal tract mapped directly without text")
    print("  • Hardware Safety:    Intelligent VAD gating (Sleeps during silence)")
    print("-" * 72)
    print("  Speak freely: talk to yourself, lecture out loud, or read books aloud.")
    print("  Jane will continuously adapt to your tone and weave concept maps.")
    print("  Press [Ctrl+C] to stop and save the learned manifold.")
    print("=" * 72 + "\n")

    def on_flux(telemetry):
        rms = telemetry.get("audio_rms", 0.0)
        tone = telemetry.get("tone_label", "Speaking")
        fired = [c[0] for c in telemetry.get("fired_concepts", [])]
        edges = telemetry.get("reinforced_edges", [])
        count = telemetry.get("utterance_count", 0)

        bars = "■" * int(min(25, rms * 250))
        print(f"\033[1;34m[Flux #{count:03d}]\033[0m RMS {rms:.4f} {bars} | Tone: \033[1;33m{tone}\033[0m")
        if fired:
            print(f"  \033[1;32m→ Fired Concepts:\033[0m {', '.join(fired)}")
        if edges:
            edge_strs = [f"{c1} <-> {c2} ({w_old:.2f}→{w_new:.2f})" for c1, c2, w_old, w_new in edges[:3]]
            print(f"  \033[1;35m→ Hebbian Wire:\033[0m   {', '.join(edge_strs)}")
        print()

    router = ModeRouter(
        mode=OperatingMode.PASSIVE,
        audio_threshold_rms=0.0280,
        autosave_interval_sec=30.0,
        on_flux_callback=on_flux
    )

    try:
        router.start_passive_listening()
        print("[Listener Active] Listening on microphone... (Speaking will trigger learning)\n")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Jane] Stopping listener and persisting learned manifold...")
    finally:
        router.stop_passive_listening()
        h = router.hebbian_manifold
        links = h.get_strongest_links(top_n=5)
        print("\n" + "=" * 55)
        print("  LEARNED CONCEPT MAP SUMMARY")
        print("=" * 55)
        print(f"  • Total Utterances Learned: {h.total_utterances_observed}")
        print(f"  • Top Wired Concept Associations:")
        for c1, c2, w in links:
            print(f"     [{c1:14s} <-> {c2:14s}] weight: {w:.3f}")
        print(f"  • Manifold checkpoint saved to: '{h.save_path}'")
        print("=" * 55)


if __name__ == "__main__":
    main()
