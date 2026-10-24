"""Vocal Calibration Protocol: Grounding Project Jane to Your Voice.

Records 3 samples of your voice for each core command:
- CHECK_PRESSURE ("Also check the pressure")
- VENT_GAS ("Vent the gas")
- EMERGENCY_HALT ("STOP! Emergency halt")
- NAVIGATE_REACTOR ("Go to reactor")

Projects each vocal waveform directly into Jane's 256D Manifold space, averages them
into canonical "Anchor Vectors", and saves them to `logs/training_runs/my_voice_anchors.pt`.

Zero Speech-to-Text: No Whisper, no transcription, pure manifold vector alignment.
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.audio_encoder import DirectAudioTransformer
from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator


def record_clip_arecord(duration: float = 1.5, sample_rate: int = 16000) -> torch.Tensor:
    """Records a single audio clip from the microphone using arecord, discarding initial pop."""
    bytes_per_sample = 2
    warmup_bytes = int(0.25 * sample_rate * bytes_per_sample)  # Discard initial 250ms pop
    bytes_to_read = int(duration * sample_rate * bytes_per_sample)
    p = subprocess.Popen(
        ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    p.stdout.read(warmup_bytes)
    raw = p.stdout.read(bytes_to_read)
    p.terminate()
    p.wait()

    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return torch.from_numpy(audio_float).unsqueeze(0)  # [1, T]


def record_clip_sounddevice(duration: float = 1.5, sample_rate: int = 16000) -> torch.Tensor:
    """Records a single audio clip from the microphone using sounddevice."""
    import sounddevice as sd
    num_samples = int(duration * sample_rate)
    rec = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return torch.from_numpy(rec[:, 0]).unsqueeze(0)  # [1, T]


def record_clip(duration: float = 1.5, sample_rate: int = 16000) -> torch.Tensor:
    """Records a single audio clip using the best available backend."""
    # Check arecord
    res = subprocess.run(["which", "arecord"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        return record_clip_arecord(duration, sample_rate)
    try:
        return record_clip_sounddevice(duration, sample_rate)
    except Exception:
        raise RuntimeError("No microphone capture backend available.")


def calibrate_user_voice(interactive: bool = True, use_synthetic: bool = False):
    print("=" * 70)
    print("PROJECT JANE — VOCAL CALIBRATION PROTOCOL (Grounding to Real Voice)")
    print("Zero Speech-to-Text: Mapping your acoustic vocal tract directly to 256D")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base audio transformer
    transformer = DirectAudioTransformer(checkpoint_path="logs/training_runs/phase4_audio_manifold.pt").to(device)
    transformer.eval()

    commands = [
        ("CHECK_PRESSURE", "Also check the pressure / Check pressure"),
        ("VENT_GAS", "Vent gas / Vent the reactor gas"),
        ("EMERGENCY_HALT", "STOP! / Emergency halt!"),
        ("NAVIGATE_REACTOR", "Go to reactor / Navigate to reactor")
    ]

    anchor_profiles = {}
    synthetic_gen = SyntheticAcousticGenerator() if use_synthetic else None

    # Step 1: Ambient Room Noise Floor Baseline
    print("\n[Step 1: Noise Floor Baseline]")
    if interactive and not use_synthetic:
        input("Press [Enter] and stay silent for 2 seconds to sample room ambient noise...")
        time.sleep(0.4)  # Wait for keyboard click release to settle
        print("Sampling room background noise (stay quiet)...")
        noise_clip = record_clip(duration=2.0).to(device)
    else:
        print("Using synthetic ambient noise baseline...")
        noise_clip = synthetic_gen.generate_noise_audio("white", duration=2.0).to(device)

    noise_rms = torch.sqrt(torch.mean(noise_clip ** 2)).item()
    # Safely bound speech threshold between quiet room (~0.015) and speech (~0.035 - 0.060)
    speech_threshold = min(0.028, max(0.015, noise_rms * 1.10))
    print(f"  Ambient Noise RMS: {noise_rms:.4f} (Calibrated baseline)")
    print(f"  Voice Activity Threshold: {speech_threshold:.4f} (Audio below this is treated as silence)")

    # Step 2: Record 3 samples for each command
    print("\n[Step 2: Command Vocal Grounding]")
    all_recorded_vectors = []
    cmd_raw_vectors = {}

    for cmd_name, example_phrase in commands:
        print(f"\n>>> Command: {cmd_name}")
        print(f"    Suggested Phrase: '{example_phrase}'")
        sample_vectors = []

        for sample_idx in range(3):
            if interactive and not use_synthetic:
                input(f"  Sample {sample_idx + 1}/3: Press [Enter] when ready to speak '{cmd_name}'...")
                time.sleep(0.15)  # Settle keypress click
                print("  Recording [2.0 seconds] — SPEAK NOW! ", end="", flush=True)
                audio_clip = record_clip(duration=2.0).to(device)
                print("Done!")
            else:
                audio_clip = synthetic_gen.generate_command_audio(cmd_name, duration=1.0).to(device)

            rms = torch.sqrt(torch.mean(audio_clip ** 2)).item()
            peak = torch.max(torch.abs(audio_clip)).item()
            bars = "#" * int(min(25, peak * 60))
            print(f"  Captured: Peak {peak:.3f}, RMS {rms:.4f} [{bars:<25}] -> Saved.")

            # Project into 256D manifold space
            with torch.no_grad():
                z = transformer(audio_clip)  # [1, 256]
                sample_vectors.append(z.squeeze(0))
                all_recorded_vectors.append(z.squeeze(0))

        cmd_raw_vectors[cmd_name] = sample_vectors

    # Step 3: Compute Mean Voice Centroid & Mean-Centered Anchors
    print("\n[Step 3: Contrastive Manifold Whitening & Mean-Voice Removal]")
    # Mean vector across all vocal samples captures vocal-tract bias
    stacked_all = torch.stack(all_recorded_vectors, dim=0)  # [12, 256]
    mean_voice_vector = stacked_all.mean(dim=0)  # [256]

    for cmd_name in cmd_raw_vectors:
        sample_mat = torch.stack(cmd_raw_vectors[cmd_name], dim=0)  # [3, 256]
        # Subtract mean voice vector to highlight pure phonetic differences
        centered_samples = sample_mat - mean_voice_vector.unsqueeze(0)
        norm_samples = F.normalize(centered_samples, p=2, dim=-1)

        # Average normalized centered vectors
        avg_anchor = F.normalize(norm_samples.mean(dim=0), p=2, dim=-1)
        anchor_profiles[cmd_name] = avg_anchor

        # Check intra-sample consistency
        sim_12 = F.cosine_similarity(norm_samples[0].unsqueeze(0), norm_samples[1].unsqueeze(0)).item()
        sim_23 = F.cosine_similarity(norm_samples[1].unsqueeze(0), norm_samples[2].unsqueeze(0)).item()
        avg_consistency = (sim_12 + sim_23) / 2.0
        print(f"  [Anchor Built] '{cmd_name}' consistency: {avg_consistency * 100:.1f}%")

    # Step 4: Check Separation Matrix
    print("\n[Step 4: Anchor Separation Matrix in 256D Manifold]")
    keys = list(anchor_profiles.keys())
    for i, k1 in enumerate(keys):
        row = []
        for j, k2 in enumerate(keys):
            sim = F.cosine_similarity(anchor_profiles[k1].unsqueeze(0), anchor_profiles[k2].unsqueeze(0)).item()
            row.append(f"{sim:.2f}")
        print(f"  {k1:18s} | " + "  ".join(row))

    # Step 5: Save Calibrated Vocal Anchors and VAD Settings
    os.makedirs("logs/training_runs", exist_ok=True)
    save_path = "logs/training_runs/my_voice_anchors.pt"
    torch.save({
        "anchors": anchor_profiles,
        "mean_voice_vector": mean_voice_vector,
        "ambient_noise_rms": noise_rms,
        "speech_threshold": speech_threshold
    }, save_path)
    print(f"\n[Success] Vocal calibration complete! Saved anchors to: {save_path}")
    print("Project Jane is now grounded to your acoustic vocal tract.")
    print("Run `python3 scripts/live_interactive_session.py` to start the live interactive embodiment!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Project Jane with your voice.")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic waveforms for automated testing.")
    parser.add_argument("--non-interactive", action="store_true", help="Run without pressing Enter.")
    args = parser.parse_args()

    calibrate_user_voice(interactive=not args.non_interactive, use_synthetic=args.synthetic)
