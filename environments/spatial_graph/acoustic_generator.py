"""Synthetic Acoustic Waveform Generator for Testing Direct Audio Perception.

Synthesizes realistic 16kHz audio waveform tensors [1, T] with formant harmonics,
cadence envelopes, and background noise.
Allows testing audio embedding, noise gating, and real-time interruption without
requiring hardware microphone hardware.
"""

import math
from typing import Tuple
import torch


class SyntheticAcousticGenerator:
    """
    Generates 16kHz audio waveform tensors.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def _generate_harmonic_signal(
        self,
        duration: float,
        base_freq: float,
        formants: Tuple[float, float, float],
        envelope_type: str = "cadence"
    ) -> torch.Tensor:
        """Synthesizes speech-like harmonic signal with envelope."""
        num_samples = int(duration * self.sample_rate)
        t = torch.linspace(0, duration, num_samples)

        # Base pitch + formant harmonics
        signal = torch.sin(2 * math.pi * base_freq * t)
        for idx, f in enumerate(formants):
            amp = 0.5 / (idx + 1.0)
            signal += amp * torch.sin(2 * math.pi * f * t)

        # Envelope modulation
        if envelope_type == "sharp":
            # Sharp attack and decay (e.g. "STOP!")
            attack = int(0.05 * self.sample_rate)
            decay = num_samples - attack
            env = torch.cat([
                torch.linspace(0.0, 1.0, attack),
                torch.linspace(1.0, 0.1, decay)
            ])
        else:
            # Smooth speech cadence
            env = 0.5 * (1.0 - torch.cos(2 * math.pi * t / duration))

        audio = signal * env
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-6)
        return audio.unsqueeze(0)  # [1, num_samples]

    def generate_command_audio(self, command_type: str, duration: float = 1.0) -> torch.Tensor:
        """
        Generates synthetic speech waveforms for known commands:
        - 'EMERGENCY_HALT': High pitch, sharp envelope (850Hz, formants [1200, 2400, 3200])
        - 'VENT_GAS': Low-mid resonance (280Hz, formants [500, 1400, 2200])
        - 'CHECK_PRESSURE': Double cadence (340Hz, formants [700, 1600, 2600])
        - 'NAVIGATE_REACTOR': Prolonged sweep (310Hz, formants [650, 1500, 2400])
        """
        if command_type == "EMERGENCY_HALT":
            return self._generate_harmonic_signal(duration, 850.0, (1200.0, 2400.0, 3200.0), envelope_type="sharp")
        elif command_type == "VENT_GAS":
            return self._generate_harmonic_signal(duration, 280.0, (500.0, 1400.0, 2200.0), envelope_type="cadence")
        elif command_type == "CHECK_PRESSURE":
            return self._generate_harmonic_signal(duration, 340.0, (700.0, 1600.0, 2600.0), envelope_type="cadence")
        elif command_type == "NAVIGATE_REACTOR":
            return self._generate_harmonic_signal(duration, 310.0, (650.0, 1500.0, 2400.0), envelope_type="cadence")
        else:
            # Generic command
            return self._generate_harmonic_signal(duration, 400.0, (800.0, 1800.0, 2500.0), envelope_type="cadence")

    def generate_noise_audio(self, noise_type: str = "white", duration: float = 1.0) -> torch.Tensor:
        """
        Generates non-speech noise:
        - 'cough': Sharp non-harmonic burst
        - 'white': Random uniform noise (HVAC, fans)
        - 'hum': Pure low-frequency drone (60Hz AC)
        """
        num_samples = int(duration * self.sample_rate)
        if noise_type == "cough":
            burst_len = int(0.2 * self.sample_rate)
            noise_burst = torch.randn(burst_len) * 0.8
            full_noise = torch.zeros(num_samples)
            start = num_samples // 4
            full_noise[start:start+burst_len] = noise_burst
            return full_noise.unsqueeze(0)
        elif noise_type == "hum":
            t = torch.linspace(0, duration, num_samples)
            return (torch.sin(2 * math.pi * 60.0 * t) * 0.3).unsqueeze(0)
        else:
            # White noise
            return (torch.randn(num_samples) * 0.2).unsqueeze(0)
