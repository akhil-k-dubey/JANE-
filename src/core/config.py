"""Project Jane: Central Architecture & Hyperparameter Configuration.

Consolidates all magic numbers, thresholds, timing ticks, and dimensions across:
- 256D Continuous Latent Manifold & Hebbian Plasticity
- Theory of Mind (ToM) Cognitive Dissonance & Watchdog
- Multimodal Audio, Speech RMS, & Direct Acoustic Intent Gates
- Asynchronous Multithreaded Embodiment Loops & Timing Slices
- Pipeline & Symbolic Verification Gates
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ManifoldConfig:
    """Hyperparameters for continuous 256D concept manifold & Hebbian plasticity."""
    latent_dim: int = 256
    decay_rate: float = 0.001
    hebbian_rate: float = 0.04
    node_drift_rate: float = 0.015
    hebbian_weight_boost: float = 0.45
    seed_noise_scale: float = 0.10
    acoustic_firing_threshold: float = 0.15
    top_k_acoustic: int = 4
    save_path: str = "logs/training_runs/learned_voice_manifold.pt"


@dataclass
class TheoryOfMindConfig:
    """Thresholds for Theory of Mind (ToM) and Metacognitive Watchdog."""
    dissonance_critical_threshold: float = 0.70
    dissonance_elevated_threshold: float = 0.40
    watchdog_poll_interval_sec: float = 0.03
    audit_log_max_len: int = 500


@dataclass
class AudioConfig:
    """Parameters for audio processing, direct acoustic intent gating, and acoustics."""
    sample_rate: int = 16000
    chunk_size_samples: int = 16000  # 1.0 second
    audio_threshold_rms: float = 0.0280
    ambient_silence_rms: float = 0.020
    max_frames: int = 250  # ~3.1 seconds
    confidence_threshold: float = 0.40
    dropped_noise_sim_cutoff: float = 0.15
    autosave_interval_sec: float = 45.0


@dataclass
class EmbodimentConfig:
    """Timing slices and similarity thresholds for asynchronous embodiment threads."""
    similarity_gate: float = 0.15
    step_duration: float = 0.05
    idle_sleep_sec: float = 0.01
    slice_time_sec: float = 0.005
    queue_timeout_sec: float = 0.05


@dataclass
class PipelineConfig:
    """Settling steps, generation limits, and DNC memory slot parameters."""
    latent_steps_cap: int = 20
    max_gen_tokens: int = 64
    dnc_memory_slots: int = 32
    dnc_word_size: int = 64


@dataclass
class JaneConfig:
    """Master configuration container for Project Jane."""
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    tom: TheoryOfMindConfig = field(default_factory=TheoryOfMindConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# Global default configuration instance
DEFAULT_CONFIG = JaneConfig()
