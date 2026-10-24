"""Mode Router: Orchestrating Active Deliberation, Passive Eavesdropping, and Hybrid Reality.

Coordinates Jane's cognitive states:
- ACTIVE: Interactive text/voice conversation, System 2 equilibrium deliberation, Z3 formal proofs.
- PASSIVE: Background continuous eavesdropping, Hebbian manifold wiring, user vocal tone learning.
- HYBRID: Terminal chat in foreground + background passive ear continuously learning.
"""

import time
import queue
import threading
from enum import Enum
from typing import Optional, Dict, Any, Callable
import torch

from src.perception.audio_encoder import DirectAudioTransformer
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from scripts.live_mic_driver import LiveMicDriver
from src.utils.logger import get_logger

logger = get_logger("mode_router")


class OperatingMode(Enum):
    ACTIVE = "ACTIVE"
    PASSIVE = "PASSIVE"
    HYBRID = "HYBRID"


class ModeRouter:
    """
    Mode Router & Background Cognitive Manager.
    Manages background audio eavesdropping, continuous Hebbian learning,
    and seamless transitions between active problem solving and passive observation.
    """

    def __init__(
        self,
        mode: OperatingMode = OperatingMode.HYBRID,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        audio_threshold_rms: float = 0.0280,
        autosave_interval_sec: float = 45.0,
        on_flux_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.mode = mode
        self.device = torch.device(device)
        self.audio_threshold_rms = audio_threshold_rms
        self.autosave_interval_sec = autosave_interval_sec
        self.on_flux_callback = on_flux_callback

        # Perception & Learning Engines
        self.audio_encoder = DirectAudioTransformer(
            checkpoint_path="logs/training_runs/phase4_audio_manifold.pt",
            calibrated_anchors_path="logs/training_runs/my_voice_anchors.pt"
        ).to(self.device)
        self.audio_encoder.eval()

        self.hebbian_manifold = HebbianConceptManifold(latent_dim=256)

        # Background Thread Components
        self.mic_driver: Optional[LiveMicDriver] = None
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_listening = False
        self._last_save_time = time.time()

        # Telemetry metrics
        self.total_utterances_heard = 0
        self.last_flux_summary: Optional[Dict[str, Any]] = None

    def start_passive_listening(self):
        """Launches non-blocking background microphone listener and Hebbian learning thread."""
        if self._is_listening:
            return

        self._stop_event.clear()
        self.mic_driver = LiveMicDriver(
            sample_rate=16000,
            chunk_duration=1.5,
            hop_duration=0.5
        )
        try:
            self.mic_driver.start()
        except Exception as e:
            print(f"[ModeRouter] Notice: Audio capture unavailable ({e}). Continuing in text-only mode.")
            return

        self._is_listening = True
        self._background_thread = threading.Thread(
            target=self._passive_listener_loop,
            daemon=True,
            name="Jane-PassiveEar"
        )
        self._background_thread.start()

    def stop_passive_listening(self):
        """Stops the background eavesdropping thread and persists the learned manifold."""
        if not self._is_listening:
            return

        self._stop_event.set()
        self._is_listening = False

        if self.mic_driver:
            self.mic_driver.stop()
            self.mic_driver = None

        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=2.0)
            self._background_thread = None

        # Persist weights
        self.hebbian_manifold.save_manifold()

    def _passive_listener_loop(self):
        """
        Background loop: listens to user speech with intelligent VAD sleep gating.
        When silent: sleeps CPU/GPU to protect hardware.
        When user speaks/reads/lectures: maps acoustic flux into 256D concept manifold.
        """
        while not self._stop_event.is_set():
            if not self.mic_driver:
                break

            chunk = self.mic_driver.get_chunk(timeout=0.2)
            if chunk is None:
                time.sleep(0.02)
                continue

            # Intelligent VAD: Check RMS energy
            rms = torch.sqrt(torch.mean(chunk ** 2)).item()
            if rms < self.audio_threshold_rms:
                # Room ambient silence: sleep cleanly, zero GPU/CPU cost
                time.sleep(0.02)
                continue

            # Genuine speech detected!
            with torch.no_grad():
                waveform = chunk.to(self.device)
                z_audio = self.audio_encoder(waveform)  # [1, 256]

                # Update Hebbian manifold and vocal tone profile
                flux_telemetry = self.hebbian_manifold.process_acoustic_flux(
                    z_audio=z_audio,
                    audio_rms=rms,
                    top_k=4,
                    firing_threshold=0.12
                )
                self.total_utterances_heard += 1
                self.last_flux_summary = flux_telemetry

                # Dispatch callback if registered
                if self.on_flux_callback:
                    try:
                        self.on_flux_callback(flux_telemetry)
                    except Exception as e:
                        logger.error("Error executing on_flux_callback: %s", e, exc_info=True)

                # Periodic auto-save
                now = time.time()
                if now - self._last_save_time > self.autosave_interval_sec:
                    self.hebbian_manifold.save_manifold()
                    self._last_save_time = now

    def set_mode(self, new_mode: OperatingMode):
        """Switches Jane's operational mode."""
        self.mode = new_mode
        if new_mode in (OperatingMode.PASSIVE, OperatingMode.HYBRID):
            self.start_passive_listening()
        else:
            self.stop_passive_listening()

    @property
    def is_listening(self) -> bool:
        return self._is_listening
