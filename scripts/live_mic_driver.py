"""Non-Blocking Live Microphone Driver for Project Jane.

Continuously captures 16kHz audio waveforms in 1.0 - 1.5 second chunks in a background thread
and feeds them to a thread-safe queue for the Audio Encoder and Deliberation Engine.

Zero speech-to-text; produces raw 16kHz PyTorch tensors [1, T].
Supports:
1. Native ALSA / arecord streaming (Zero extra C-dependencies, rock-solid on Linux).
2. SoundDevice streaming (Cross-platform).
"""

import sys
import os
import time
import queue
import threading
import subprocess
from typing import Optional, Tuple
import numpy as np
import torch


class LiveMicDriver:
    """
    Non-blocking live microphone capture driver.
    Captures 16kHz mono audio and pushes [1, T] torch.Tensor chunks to audio_queue.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        hop_duration: float = 0.5,
        audio_queue: Optional[queue.Queue] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.hop_samples = int(sample_rate * hop_duration)
        self.audio_queue = audio_queue or queue.Queue()

        self.is_running = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self.backend = "unknown"

    def start(self):
        """Starts background non-blocking audio capture."""
        if self.is_running:
            return

        self._stop_event.clear()
        self.is_running = True

        # Test backend: Prefer arecord on Linux, fallback to sounddevice
        if self._check_arecord():
            self.backend = "arecord"
            self._worker_thread = threading.Thread(target=self._arecord_worker, daemon=True, name="MicReader-arecord")
        elif self._check_sounddevice():
            self.backend = "sounddevice"
            self._worker_thread = threading.Thread(target=self._sounddevice_worker, daemon=True, name="MicReader-sounddevice")
        else:
            raise RuntimeError("No working audio capture backend found (checked 'arecord' and 'sounddevice').")

        self._worker_thread.start()
        print(f"[LiveMicDriver] Background capture started via '{self.backend}' at {self.sample_rate}Hz (window: {self.chunk_duration}s, hop: {self.hop_duration}s).")

    def stop(self):
        """Stops live microphone capture."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=1.0)
            except Exception as e:
                print(f"[LiveMicDriver Debug] arecord termination: {e}")
            self._process = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)

        print("[LiveMicDriver] Microphone capture stopped.")

    def get_chunk(self, timeout: float = 0.5) -> Optional[torch.Tensor]:
        """Pulls an audio chunk [1, T] from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _check_arecord(self) -> bool:
        """Checks if arecord binary is available."""
        try:
            res = subprocess.run(["which", "arecord"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0
        except Exception:
            return False

    def _check_sounddevice(self) -> bool:
        """Checks if sounddevice is available and working."""
        try:
            import sounddevice as sd
            sd.check_input_settings()
            return True
        except Exception:
            return False

    def _arecord_worker(self):
        """Background thread streaming raw PCM bytes from arecord with rolling window."""
        bytes_per_sample = 2  # 16-bit PCM = 2 bytes
        bytes_per_hop = self.hop_samples * bytes_per_sample
        rolling_buffer = np.zeros(self.chunk_samples, dtype=np.float32)

        try:
            self._process = subprocess.Popen(
                ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            # Discard first 250ms power-on transient buffer pop
            warmup_bytes = int(0.25 * self.sample_rate * bytes_per_sample)
            self._process.stdout.read(warmup_bytes)
        except Exception as e:
            print(f"[LiveMicDriver ERROR] Failed to spawn arecord: {e}")
            self.is_running = False
            return

        hops_collected = 0
        min_hops = int(self.chunk_duration / self.hop_duration)

        while not self._stop_event.is_set():
            try:
                raw_bytes = self._process.stdout.read(bytes_per_hop)
                if not raw_bytes or len(raw_bytes) < bytes_per_hop:
                    time.sleep(0.02)
                    continue

                audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                new_hop = audio_int16.astype(np.float32) / 32768.0

                # Roll circular window
                rolling_buffer = np.roll(rolling_buffer, -self.hop_samples)
                rolling_buffer[-self.hop_samples:] = new_hop
                hops_collected += 1

                # Only emit once buffer is filled with live data
                if hops_collected >= min_hops:
                    tensor = torch.from_numpy(rolling_buffer.copy()).unsqueeze(0)
                    self.audio_queue.put(tensor)

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"[LiveMicDriver ERROR] arecord stream read error: {e}")
                break

    def _sounddevice_worker(self):
        """Background thread using sounddevice stream with rolling window."""
        import sounddevice as sd

        rolling_buffer = np.zeros(self.chunk_samples, dtype=np.float32)
        hops_collected = 0
        min_hops = int(self.chunk_duration / self.hop_duration)

        def callback(indata, frames, time_info, status):
            nonlocal rolling_buffer, hops_collected
            if status:
                print(f"[LiveMicDriver Warning] SoundDevice status: {status}")
            new_hop = indata[:, 0].copy()
            rolling_buffer = np.roll(rolling_buffer, -self.hop_samples)
            rolling_buffer[-self.hop_samples:] = new_hop
            hops_collected += 1
            if hops_collected >= min_hops:
                tensor = torch.from_numpy(rolling_buffer.copy()).unsqueeze(0)
                self.audio_queue.put(tensor)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.hop_samples,
                dtype="float32",
                callback=callback
            ):
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print(f"[LiveMicDriver ERROR] sounddevice stream error: {e}")


def test_live_mic():
    """CLI tester for live microphone."""
    print("=" * 60)
    print("PROJECT JANE — LIVE MICROPHONE DRIVER TEST")
    print("Listening for 5 seconds. Speak into your microphone!")
    print("=" * 60)

    driver = LiveMicDriver(sample_rate=16000, chunk_duration=1.0)
    try:
        driver.start()
        start_time = time.time()
        chunk_count = 0

        while time.time() - start_time < 5.0:
            chunk = driver.get_chunk(timeout=1.0)
            if chunk is not None:
                chunk_count += 1
                rms = torch.sqrt(torch.mean(chunk ** 2)).item()
                peak = torch.max(torch.abs(chunk)).item()
                bars = "#" * int(min(50, rms * 200))
                print(f"[Chunk {chunk_count:02d}] RMS: {rms:.4f} | Peak: {peak:.4f} | {bars}")

        print("\nTest completed successfully. Captured", chunk_count, "audio chunks.")
    finally:
        driver.stop()


if __name__ == "__main__":
    test_live_mic()
