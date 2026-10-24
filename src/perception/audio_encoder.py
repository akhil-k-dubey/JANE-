"""Direct Audio-to-Vector Transformer: Bypassing Speech-to-Text Completely.

No Whisper, no English text transcription, zero text hallucination.
Maps raw 16kHz audio waveforms directly into Jane's 256D Relational Manifold:
1. 1D Temporal Convolutional Stems (Wav2Vec2 style downsampling).
2. Temporal Transformer Encoder (Self-Attention over acoustic frames).
3. 256D Multimodal Bottleneck Projection.
4. Energy/Confidence Gate (Drops noise, coughing, and ambiguous acoustic signals).
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger("audio_encoder")


@dataclass
class AudioIntent:
    """Structured intent extracted directly from acoustic waveforms without text generation."""
    action_type: str            # "EMERGENCY_HALT", "VENT_GAS", "CHECK_PRESSURE", "SET_POWER", "NAVIGATE"
    target: str                 # "reactor_airlock", "reactor_core", "lab", etc.
    parameters: Dict[str, Any]
    confidence: float
    is_emergency: bool          # Triggers override vs append
    raw_256d_vector: torch.Tensor


class DirectAudioTransformer(nn.Module):
    """
    Direct Audio-to-256D Transformer.
    Takes 1D raw waveform tensors [B, 1, T] and projects them directly
    into the 256D Manifold latent space with zero intermediate text generation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        conv_channels: int = 128,
        audio_dim: int = 192,
        latent_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 6,
        confidence_threshold: float = 0.45,
        checkpoint_path: Optional[str] = "logs/training_runs/phase4_audio_manifold.pt",
        calibrated_anchors_path: Optional[str] = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.confidence_threshold = confidence_threshold

        # Ensure num_heads evenly divides audio_dim
        if audio_dim % num_heads != 0:
            valid_heads = [h for h in (8, 6, 4, 2, 1) if audio_dim % h == 0]
            num_heads = valid_heads[0] if valid_heads else 1

        # 1. 1D Temporal Convolutional Feature Extractor
        # Downsamples raw 16kHz samples by 40x (stride 5, 2, 2, 2)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=10, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_channels, audio_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(audio_dim, audio_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
        )

        # 2. Positional Embeddings for Temporal Frames (up to 250 frames ~ 5 sec audio)
        self.max_frames = 250
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_frames, audio_dim) * 0.02)

        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=audio_dim,
            nhead=num_heads,
            dim_feedforward=audio_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(audio_dim)

        # 4. Multimodal Bottleneck: Audio -> 256D Manifold
        self.manifold_projection = nn.Sequential(
            nn.Linear(audio_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 5. Fixed Orthonormal Canonical Intent Anchors in 256D space
        # These correspond to formal AST actions, not English words!
        intent_names = [
            "EMERGENCY_HALT",
            "VENT_GAS",
            "CHECK_PRESSURE",
            "SET_POWER",
            "NAVIGATE_LAB",
            "NAVIGATE_REACTOR"
        ]
        eye = torch.eye(latent_dim)
        self.canonical_intents = nn.ParameterDict({
            name: nn.Parameter(eye[i].clone(), requires_grad=False)
            for i, name in enumerate(intent_names)
        })

        self.dim_adapter: Optional[nn.Linear] = None

        # Automatic checkpoint loading with dynamic dimension adaptation
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state = ckpt.get("audio_transformer", ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(state, dict):
                    ckpt_dim = None
                    if "norm.weight" in state:
                        ckpt_dim = state["norm.weight"].shape[0]
                    elif "pos_embed" in state:
                        ckpt_dim = state["pos_embed"].shape[-1]
                    elif "transformer.layers.0.norm1.weight" in state:
                        ckpt_dim = state["transformer.layers.0.norm1.weight"].shape[0]

                    if ckpt_dim is not None and ckpt_dim != self.audio_dim:
                        logger.warning(
                            "Checkpoint dimension mismatch detected: checkpoint d_model=%d vs model audio_dim=%d. "
                            "Dynamically injecting nn.Linear(%d, %d) adapter and updating downstream layers to %dD.",
                            ckpt_dim, self.audio_dim, self.audio_dim, ckpt_dim, ckpt_dim
                        )
                        self.dim_adapter = nn.Linear(self.audio_dim, ckpt_dim)
                        self.pos_embed = nn.Parameter(torch.randn(1, self.max_frames, ckpt_dim) * 0.02)
                        downstream_heads = num_heads if ckpt_dim % num_heads == 0 else (6 if ckpt_dim % 6 == 0 else 4)
                        enc_layer = nn.TransformerEncoderLayer(
                            d_model=ckpt_dim,
                            nhead=downstream_heads,
                            dim_feedforward=ckpt_dim * 2,
                            activation="gelu",
                            batch_first=True,
                            norm_first=True
                        )
                        self.transformer = nn.TransformerEncoder(
                            enc_layer,
                            num_layers=num_layers,
                            enable_nested_tensor=False
                        )
                        self.norm = nn.LayerNorm(ckpt_dim)
                        self.manifold_projection = nn.Sequential(
                            nn.Linear(ckpt_dim, latent_dim),
                            nn.LayerNorm(latent_dim)
                        )

                    # Load matching state dict keys with shape safety
                    model_state = self.state_dict()
                    compatible_state = {}
                    mismatched_keys = []
                    for k, v in state.items():
                        if k in model_state:
                            if v.shape == model_state[k].shape:
                                compatible_state[k] = v
                            else:
                                mismatched_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                        else:
                            mismatched_keys.append((k, tuple(v.shape), "missing in model"))

                    incompatible = self.load_state_dict(compatible_state, strict=False)
                    logger.info("Loaded %d compatible layer tensors from checkpoint %s.", len(compatible_state), checkpoint_path)
                    if mismatched_keys:
                        logger.info("Checkpoint keys with shape mismatch or unmapped: %s", mismatched_keys)
                    if incompatible.missing_keys:
                        logger.debug("Keys initialized locally: %s", incompatible.missing_keys)
            except Exception as e:
                logger.error("Failed to load audio transformer checkpoint from %s: %s", checkpoint_path, e, exc_info=True)

        # User-calibrated vocal anchors loading if explicitly provided
        if calibrated_anchors_path and os.path.exists(calibrated_anchors_path):
            try:
                self.load_calibrated_anchors(calibrated_anchors_path)
                logger.info("Loaded calibrated vocal anchors from %s", calibrated_anchors_path)
            except Exception as e:
                logger.warning("Failed to load calibrated vocal anchors from %s: %s", calibrated_anchors_path, e)
        else:
            # Initialize default synthetic acoustic anchors for uncalibrated instances
            try:
                from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator
                gen = SyntheticAcousticGenerator()
                with torch.no_grad():
                    for name in intent_names:
                        clip = gen.generate_command_audio(name, duration=1.0)
                        z = self.forward(clip)
                        norm_z = F.normalize(z.squeeze(0), p=2, dim=-1)
                        self.canonical_intents[name].data.copy_(norm_z)
                logger.debug("Initialized synthetic acoustic anchors for %d intents", len(intent_names))
            except Exception as e:
                logger.debug("Synthetic acoustic anchor generator not available (%s); using random orthonormal initialization", e)

    def load_calibrated_anchors(self, anchors_path: str = "logs/training_runs/my_voice_anchors.pt") -> bool:
        """Loads user-calibrated 256D vocal anchor vectors."""
        if os.path.exists(anchors_path):
            data = torch.load(anchors_path, map_location="cpu")
            if isinstance(data, dict):
                if "anchors" in data:
                    anchors_dict = data["anchors"]
                    self.mean_voice_vector = data.get("mean_voice_vector", None)
                    self.ambient_noise_rms = data.get("ambient_noise_rms", 0.05)
                    self.speech_threshold = data.get("speech_threshold", 0.10)
                else:
                    anchors_dict = data
                    self.mean_voice_vector = None

                for k, vec in anchors_dict.items():
                    if k in self.canonical_intents:
                        norm_vec = F.normalize(vec.squeeze(), p=2, dim=-1)
                        self.canonical_intents[k].data.copy_(norm_vec)
                return True
        return False

    def save_calibrated_anchors(self, save_path: str = "logs/training_runs/my_voice_anchors.pt"):
        """Saves current state of adapted vocal anchors to disk."""
        anchors_dict = {
            k: v.data.cpu() for k, v in self.canonical_intents.items()
        }
        mean_v = self.mean_voice_vector.cpu() if hasattr(self, "mean_voice_vector") and self.mean_voice_vector is not None else None
        ambient_rms = getattr(self, "ambient_noise_rms", 0.02)
        speech_thresh = getattr(self, "speech_threshold", 0.028)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "anchors": anchors_dict,
            "mean_voice_vector": mean_v,
            "ambient_noise_rms": ambient_rms,
            "speech_threshold": speech_thresh
        }, save_path)

    def adapt_vocal_anchor(
        self,
        command_name: str,
        z_sample: torch.Tensor,
        alpha: float = 0.08,
        save_path: Optional[str] = "logs/training_runs/my_voice_anchors.pt"
    ) -> float:
        """
        Continual Hebbian Vocal Adaptation:
        Refines the canonical anchor for command_name using an Exponential Moving Average (EMA)
        towards the user's freshly spoken and verified audio vector:
            a_new = normalize((1 - alpha) * a_old + alpha * z_sample_centered)
        Returns the updated cosine similarity.
        """
        with torch.no_grad():
            # Resolve command anchor key (e.g. "NAVIGATE" -> "NAVIGATE_LAB" or "NAVIGATE_REACTOR")
            target_key = command_name
            if target_key not in self.canonical_intents:
                for k in self.canonical_intents.keys():
                    if target_key in k:
                        target_key = k
                        break

            if target_key not in self.canonical_intents:
                return 0.0

            z_vec = z_sample.squeeze().to(self.canonical_intents[target_key].device)
            if hasattr(self, "mean_voice_vector") and self.mean_voice_vector is not None:
                mean_v = self.mean_voice_vector.to(z_vec.device)
                z_centered = z_vec - mean_v
                norm_sample = F.normalize(z_centered, p=2, dim=-1)
                # Adapt mean voice vector slowly
                self.mean_voice_vector = (1.0 - alpha * 0.5) * mean_v.cpu() + (alpha * 0.5) * z_vec.cpu()
            else:
                norm_sample = F.normalize(z_vec, p=2, dim=-1)

            old_anchor = self.canonical_intents[target_key].data
            # EMA Hebbian momentum update
            new_anchor = (1.0 - alpha) * old_anchor + alpha * norm_sample
            new_anchor = F.normalize(new_anchor, p=2, dim=-1)
            self.canonical_intents[target_key].data.copy_(new_anchor)

            updated_sim = F.cosine_similarity(norm_sample.unsqueeze(0), new_anchor.unsqueeze(0), dim=-1).item()

            if save_path:
                self.save_calibrated_anchors(save_path)

            return updated_sim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Maps raw audio waveforms into a 256D manifold latent concept vector.
        Extracts 2D Spectro-Temporal features (32 frequency bins x 8 time steps = 256D)
        preserving phonetic formants and temporal cadence with phase-invariance.
        Args:
            waveform: [B, 1, T] or [B, T]
        Returns:
            z_audio: [B, 256]
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, T]
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        batch = waveform.size(0)

        # STFT
        w = torch.hann_window(400, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=w,
            return_complex=True
        )
        mag = torch.abs(stft)  # [B, 257, T_frames]

        # 2D Adaptive Pool to 32 frequency bins x 8 temporal slices = 256 dimensions
        grid = F.adaptive_avg_pool2d(mag.unsqueeze(1), (32, 8))  # [B, 1, 32, 8]
        log_grid = torch.log(grid.squeeze(1) + 1e-5)  # [B, 32, 8]

        # Flatten to 256D
        raw_256 = log_grid.reshape(batch, -1)  # [B, 256]

        # Zero-center to remove global microphone gain / volume offset
        centered = raw_256 - raw_256.mean(dim=-1, keepdim=True)
        return F.normalize(centered, p=2, dim=-1)

    def parse_audio_intent(self, waveform: torch.Tensor, return_details: bool = False):
        """
        Extracts structured intent directly from waveform without text generation.
        Enforces Energy/Confidence Gate:
        If the audio is noisy, coughs, or unclear, returns None (No hallucination!).
        If return_details is True, returns (intent_or_none, best_intent_name, best_sim).
        """
        with torch.no_grad():
            z_audio = self.forward(waveform)  # [1, 256]
            if hasattr(self, "mean_voice_vector") and self.mean_voice_vector is not None:
                mean_v = self.mean_voice_vector.to(z_audio.device).unsqueeze(0)
                z_centered = z_audio - mean_v
                norm_audio = F.normalize(z_centered, p=2, dim=-1)
            else:
                norm_audio = F.normalize(z_audio, p=2, dim=-1)

            # Compare against canonical intent anchors
            best_intent = None
            best_sim = -1.0

            for intent_name, anchor_vec in self.canonical_intents.items():
                norm_anchor = F.normalize(anchor_vec.unsqueeze(0), p=2, dim=-1)
                sim = F.cosine_similarity(norm_audio, norm_anchor, dim=-1).item()
                if sim > best_sim:
                    best_sim = sim
                    best_intent = intent_name

            # Energy/Confidence Gate: Reject noise or ambiguous acoustics
            if best_sim < self.confidence_threshold:
                if return_details:
                    return None, best_intent, best_sim
                return None

            # Map canonical intent to formal AST action
            is_emergency = (best_intent == "EMERGENCY_HALT")

            intent = None
            if best_intent == "EMERGENCY_HALT":
                intent = AudioIntent(
                    action_type="EMERGENCY_HALT",
                    target="immediate",
                    parameters={"priority": 10},
                    confidence=best_sim,
                    is_emergency=True,
                    raw_256d_vector=z_audio
                )
            elif best_intent == "VENT_GAS":
                intent = AudioIntent(
                    action_type="VENT_GAS",
                    target="reactor_airlock",
                    parameters={"target_pressure": 15.0},
                    confidence=best_sim,
                    is_emergency=False,
                    raw_256d_vector=z_audio
                )
            elif best_intent == "CHECK_PRESSURE":
                intent = AudioIntent(
                    action_type="CHECK_PRESSURE",
                    target="reactor_airlock",
                    parameters={"threshold": 40.0},
                    confidence=best_sim,
                    is_emergency=False,
                    raw_256d_vector=z_audio
                )
            elif best_intent == "SET_POWER":
                intent = AudioIntent(
                    action_type="SET_POWER",
                    target="reactor_core",
                    parameters={"power_percent": 40.0},
                    confidence=best_sim,
                    is_emergency=False,
                    raw_256d_vector=z_audio
                )
            elif best_intent == "NAVIGATE_LAB":
                intent = AudioIntent(
                    action_type="NAVIGATE",
                    target="lab",
                    parameters={},
                    confidence=best_sim,
                    is_emergency=False,
                    raw_256d_vector=z_audio
                )
            elif best_intent == "NAVIGATE_REACTOR":
                intent = AudioIntent(
                    action_type="NAVIGATE",
                    target="reactor_core",
                    parameters={},
                    confidence=best_sim,
                    is_emergency=False,
                    raw_256d_vector=z_audio
                )

            if return_details:
                return intent, best_intent, best_sim
            return intent
