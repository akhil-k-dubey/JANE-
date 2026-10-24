"""Theory of Mind (ToM) Manifold: Self-Model (Ego), User-Model (Alter), and Causal Agency.

Implements Phase 5 Cognitive Architecture:
1. The Latent Self-Model (The Ego Node):
   Dedicated 256D cluster representing Jane's proprioceptive self-awareness.
   Every action taken by the Body writes back to Self_State, creating a self-referential observation loop.
2. The Akhil Model (The Alter Node):
   Dedicated 256D cluster tracking Akhil's affective and cognitive state
   (User_Calm, User_Learning, User_Teaching, User_Alarm, User_Affirming, User_Frustrated).
3. Causal Agency (Impact Modeling & Hebbian Triadic Super-Highways):
   Wires (Ego Action -> Environment Delta -> Alter Reaction) into reinforced causal pathways.
   Example: Jane_Vents_Gas -> Room_Pressure_Drops -> Akhil_Affirms.
"""

import os
import math
import time
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


EGO_STATES = [
    "Self_Idle",
    "Self_Navigating",
    "Self_Venting",
    "Self_SettingPower",
    "Self_PickingUp",
    "Self_Deliberating",
    "Self_Halted"
]

ALTER_STATES = [
    "User_Calm",
    "User_Learning",
    "User_Teaching",
    "User_Alarm",
    "User_Stress",
    "User_Affirming",
    "User_Frustrated"
]

ENV_DELTA_STATES = [
    "Env_Pressure_Drops",
    "Env_Door_Opened",
    "Env_Door_Locked",
    "Env_Item_Acquired",
    "Env_Power_Adjusted",
    "Env_Trajectory_Progress"
]


class TheoryOfMindManifold(nn.Module):
    """
    256D Theory of Mind Relational Manifold.
    Coordinates the Ego Cluster, the Alter (Akhil) Cluster, and Causal Agency Triads.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hebbian_rate: float = 0.05,
        decay_rate: float = 0.001,
        save_path: str = "logs/training_runs/tom_manifold.pt"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hebbian_rate = hebbian_rate
        self.decay_rate = decay_rate
        self.save_path = save_path

        # Combined node catalog
        self.all_nodes = EGO_STATES + ALTER_STATES + ENV_DELTA_STATES
        self.num_nodes = len(self.all_nodes)
        self.node_to_idx = {name: i for i, name in enumerate(self.all_nodes)}

        # Deterministic orthonormal initialization for 256D clusters
        torch.manual_seed(1337)
        base_coords = torch.randn(self.num_nodes, latent_dim)
        base_coords = F.normalize(base_coords, p=2, dim=-1)
        self.node_embeddings = nn.Parameter(base_coords, requires_grad=False)

        # Dynamic relational adjacency matrix W [num_nodes, num_nodes]
        # Base connectivity: low uniform prior + self-loops
        initial_w = torch.eye(self.num_nodes) * 0.5 + 0.05
        self.adjacency = nn.Parameter(initial_w, requires_grad=False)

        # Dynamic live state representations
        self.register_buffer("z_ego", torch.zeros(latent_dim))
        self.register_buffer("z_alter", torch.zeros(latent_dim))

        self.current_ego_label: str = "Self_Idle"
        self.current_alter_label: str = "User_Calm"
        self.last_action_taken: Optional[str] = None
        self.last_env_delta: Optional[str] = None

        # Causal Triad history log
        self.causal_history: List[Dict[str, Any]] = []
        self._ego_proto_vecs = None
        self._env_proto_vecs = None

        # Initialize baseline buffers
        self.z_ego.copy_(self.node_embeddings[self.node_to_idx["Self_Idle"]])
        self.z_alter.copy_(self.node_embeddings[self.node_to_idx["User_Calm"]])

        self.load_tom_manifold()

    # -------------------------------------------------------------------------
    # 1. The Latent Self-Model (The Ego Node)
    # -------------------------------------------------------------------------
    def update_ego_state(
        self,
        action_type: str,
        target: str = "",
        status: str = "EXECUTING"
    ) -> Dict[str, Any]:
        """
        Proprioceptive feedback loop: The Body writes to the Ego Node.
        Jane observes herself taking actions in real time.
        """
        with torch.no_grad():
            self.last_action_taken = action_type
            if status == "HALTED":
                ego_label = "Self_Halted"
            else:
                ego_label = self._resolve_ego_node(action_type)

            self.current_ego_label = ego_label
            target_vec = self.node_embeddings[self.node_to_idx[ego_label]]

            # Momentum blend into continuous z_ego
            self.z_ego.copy_(0.7 * self.z_ego + 0.3 * target_vec)
            self.z_ego.copy_(F.normalize(self.z_ego, p=2, dim=-1))

            return {
                "ego_label": ego_label,
                "action_type": action_type,
                "target": target,
                "status": status,
                "z_ego_norm": torch.norm(self.z_ego).item()
            }

    # -------------------------------------------------------------------------
    # 2. The Akhil Model (The Alter Node)
    # -------------------------------------------------------------------------
    def update_alter_state(
        self,
        audio_rms: float,
        acoustic_vector: Optional[torch.Tensor] = None,
        speech_text_or_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tracks Akhil's affective and cognitive state through the Passive Ear.
        Maps acoustic flux directly into User_State (Alter Node).
        """
        with torch.no_grad():
            from src.reasoning_engine.intent_manifold import TrueIntentManifold
            m = TrueIntentManifold.get_instance()
            text_str = speech_text_or_intent or ""

            sim_alarm = m.similarity(text_str, "emergency") if text_str else 0.0
            sim_affirm = m.similarity(text_str, "pleasantry") if text_str else 0.0
            sim_frust = m.similarity(text_str, "conversational_repair") if text_str else 0.0
            sim_teach = m.similarity(text_str, "active_teaching") if text_str else 0.0

            # Determine dominant Alter state via continuous manifold resonance
            if sim_alarm >= 0.55:
                alter_label = "User_Alarm" if audio_rms > 0.06 else "User_Stress"
            elif sim_affirm >= 0.55:
                alter_label = "User_Affirming"
            elif sim_frust >= 0.60:
                alter_label = "User_Frustrated"
            elif sim_teach >= 0.55 or audio_rms > 0.045:
                alter_label = "User_Teaching"
            elif audio_rms >= 0.025:
                alter_label = "User_Learning"
            else:
                alter_label = "User_Calm"

            self.current_alter_label = alter_label
            target_vec = self.node_embeddings[self.node_to_idx[alter_label]]

            # If acoustic vector provided, incorporate acoustic projection
            if acoustic_vector is not None:
                z_ac = F.normalize(acoustic_vector.squeeze(), p=2, dim=-1).to(self.z_alter.device)
                blended = 0.5 * target_vec + 0.5 * z_ac
            else:
                blended = target_vec

            self.z_alter.copy_(0.6 * self.z_alter + 0.4 * blended)
            self.z_alter.copy_(F.normalize(self.z_alter, p=2, dim=-1))

            return {
                "alter_label": alter_label,
                "audio_rms": audio_rms,
                "z_alter_norm": torch.norm(self.z_alter).item()
            }

    # -------------------------------------------------------------------------
    # 3. Metacognitive Conflict & Dissonance Index
    # -------------------------------------------------------------------------
    def compute_cognitive_dissonance(self) -> float:
        """
        Thread 4 Watchdog Metric:
        Quantifies dissonance between Jane's Ego State and Akhil's Alter State.
        High dissonance (> 0.70) signifies dangerous misalignment (e.g. moving while user is alarmed).
        """
        with torch.no_grad():
            ego_idx = self.node_to_idx[self.current_ego_label]
            alter_idx = self.node_to_idx[self.current_alter_label]

            # Inherent psychological conflict matrix
            # High risk: Jane is executing physical actions while Akhil is Alarmed or Stressed
            if self.current_alter_label in ("User_Alarm", "User_Stress"):
                if self.current_ego_label in ("Self_Navigating", "Self_Venting", "Self_SettingPower"):
                    return 0.95  # Critical emergency dissonance!
                elif self.current_ego_label == "Self_Deliberating":
                    return 0.75

            if self.current_alter_label == "User_Frustrated":
                if self.current_ego_label == "Self_Deliberating":
                    return 0.80  # Deliberating too long while user is frustrated!

            if self.current_alter_label == "User_Affirming":
                return 0.05  # Perfect cognitive harmony

            # General geometric distance between ego and alter in 256D
            sim = F.cosine_similarity(self.z_ego.unsqueeze(0), self.z_alter.unsqueeze(0), dim=-1).item()
            # Inverted: low similarity when expected alignment = higher dissonance
            dissonance = max(0.0, min(1.0, (1.0 - sim) * 0.4))
            return round(dissonance, 3)

    # -------------------------------------------------------------------------
    # 4. Causal Agency (Impact Modeling & Hebbian Triadic Super-Highways)
    # -------------------------------------------------------------------------
    def register_causal_triad(
        self,
        ego_action: str,
        env_delta: str,
        alter_reaction: str
    ) -> Dict[str, Any]:
        """
        Wires a Causal Super-Highway:
            Ego Action -> Environment Delta -> Alter Reaction
        Applies directional Hebbian reinforcement across the triadic graph.
        """
        with torch.no_grad():
            # Map strings to node names
            ego_node = self._resolve_ego_node(ego_action)
            env_node = self._resolve_env_node(env_delta)
            alter_node = self._resolve_alter_node(alter_reaction)

            i_ego = self.node_to_idx[ego_node]
            i_env = self.node_to_idx[env_node]
            i_alter = self.node_to_idx[alter_node]

            # 1. Wire Ego -> Env Delta (Sense of physical causation)
            w_ego_env_before = self.adjacency[i_ego, i_env].item()
            w_ego_env_after = min(1.0, (1.0 - self.decay_rate) * w_ego_env_before + self.hebbian_rate * 2.0)
            self.adjacency[i_ego, i_env] = w_ego_env_after
            self.adjacency[i_env, i_ego] = w_ego_env_after

            # 2. Wire Env Delta -> Alter Reaction (Understanding emotional impact of physical changes)
            w_env_alt_before = self.adjacency[i_env, i_alter].item()
            w_env_alt_after = min(1.0, (1.0 - self.decay_rate) * w_env_alt_before + self.hebbian_rate * 2.0)
            self.adjacency[i_env, i_alter] = w_env_alt_after
            self.adjacency[i_alter, i_env] = w_env_alt_after

            # 3. Transitive Super-Highway: Ego -> Alter (Direct agency attribution)
            w_direct_before = self.adjacency[i_ego, i_alter].item()
            w_direct_after = min(1.0, (1.0 - self.decay_rate) * w_direct_before + self.hebbian_rate * 1.5)
            self.adjacency[i_ego, i_alter] = w_direct_after
            self.adjacency[i_alter, i_ego] = w_direct_after

            triad_record = {
                "timestamp": time.time(),
                "ego_action": ego_node,
                "env_delta": env_node,
                "alter_reaction": alter_node,
                "ego_to_env_weight": round(w_ego_env_after, 3),
                "env_to_alter_weight": round(w_env_alt_after, 3),
                "superhighway_weight": round(w_direct_after, 3),
                "causal_sentence": f"{ego_node} caused {env_node}, which led to {alter_node}."
            }
            self.causal_history.append(triad_record)
            return triad_record

    def predict_alter_reaction(self, proposed_action: str) -> Tuple[str, float]:
        """
        Theory of Mind Forecasting:
        Given a proposed action, queries the causal super-highways to predict Akhil's reaction.
        """
        with torch.no_grad():
            ego_node = self._resolve_ego_node(proposed_action)
            i_ego = self.node_to_idx[ego_node]

            best_alter = "User_Calm"
            best_weight = -1.0

            for alt in ALTER_STATES:
                i_alt = self.node_to_idx[alt]
                w = self.adjacency[i_ego, i_alt].item()
                if w > best_weight:
                    best_weight = w
                    best_alter = alt

            return best_alter, round(best_weight, 3)

    def _init_semantic_prototypes(self):
        """Initializes continuous 256D semantic prototype embeddings for ego and env states."""
        if getattr(self, "_ego_proto_vecs", None) is not None:
            return
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()
        ego_prototypes = {
            "Self_Venting": "releasing pressure venting gas or valves exhaust",
            "Self_Navigating": "moving to location navigating traveling path motion",
            "Self_SettingPower": "adjusting reactor power electrical grid settings wattage",
            "Self_PickingUp": "picking up object manipulating physical item holding",
            "Self_Deliberating": "thinking planning reasoning calculating deliberating reflecting",
            "Self_Halted": "emergency stop system halted shutting down paused"
        }
        self._ego_proto_vecs = {k: m.encode(proto) for k, proto in ego_prototypes.items()}

        env_prototypes = {
            "Env_Pressure_Drops": "air pressure drops atmospheric decompression vacuum leak venting",
            "Env_Door_Locked": "door locked access barred sealed barrier",
            "Env_Door_Opened": "door opened airlock access cleared unlocked",
            "Env_Item_Acquired": "item object acquired obtained picked up tool collected",
            "Env_Power_Adjusted": "power adjusted electrical wattage voltage generator output modified",
            "Env_Trajectory_Progress": "path traversed trajectory distance progress forward motion"
        }
        self._env_proto_vecs = {k: m.encode(proto) for k, proto in env_prototypes.items()}

    def _resolve_ego_node(self, act: str) -> str:
        """Resolves an action string to an ego state node using continuous manifold projection."""
        if not act:
            return "Self_Idle"
        if act in EGO_STATES or act in self.all_nodes:
            return act
        self._init_semantic_prototypes()
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()
        qv = m.encode(act)
        best_state = "Self_Idle"
        best_sim = 0.35
        for state, proto_vec in self._ego_proto_vecs.items():
            sim = torch.dot(qv, proto_vec).item()
            if sim > best_sim:
                best_sim = sim
                best_state = state
        return best_state

    def _resolve_env_node(self, env: str) -> str:
        """Resolves an environment delta string to an env state node using continuous manifold projection."""
        if not env:
            return "Env_Trajectory_Progress"
        if env in ENV_DELTA_STATES or env in self.all_nodes:
            return env
        self._init_semantic_prototypes()
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()
        qv = m.encode(env)
        best_state = "Env_Trajectory_Progress"
        best_sim = 0.25
        for state, proto_vec in self._env_proto_vecs.items():
            sim = torch.dot(qv, proto_vec).item()
            if sim > best_sim:
                best_sim = sim
                best_state = state
        return best_state

    def _resolve_alter_node(self, alt: str) -> str:
        if alt in ALTER_STATES or alt in self.all_nodes:
            return alt

        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()

        sim_alarm = m.similarity(alt, "emergency")
        sim_affirm = m.similarity(alt, "pleasantry")
        sim_frust = m.similarity(alt, "conversational_repair")
        sim_teach = m.similarity(alt, "active_teaching")

        if sim_alarm >= 0.55:
            return "User_Alarm"
        elif sim_affirm >= 0.55:
            return "User_Affirming"
        elif sim_frust >= 0.60:
            return "User_Frustrated"
        elif sim_teach >= 0.55:
            return "User_Teaching"
        return "User_Calm"

    def get_superhighway_summary(self) -> List[str]:
        """Summarizes established causal super-highways in the manifold."""
        summaries = []
        for record in self.causal_history[-5:]:
            summaries.append(
                f"[{record['ego_action']} → {record['env_delta']} → {record['alter_reaction']}] (Link: {record['superhighway_weight']:.2f})"
            )
        return summaries

    def save_tom_manifold(self, path: Optional[str] = None):
        """Persists ToM manifold weights and causal graph."""
        target = path or self.save_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        torch.save({
            "node_embeddings": self.node_embeddings.data.cpu(),
            "adjacency": self.adjacency.data.cpu(),
            "z_ego": self.z_ego.cpu(),
            "z_alter": self.z_alter.cpu(),
            "causal_history": self.causal_history
        }, target)

    def load_tom_manifold(self, path: Optional[str] = None) -> bool:
        """Loads previously learned ToM manifold if present."""
        target = path or self.save_path
        if not os.path.exists(target):
            return False
        try:
            ckpt = torch.load(target, map_location=self.node_embeddings.device)
            if "node_embeddings" in ckpt and ckpt["node_embeddings"].shape == self.node_embeddings.shape:
                self.node_embeddings.data.copy_(ckpt["node_embeddings"].to(self.node_embeddings.device))
            if "adjacency" in ckpt and ckpt["adjacency"].shape == self.adjacency.shape:
                self.adjacency.data.copy_(ckpt["adjacency"].to(self.adjacency.device))
            if "z_ego" in ckpt:
                self.z_ego.copy_(ckpt["z_ego"].to(self.z_ego.device))
            if "z_alter" in ckpt:
                self.z_alter.copy_(ckpt["z_alter"].to(self.z_alter.device))
            if "causal_history" in ckpt:
                self.causal_history = ckpt["causal_history"]
            return True
        except Exception:
            return False
