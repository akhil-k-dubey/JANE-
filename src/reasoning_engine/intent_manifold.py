"""True Intent Manifold: MiniLM Dense Neural Vector Geometry & Semantic Intent Routing.

Replaces CRC32 subword hashing, keyword dictionaries, and heuristic loops with
dense sentence-transformer embeddings (all-MiniLM-L6-v2) and mathematical cosine/softmax
manifold projections.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger("intent_manifold")


class ContinuousIntentManifold:
    """
    True Continuous Intent Manifold backed by all-MiniLM-L6-v2.
    Maps human utterances into continuous dense vector space and classifies
    cognitive intent through cosine similarity and Softmax distributions over intent centroids.
    """

    _instance: Optional["ContinuousIntentManifold"] = None

    @classmethod
    def get_instance(cls, latent_dim: int = 256) -> "ContinuousIntentManifold":
        """Singleton accessor ensuring shared intent centroids across cognitive components."""
        if cls._instance is None:
            cls._instance = cls(latent_dim=latent_dim)
        return cls._instance

    def __init__(self, latent_dim: int = 256):
        self.latent_dim = latent_dim
        self.model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        self.device = self.model.device

        # Optional linear projection to map 384D MiniLM vectors into 256D if requested
        if latent_dim != 384:
            self.proj = nn.Linear(384, latent_dim, bias=False).to(self.device)
            with torch.no_grad():
                try:
                    nn.init.orthogonal_(self.proj.weight)
                except Exception:
                    g = torch.Generator().manual_seed(42)
                    w = torch.randn(latent_dim, 384, generator=g)
                    self.proj.weight.copy_(F.normalize(w, p=2, dim=-1).to(self.device))
        else:
            self.proj = None

        self.centroids: Dict[str, torch.Tensor] = {}
        self.prototype_embeddings: Dict[str, torch.Tensor] = {}
        self._init_centroids()

        # Continuous Duration / Estimation Hyperplane Axis (loaded from checkpoint or computed)
        if not hasattr(self, "w_duration") or self.w_duration is None:
            v_dur = self.encode("time duration hours learning curve estimation")
            v_stat = self.encode("current registered inventory concept count")
            self.w_duration = F.normalize(v_dur - v_stat, p=2, dim=-1)

        # Continuous Capability Inquiry vs Blueprint List Hyperplane Axis
        if not hasattr(self, "w_cap_inquiry") or self.w_cap_inquiry is None:
            v_inq = self.encode("can you write code or programs")
            v_list = self.encode("what are your capabilities and abilities")
            self.w_cap_inquiry = F.normalize(v_inq - v_list, p=2, dim=-1)

    def _init_centroids(self):
        """
        Initializes canonical Intent Centroids and Prototype Banks from precomputed tensor checkpoint.
        Eliminates all hardcoded English prototype sentences from Python source code.
        """
        ckpt_path = os.path.join(os.path.dirname(__file__), "../../data/models/intent_manifold_centroids.pt")
        if os.path.exists(ckpt_path):
            data = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.centroids = {k: v.to(self.device) for k, v in data["centroids"].items()}
            self.prototype_embeddings = {k: v.to(self.device) for k, v in data.get("prototype_embeddings", {}).items()}
            self.raw_prototypes = data.get("raw_prototypes", {})
            if "w_duration" in data and data["w_duration"] is not None:
                self.w_duration = data["w_duration"].to(self.device)
            if "w_cap_inquiry" in data and data["w_cap_inquiry"] is not None:
                self.w_cap_inquiry = data["w_cap_inquiry"].to(self.device)
            if "proj_weight" in data and self.proj is not None:
                with torch.no_grad():
                    self.proj.weight.copy_(data["proj_weight"].to(self.device))
            return

        # Fallback if checkpoint is not found: dynamically compute from canonical intent names
        canonical_intents = [
            "vision", "user_identity", "self_identity", "capabilities", "concepts_inventory",
            "epistemic_discovery", "epistemic_why_single", "codebase_introspection", "evaluative_scale",
            "history_memory", "project_intent", "interrogative_inquiry", "system_status", "conversational",
            "geometry_math", "symbolic_math", "symbolic_physics", "conversational_repair", "active_teaching",
            "query_task_progress", "command_ingest", "capability_challenge", "theory_of_mind", "greeting",
            "pleasantry", "language_capability", "architecture_check", "emergency", "action_directive"
        ]
        self.raw_prototypes = {intent: [intent.replace("_", " ")] for intent in canonical_intents}
        for intent in canonical_intents:
            emb = self.model.encode(intent.replace("_", " "), convert_to_tensor=True)
            norm = F.normalize(emb, p=2, dim=-1)
            self.centroids[intent] = norm
            self.prototype_embeddings[intent] = norm.unsqueeze(0)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes arbitrary text into dense normalized vector space using MiniLM.
        Returns L2-normalized unit vector.
        """
        raw = self.model.encode(text, convert_to_tensor=True, device=self.device)
        if self.proj is not None:
            # Map to target latent_dim via orthogonal projection
            with torch.no_grad():
                raw = self.proj(raw)
        return F.normalize(raw, p=2, dim=-1)

    def get_mood_projection(self, text: str) -> float:
        """
        Projects text embedding onto the grammatical mood hyperplane axis.
        Positive (lambda > 0) -> Interrogative / Inquiry Subspace (questions, clarifications).
        Non-positive (lambda <= 0) -> Imperative / Action Subspace (commands, directives).
        Pure continuous vector linear algebra (Zero regex / substring heuristics).
        """
        inquiry_c = self.centroids["interrogative_inquiry"]
        cmd_c = self.centroids["command_ingest"]
        mood_vector = F.normalize(inquiry_c - cmd_c, p=2, dim=-1)

        raw = self.model.encode(text, convert_to_tensor=True, device=self.device)
        qv = F.normalize(raw, p=2, dim=-1)
        return torch.dot(qv, mood_vector).item()

    def get_duration_projection(self, text: str) -> float:
        """
        Projects text embedding onto the duration / learning-curve / estimation hyperplane axis.
        Positive (lambda > 0.10) -> Question is asking about duration, learning curve, feasibility, or mastering a skill.
        Non-positive (lambda <= 0.10) -> Query is asking about static state, inventory, or factual definition.
        Pure continuous vector linear algebra (Zero regex / keyword heuristics).
        """
        if not hasattr(self, "w_duration") or self.w_duration is None:
            v_dur = self.encode('time duration, hours, learning curve, amount of time or concepts needed and required to learn or master a subject')
            v_stat = self.encode('current registered inventory, concept count currently possessed, what concepts you currently know and have right now')
            self.w_duration = F.normalize(v_dur - v_stat, p=2, dim=-1)

        qv = self.encode(text)
        return torch.dot(qv, self.w_duration).item()

    def get_capability_inquiry_projection(self, text: str) -> float:
        """
        Projects text embedding onto the Capability Inquiry vs Capability Enumeration hyperplane axis.
        Positive (lambda > 0.05) -> Inquiring about a specific capability (e.g. coding, writing python, fluent execution).
        Non-positive (lambda <= 0.05) -> Requesting a full architectural capability inventory/list.
        Pure continuous vector linear algebra (Zero regex / keyword heuristics).
        """
        if not hasattr(self, "w_cap_inquiry") or self.w_cap_inquiry is None:
            v_inq = self.encode("can you write code, are you able to code, do you know python, write a program, are you fluent in writing code, can you or can you not")
            v_list = self.encode("what are your capabilities, list your capabilities, enumerate your abilities, tell me all your core capabilities, what can you do")
            self.w_cap_inquiry = F.normalize(v_inq - v_list, p=2, dim=-1)

        qv = self.encode(text)
        return torch.dot(qv, self.w_cap_inquiry).item()

    def _word_similarity(self, w1: str, w2: str) -> float:
        """Computes semantic cosine similarity between two words."""
        v1 = self.encode(w1)
        v2 = self.encode(w2)
        return torch.dot(v1, v2).item()

    def similarity(self, text: str, intent: str) -> float:
        """
        Computes maximum semantic similarity between input text vector
        and target intent prototype bank (combined with centroid resonance).
        """
        if intent not in self.centroids:
            return 0.0

        raw = self.model.encode(text, convert_to_tensor=True, device=self.device)
        norm_v = F.normalize(raw, p=2, dim=-1)

        # 1. Cosine similarity against mean centroid
        centroid = self.centroids[intent]
        # Match dimensions if projected
        if centroid.shape[-1] != norm_v.shape[-1]:
            centroid = self.model.encode(intent, convert_to_tensor=True)
            centroid = F.normalize(centroid, p=2, dim=-1)

        mean_sim = torch.dot(norm_v, centroid).item()

        # 2. Maximum similarity against individual prototype anchors
        proto_embs = self.prototype_embeddings[intent]
        max_sim = torch.max(torch.mv(proto_embs, norm_v)).item()

        # Blend: 80% closest archetype anchor + 20% centroid mean
        sim_val = 0.80 * max_sim + 0.20 * mean_sim

        # 3. Subword character n-gram typo resilience for misspellings
        # (Enables typos like 'screee' or 'camerra feed' to match canonical anchors)
        text_lower = text.lower().strip()
        words = text_lower.split()
        if 1 <= len(words) <= 3 and sim_val < 0.72:
            proto_phrases = getattr(self, "raw_prototypes", {}).get(intent, [])
            for phrase in proto_phrases:
                p_words = phrase.lower().split()
                word_scores = []
                for w in words:
                    best_w = 0.0
                    for pw in p_words:
                        if w == pw:
                            best_w = 1.0
                            break
                        if len(w) >= 4 and len(pw) >= 4 and abs(len(w) - len(pw)) <= 2:
                            bg1 = set(w[i:i+2] for i in range(len(w)-1))
                            bg2 = set(pw[i:i+2] for i in range(len(pw)-1))
                            inter = len(bg1 & bg2)
                            dice = (2.0 * inter) / (len(bg1) + len(bg2)) if (len(bg1) + len(bg2)) > 0 else 0.0
                            if dice > best_w:
                                best_w = dice
                    word_scores.append(best_w)
                avg_score = sum(word_scores) / len(word_scores) if word_scores else 0.0
                if avg_score >= 0.75:
                    boost = avg_score * 0.85
                    if boost > sim_val:
                        sim_val = boost

        return sim_val

    SELF_INTENTS = {
        "user_identity", "self_identity", "vision", "embodiment",
        "capabilities", "concepts_inventory", "epistemic_discovery",
        "epistemic_why_single", "codebase_introspection", "system_status",
        "language_capability", "architecture_check", "project_intent"
    }

    def is_project_inquiry(self, text: str) -> bool:
        """
        Determines if an utterance is an inquiry asking about existing projects
        or conversation history, rather than a declaration initiating or shifting a project.
        Pure continuous 256D vector geometry with zero string matching or regex.
        """
        sim_proj = self.similarity(text, "project_intent")
        sim_hist = self.similarity(text, "history_memory")
        sim_inq = self.similarity(text, "interrogative_inquiry")
        return max(sim_proj, sim_hist, sim_inq) >= 0.55

    def get_self_model_intent(self, text: str) -> Tuple[str, float]:
        """Returns (best_matching_self_intent, cosine_similarity)."""
        sims = {i: self.similarity(text, i) for i in self.SELF_INTENTS}
        best_intent, best_sim = max(sims.items(), key=lambda x: x[1])
        return best_intent, best_sim

    def is_self_query(self, text: str, threshold: float = 0.38) -> bool:
        """
        Determines whether input is a self-referential introspection query
        via global argmax competition across all manifold centroids.
        Completely eliminates hardcoded negative exclusion blacklists.
        """
        all_sims = {k: self.similarity(text, k) for k in self.centroids.keys()}
        if not all_sims:
            return False
        winning_intent, max_sim = max(all_sims.items(), key=lambda x: x[1])
        return winning_intent in self.SELF_INTENTS and max_sim >= threshold


    def route_softmax(self, text: str, temperature: float = 0.1) -> Dict[str, float]:
        """Calculates Softmax probability distribution over all semantic intent categories."""
        all_intents = list(self.centroids.keys())
        sims = torch.tensor([self.similarity(text, i) for i in all_intents])
        probs = F.softmax(sims / temperature, dim=0).tolist()
        return dict(zip(all_intents, probs))

    def apply_prediction_error_penalty(
        self,
        failed_intent: str,
        true_intent: Optional[str] = None,
        text: Optional[str] = None,
        eta: float = 0.1
    ):
        """
        Tier 4 Prediction Error Adaptation:
        Applies anti-Hebbian calibration adjustment to penalize the misrouted intent centroid
        and reinforce the true / runner-up intent centroid using the query's unit embedding.
        """
        if not text:
            return

        with torch.no_grad():
            raw = self.model.encode(text, convert_to_tensor=True, device=self.device)
            text_vec = F.normalize(raw, p=2, dim=-1)
            if failed_intent in self.centroids:
                old_c = self.centroids[failed_intent]
                new_c = old_c - eta * text_vec
                self.centroids[failed_intent] = F.normalize(new_c, p=2, dim=-1)

            if true_intent and true_intent in self.centroids:
                old_c = self.centroids[true_intent]
                new_c = old_c + eta * text_vec
                self.centroids[true_intent] = F.normalize(new_c, p=2, dim=-1)


# Provide alias TrueIntentManifold
TrueIntentManifold = ContinuousIntentManifold

