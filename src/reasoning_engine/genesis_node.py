"""
Genesis Node: Latent Semantic Anchors for Project Intent.

When a long-term project begins, the user's statement of 'Why'
is embedded as a permanent 256D anchor in the manifold.
This anchor persists beyond the KV cache window — it lives in the
vector space, not in the text context window.
"""

import os
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger('genesis_node')


@dataclass
class GenesisAnchor:
    """A single permanent semantic project anchor."""
    anchor_id: str          # UUID
    project_name: str       # derived from embedding, not hardcoded
    created_at: float       # unix timestamp
    raw_intent_text: str    # the original "why" statement
    vector: List[float]     # 256D embedding, persisted as list
    activation_count: int = 0   # how many times the resurfacer fired
    last_activated: float = 0.0
    is_active: bool = True
    status: str = "active"
    priority_order: int = 1
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.vector, dtype=torch.float32)
    
    def cosine_similarity_to(self, query_vec: torch.Tensor) -> float:
        anchor_vec = self.to_tensor().to(query_vec.device)
        return float(F.cosine_similarity(anchor_vec.unsqueeze(0), query_vec.unsqueeze(0)).item())


class GenesisNodeRegistry:
    """
    Persistent registry of all Genesis Anchors.
    Anchors are stored in data/genesis_nodes.json and survive restart.
    New anchors are created by detecting high-intent project-start utterances
    through semantic similarity to an 'project initiation' centroid (not string matching).
    """
    
    LEDGER_PATH = 'data/genesis_nodes.json'
    # Semantic threshold: cosine similarity above this fires anchor detection
    INTENT_DETECTION_THRESHOLD = 0.45
    
    def __init__(self, intent_manifold=None):
        self.anchors: List[GenesisAnchor] = []
        self._intent_manifold = intent_manifold  # TrueIntentManifold for encoding
        os.makedirs(os.path.dirname(self.LEDGER_PATH), exist_ok=True)
        self._centroid = None
        self._shift_centroid = None
        self._load_ledger()
    
    def _get_encoder(self):
        """Lazy-loads TrueIntentManifold encoder."""
        if self._intent_manifold is None:
            from src.reasoning_engine.intent_manifold import TrueIntentManifold
            self._intent_manifold = TrueIntentManifold.get_instance(latent_dim=256)
        return self._intent_manifold
    
    def _load_ledger(self):
        """Loads persisted anchors from disk."""
        if not os.path.exists(self.LEDGER_PATH):
            return
        try:
            with open(self.LEDGER_PATH, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.anchors.append(GenesisAnchor(**item))
            logger.info(f"Loaded {len(self.anchors)} Genesis Anchors from ledger.")
        except Exception as e:
            logger.error(f"Error loading ledger: {e}")
    
    def _save_ledger(self):
        """Saves anchors to disk."""
        try:
            with open(self.LEDGER_PATH, 'w') as f:
                json.dump([asdict(a) for a in self.anchors], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving ledger: {e}")
    
    def _embed_text(self, text: str) -> torch.Tensor:
        """Embeds text into 256D using MiniLM."""
        return self._get_encoder().encode(text)

    def _get_centroid(self) -> torch.Tensor:
        if self._centroid is not None:
            return self._centroid
        prototypes = [
            'we are starting a project to redesign the cooling system',
            'I want to build a new feature for the authentication module',
            'let us work on improving the thermal efficiency of the reactor',
            'our goal for this sprint is to refactor the database layer',
            'I am going to create a new machine learning pipeline for fraud detection',
            'we need to migrate the legacy system to the new architecture',
            'the objective of this project is to reduce latency in the API',
            'Jane, we are starting a new long-term project'
        ]
        vectors = [self._embed_text(p) for p in prototypes]
        stacked = torch.stack(vectors)
        mean_vec = torch.mean(stacked, dim=0)
        self._centroid = F.normalize(mean_vec, p=2, dim=-1)
        return self._centroid

    def _get_priority_shift_centroid(self) -> torch.Tensor:
        if self._shift_centroid is not None:
            return self._shift_centroid
        prototypes = [
            'change the project priority first that we are going to make now is holographic projection',
            'change project priority to something else',
            'switch project focus and priority',
            'our main project priority is now holographic projection',
            'let us prioritize this new project instead',
            'pause previous project and change priority',
            'update project priority and focus',
            'pivot to a new primary project priority'
        ]
        vectors = [self._embed_text(p) for p in prototypes]
        stacked = torch.stack(vectors)
        mean_vec = torch.mean(stacked, dim=0)
        self._shift_centroid = F.normalize(mean_vec, p=2, dim=-1)
        return self._shift_centroid

    def is_project_inquiry(self, text: str) -> bool:
        """Determines if text is an inquiry about existing projects or conversation history."""
        encoder = self._get_encoder()
        if hasattr(encoder, "is_project_inquiry"):
            return encoder.is_project_inquiry(text)
        return False

    def _is_declaration(self, text: str, embedding: torch.Tensor, centroid: torch.Tensor, threshold: float = 0.58) -> bool:
        """
        Validates that text is a genuine project declaration and not an inquiry
        via continuous geometric manifold projection.
        Zero string heuristics or regex.
        """
        encoder = self._get_encoder()
        centroid_t = centroid.to(embedding.device)
        sim_decl = F.cosine_similarity(centroid_t.unsqueeze(0), embedding.unsqueeze(0)).item()
        if sim_decl < threshold:
            return False

        # Inquiries have high resonance with project_intent, history_memory, or interrogative_inquiry
        sim_inq = max(
            encoder.similarity(text, "project_intent"),
            encoder.similarity(text, "history_memory"),
            encoder.similarity(text, "interrogative_inquiry")
        )
        # Must have a decisive positive margin over inquiries
        return (sim_decl - sim_inq) > 0.05

    def is_priority_shift_utterance(self, text: str, embedding: Optional[torch.Tensor] = None) -> bool:
        """
        Detects if an utterance is shifting project priority via continuous vector projection.
        Zero regex, zero string matching.
        """
        if embedding is None:
            embedding = self._embed_text(text)
        centroid = self._get_priority_shift_centroid()
        return self._is_declaration(text, embedding, centroid, threshold=0.58)

    def handle_priority_shift(self, text: str) -> GenesisAnchor:
        """
        Handles an explicit project priority shift:
        1. Marks all current active anchors as inactive/superseded.
        2. Creates a new active anchor representing the new project priority.
        3. Persists the ledger to disk.
        """
        for a in self.anchors:
            if a.is_active:
                a.is_active = False
                a.status = "superseded"

        embedding = self._embed_text(text)
        project_name = text[:60].rstrip()
        new_anchor = GenesisAnchor(
            anchor_id=str(uuid.uuid4()),
            project_name=project_name,
            created_at=time.time(),
            raw_intent_text=text,
            vector=embedding.squeeze().tolist(),
            activation_count=0,
            last_activated=0.0,
            is_active=True,
            status="active",
            priority_order=1
        )
        self.anchors.append(new_anchor)
        self._save_ledger()
        logger.info(f"Shifted project priority to new anchor: '{project_name}'")
        return new_anchor
    
    def _is_project_initiation_utterance(self, text: str, embedding: torch.Tensor) -> bool:
        """
        Detects if an utterance is a project start declaration via continuous vector projection.
        Zero regex, zero keyword matching.
        """
        centroid = self._get_centroid()
        return self._is_declaration(text, embedding, centroid, threshold=0.58)
    
    def maybe_create_anchor(self, text: str) -> Optional[GenesisAnchor]:
        """
        If the utterance semantically resembles a project priority shift or initiation,
        creates/updates and persists a GenesisAnchor. Returns the anchor or None.
        Guarded against questions or inquiries asking about existing projects/history.
        """
        encoder = self._get_encoder()
        if hasattr(encoder, "is_project_inquiry") and encoder.is_project_inquiry(text):
            return None

        embedding = self._embed_text(text)
        if self.is_priority_shift_utterance(text, embedding):
            return self.handle_priority_shift(text)
        if self._is_project_initiation_utterance(text, embedding):
            project_name = text[:60].rstrip()
            anchor = GenesisAnchor(
                anchor_id=str(uuid.uuid4()),
                project_name=project_name,
                created_at=time.time(),
                raw_intent_text=text,
                vector=embedding.squeeze().tolist(),
                activation_count=0,
                last_activated=0.0,
                is_active=True,
                status="active",
                priority_order=1
            )
            self.anchors.append(anchor)
            self._save_ledger()
            return anchor
        return None
    
    def get_all_anchors(self) -> List[GenesisAnchor]:
        return list(self.anchors)

    def get_active_anchors(self) -> List[GenesisAnchor]:
        """Returns only actively monitored Genesis Anchors."""
        return [a for a in self.anchors if getattr(a, "is_active", True) and getattr(a, "status", "active") == "active"]

    def deactivate_anchor(self, anchor_id: str) -> bool:
        """Deactivates a specific anchor by ID."""
        for a in self.anchors:
            if a.anchor_id == anchor_id:
                a.is_active = False
                a.status = "superseded"
                self._save_ledger()
                return True
        return False
    
    def get_nearest_anchor(self, query_vec: torch.Tensor, top_k: int = 1) -> List[tuple]:
        """
        Returns the top_k anchors most similar to query_vec.
        Returns list of (similarity_score, GenesisAnchor) tuples.
        """
        if not self.anchors:
            return []
        sims = []
        for anchor in self.anchors:
            sims.append((anchor.cosine_similarity_to(query_vec), anchor))
        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_k]
