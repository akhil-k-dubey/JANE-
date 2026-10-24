"""Hebbian Relational Concept Manifold: Continuous Acoustic-Semantic Mapping.

Enables Jane to passively listen to the user (talking to themselves, reading books,
teaching out loud) and wire concepts together in a 256D topological manifold
without intermediate speech-to-text.

Principles:
1. Zero Speech-to-Text: Projects raw acoustic flux directly into 256D concept coordinates.
2. Hebbian Plasticity ("Neurons that fire together wire together"):
   Reinforces relational edge weights W_ij between co-occurring concepts.
3. Adaptive Node Drift:
   Attracts concept centroids toward the user's acoustic timbre, cadence, and vocal tract.
4. Continuous Tone Profiler:
   Monitors fundamental cadence, vocal energy dynamics, and classifies speech modality
   (Reading Aloud, Teaching/Explaining, Casual Monologue, Focused Inquiry).
"""

import os
import math
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.config import DEFAULT_CONFIG, ManifoldConfig


CORE_CONCEPTS = [
    # Physics & Energy
    "energy", "entropy", "momentum", "force", "pressure",
    "temperature", "work", "power", "flux", "equilibrium",
    "conservation", "thermodynamics", "velocity", "gravity",
    # Mathematics & Logic
    "equation", "derivative", "integral", "limit", "geometry",
    "probability", "ratio", "dimension", "symmetry", "topology",
    # Cognitive & Monologue
    "self_reflection", "hypothesis", "verification", "curiosity",
    "structure", "deduction", "axiom", "causality",
    # Tone / Modality Anchors
    "teaching_explaining", "calm_reading", "focused_inquiry", "assertive"
]


class HebbianConceptManifold(nn.Module):
    """
    Continuous 256D Concept Manifold with Hebbian Plasticity and Tone Profiling.
    Tracks live acoustic flux and maps conceptual associations over time.
    """

    def __init__(
        self,
        latent_dim: Optional[int] = None,
        concept_names: Optional[List[str]] = None,
        decay_rate: Optional[float] = None,
        hebbian_rate: Optional[float] = None,
        node_drift_rate: Optional[float] = None,
        save_path: Optional[str] = None,
        config: Optional[ManifoldConfig] = None
    ):
        super().__init__()
        cfg = config or DEFAULT_CONFIG.manifold
        self.latent_dim = latent_dim if latent_dim is not None else cfg.latent_dim
        self.concept_names = list(concept_names or CORE_CONCEPTS)
        self.num_concepts = len(self.concept_names)
        self.decay_rate = decay_rate if decay_rate is not None else cfg.decay_rate
        self.hebbian_rate = hebbian_rate if hebbian_rate is not None else cfg.hebbian_rate
        self.node_drift_rate = node_drift_rate if node_drift_rate is not None else cfg.node_drift_rate
        self.save_path = save_path or cfg.save_path

        # Concept index lookup
        self.name_to_idx = {name: i for i, name in enumerate(self.concept_names)}

        # Concept node coordinates [num_concepts, 256] initialized deterministically
        torch.manual_seed(42)
        base_vectors = torch.randn(self.num_concepts, self.latent_dim)
        base_vectors = F.normalize(base_vectors, p=2, dim=-1)
        self.concept_nodes = nn.Parameter(base_vectors, requires_grad=False)

        # Dynamic relational adjacency matrix W [num_concepts, num_concepts]
        # Represents topological edge connections between concepts
        initial_adj = torch.eye(self.num_concepts) * 0.5 + 0.05
        self.adjacency = nn.Parameter(initial_adj, requires_grad=False)

        # User Vocal Tone Profile (256D centroid + running statistics)
        self.register_buffer("user_tone_centroid", torch.zeros(self.latent_dim))
        self.register_buffer("user_tone_variance", torch.ones(self.latent_dim) * 0.1)
        self.total_utterances_observed = 0
        self.last_fired_concepts: List[str] = []
        self.last_tone_label = "ambient"
        self._manifold_lock = threading.Lock()

        # Attempt to load previously learned manifold if it exists
        self.load_manifold()

    def get_vector(self, concept_name: str) -> torch.Tensor:
        """Retrieves 256D vector for a concept by name."""
        with torch.no_grad():
            key = concept_name.lower().replace(" ", "_")
            if key in self.name_to_idx:
                return self.concept_nodes[self.name_to_idx[key]].clone()

            # Partial / substring match
            for name, idx in self.name_to_idx.items():
                if key in name or name in key:
                    return self.concept_nodes[idx].clone()

            # Concept not yet present: derive a deterministic seed projection vector
            torch.manual_seed(abs(hash(key)) % (2**31 - 1))
            seed_vec = torch.randn(self.latent_dim, device=self.concept_nodes.device)
            return F.normalize(seed_vec, p=2, dim=-1)

    def find_nearest_concepts(
        self,
        query_vector: torch.Tensor,
        top_k: int = 3,
        exclude: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Computes cosine similarities between query_vector and all concept nodes in the 256D manifold,
        returning top_k nearest concept names and their similarity scores.
        """
        with torch.no_grad():
            if self.num_concepts == 0:
                return []

            q = query_vector.squeeze().to(self.concept_nodes.device)
            if q.dim() > 1:
                q = q[0]
            q_norm = F.normalize(q, p=2, dim=-1)

            sims = F.cosine_similarity(q_norm.unsqueeze(0), self.concept_nodes, dim=-1)

            exclude_set = set(e.lower().replace(" ", "_") for e in (exclude or []))

            # Retrieve top candidates (request more than top_k to account for exclusions)
            k_candidates = min(self.num_concepts, top_k + len(exclude_set) + 4)
            top_vals, top_indices = torch.topk(sims, k=k_candidates)

            results: List[Tuple[str, float]] = []
            for val, idx in zip(top_vals, top_indices):
                c_name = self.concept_names[idx.item()]
                if c_name in exclude_set:
                    continue
                results.append((c_name, float(val.item())))
                if len(results) >= top_k:
                    break
            return results

    def find_nearest_concepts_for_text(
        self,
        text: str,
        top_k: int = 3,
        exclude: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Derives an embedding for arbitrary text from known words or deterministic hashes,
        and retrieves the top_k nearest concepts on the manifold.
        """
        tokens = []
        curr = []
        for ch in text.lower():
            if ch.isalnum() or ch == '_':
                curr.append(ch)
            else:
                if curr:
                    tokens.append("".join(curr))
                    curr = []
        if curr:
            tokens.append("".join(curr))
        words = [w for w in tokens if len(w) > 2]
        if not words:
            return self.find_nearest_concepts(self.concept_nodes[0], top_k=top_k, exclude=exclude)

        vectors = [self.get_vector(w) for w in words]
        centroid = torch.mean(torch.stack(vectors), dim=0)
        return self.find_nearest_concepts(centroid, top_k=top_k, exclude=exclude)

    def add_node(self, concept_name: str, vector: torch.Tensor) -> int:
        """
        Adds or updates a concept node coordinate in the 256D manifold space.
        Thread-safe under concurrent background ingestion and conversational auto-fetch.
        """
        with self._manifold_lock:
            with torch.no_grad():
                key = concept_name.lower().replace(" ", "_")
                norm_vec = F.normalize(vector.squeeze().to(self.concept_nodes.device), p=2, dim=-1)

                if key in self.name_to_idx:
                    idx = self.name_to_idx[key]
                    self.concept_nodes[idx].copy_(norm_vec)
                    return idx

                # Append new concept
                old_nodes = self.concept_nodes.data
                old_n = old_nodes.shape[0]
                idx = old_n

                self.concept_names.append(key)
                self.name_to_idx[key] = idx
                self.num_concepts = len(self.concept_names)

                # Expand concept_nodes [N+1, 256]
                new_nodes = torch.cat([old_nodes, norm_vec.unsqueeze(0)], dim=0)
                self.concept_nodes = nn.Parameter(new_nodes, requires_grad=False)

                # Expand adjacency matrix [N+1, N+1]
                old_adj = self.adjacency.data
                old_h, old_w = old_adj.shape
                new_size = self.num_concepts
                new_adj = torch.full(
                    (new_size, new_size),
                    0.05,
                    device=old_adj.device,
                    dtype=old_adj.dtype
                )
                new_adj[:old_h, :old_w] = old_adj
                new_adj[idx, idx] = 0.55  # self-loop
                self.adjacency = nn.Parameter(new_adj, requires_grad=False)

                return idx

    def reinforce_connection(self, concept_a: str, concept_b: str, weight_boost: float = 0.40) -> float:
        """
        Hebbian Plasticity: Permanently reinforces relational connection between two concepts.
        """
        key_a = concept_a.lower().replace(" ", "_")
        key_b = concept_b.lower().replace(" ", "_")

        if key_a not in self.name_to_idx:
            vec_a = self.get_vector(key_a)
            self.add_node(key_a, vec_a)

        if key_b not in self.name_to_idx:
            vec_b = self.get_vector(key_b)
            self.add_node(key_b, vec_b)

        with self._manifold_lock:
            with torch.no_grad():
                idx_a = self.name_to_idx[key_a]
                idx_b = self.name_to_idx[key_b]

                old_w = self.adjacency[idx_a, idx_b].item()
                new_w = min(1.0, old_w + weight_boost)
                self.adjacency[idx_a, idx_b] = new_w
                self.adjacency[idx_b, idx_a] = new_w

                return new_w

    def process_acoustic_flux(
        self,
        z_audio: torch.Tensor,
        audio_rms: float,
        top_k: int = 4,
        firing_threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Processes a single acoustic flux vector from the user's voice.
        Applies Hebbian wiring, node drift, and updates vocal tone profile.

        Args:
            z_audio: [1, 256] or [256] raw acoustic vector from DirectAudioTransformer
            audio_rms: Root-Mean-Square volume of the chunk
            top_k: Number of co-occurring concepts to wire
            firing_threshold: Minimum cosine similarity to trigger concept firing

        Returns:
            Telemetry dict detailing fired concepts, edge adjustments, and tone.
        """
        with torch.no_grad():
            z = z_audio.squeeze().to(self.concept_nodes.device)
            z_norm = F.normalize(z, p=2, dim=-1)

            # 1. Update User Vocal Tone Profile (Running EMA)
            alpha_tone = 0.05 if self.total_utterances_observed < 50 else 0.015
            if self.total_utterances_observed == 0 or torch.norm(self.user_tone_centroid) < 1e-4:
                self.user_tone_centroid.copy_(z_norm)
            else:
                self.user_tone_centroid.copy_(
                    (1.0 - alpha_tone) * self.user_tone_centroid + alpha_tone * z_norm
                )
                self.user_tone_centroid.copy_(F.normalize(self.user_tone_centroid, p=2, dim=-1))

            # Tone modality classification based on acoustic dynamics
            tone_label = self._classify_tone(z_norm, audio_rms)
            self.last_tone_label = tone_label
            self.total_utterances_observed += 1

            # 2. Compute similarity to all concept nodes in 256D
            sims = F.cosine_similarity(z_norm.unsqueeze(0), self.concept_nodes, dim=-1)  # [num_concepts]

            # Shift similarities relative to average to find distinct active nodes
            top_values, top_indices = torch.topk(sims, k=top_k)

            fired_concepts: List[Tuple[str, float]] = []
            fired_idx_list = []
            for val, idx in zip(top_values, top_indices):
                v_item = val.item()
                if v_item >= firing_threshold:
                    c_name = self.concept_names[idx.item()]
                    fired_concepts.append((c_name, v_item))
                    fired_idx_list.append(idx.item())

            # 3. Hebbian Learning: Wire co-firing concepts together
            # W_ij = (1 - decay) * W_ij + rate * (s_i * s_j)
            reinforced_edges: List[Tuple[str, str, float, float]] = []
            if len(fired_idx_list) >= 2:
                for i_pos in range(len(fired_idx_list)):
                    for j_pos in range(i_pos + 1, len(fired_idx_list)):
                        i_idx = fired_idx_list[i_pos]
                        j_idx = fired_idx_list[j_pos]
                        s_i = sims[i_idx].item()
                        s_j = sims[j_idx].item()

                        old_w = self.adjacency[i_idx, j_idx].item()
                        hebbian_delta = self.hebbian_rate * (s_i * s_j)
                        new_w = min(1.0, (1.0 - self.decay_rate) * old_w + hebbian_delta)

                        self.adjacency[i_idx, j_idx] = new_w
                        self.adjacency[j_idx, i_idx] = new_w

                        reinforced_edges.append((
                            self.concept_names[i_idx],
                            self.concept_names[j_idx],
                            old_w,
                            new_w
                        ))

            # 4. Adaptive Node Drift: Pull fired nodes slightly towards user's acoustic vector
            for c_idx in fired_idx_list:
                s_val = sims[c_idx].item()
                old_node = self.concept_nodes[c_idx]
                drift_step = self.node_drift_rate * s_val * (z_norm - old_node)
                new_node = F.normalize(old_node + drift_step, p=2, dim=-1)
                self.concept_nodes[c_idx].copy_(new_node)

            self.last_fired_concepts = [c[0] for c in fired_concepts]

            return {
                "utterance_count": self.total_utterances_observed,
                "audio_rms": audio_rms,
                "tone_label": tone_label,
                "fired_concepts": fired_concepts,
                "reinforced_edges": reinforced_edges,
                "centroid_norm": torch.norm(self.user_tone_centroid).item()
            }

    def _classify_tone(self, z_norm: torch.Tensor, rms: float) -> str:
        """Classifies user's vocal tone modality."""
        if rms < 0.030:
            return "Whispering / Quiet Reflection"
        elif rms > 0.080:
            return "Emphatic / Teaching Loudly"

        # Compare similarity to tone anchors if present
        best_tone = "Explaining / Reading"
        tone_anchors = ["teaching_explaining", "calm_reading", "focused_inquiry", "assertive"]
        best_sim = -1.0
        for t in tone_anchors:
            if t in self.name_to_idx:
                idx = self.name_to_idx[t]
                s = F.cosine_similarity(z_norm.unsqueeze(0), self.concept_nodes[idx].unsqueeze(0)).item()
                if s > best_sim:
                    best_sim = s
                    best_tone = t.replace("_", " ").title()

        return best_tone

    def get_strongest_links(self, top_n: int = 8) -> List[Tuple[str, str, float]]:
        """Returns the top N strongest wired concept pairs in the manifold."""
        with torch.no_grad():
            adj_np = self.adjacency.clone()
            # Zero out diagonal
            adj_np.fill_diagonal_(0.0)
            rows, cols = torch.triu_indices(self.num_concepts, self.num_concepts, offset=1)
            values = adj_np[rows, cols]
            top_vals, top_k_indices = torch.topk(values, k=min(top_n, len(values)))

            links = []
            for val, k_idx in zip(top_vals, top_k_indices):
                r = rows[k_idx].item()
                c = cols[k_idx].item()
                links.append((self.concept_names[r], self.concept_names[c], round(val.item(), 3)))
            return links

    def save_manifold(self, path: Optional[str] = None):
        """Persists the learned concept manifold and user vocal tone profile."""
        target = path or self.save_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        torch.save({
            "concept_nodes": self.concept_nodes.data.cpu(),
            "adjacency": self.adjacency.data.cpu(),
            "concept_names": self.concept_names,
            "user_tone_centroid": self.user_tone_centroid.cpu(),
            "total_utterances": self.total_utterances_observed
        }, target)

    def load_manifold(self, path: Optional[str] = None) -> bool:
        """Loads previously learned concept manifold if available."""
        target = path or self.save_path
        if not os.path.exists(target):
            return False
        try:
            ckpt = torch.load(target, map_location=self.concept_nodes.device)
            if "concept_nodes" in ckpt and ckpt["concept_nodes"].shape == self.concept_nodes.shape:
                self.concept_nodes.data.copy_(ckpt["concept_nodes"].to(self.concept_nodes.device))
            if "adjacency" in ckpt and ckpt["adjacency"].shape == self.adjacency.shape:
                self.adjacency.data.copy_(ckpt["adjacency"].to(self.adjacency.device))
            if "user_tone_centroid" in ckpt:
                self.user_tone_centroid.copy_(ckpt["user_tone_centroid"].to(self.user_tone_centroid.device))
            if "total_utterances" in ckpt:
                self.total_utterances_observed = ckpt["total_utterances"]
            return True
        except Exception:
            return False

    def find_sparse_cluster(self) -> str:
        """
        Tier 1 Autonomous Curiosity:
        Identifies the most sparse or unrepresented knowledge cluster in the 256D manifold.
        Scans frontier scientific domains first; if all represented, locates the concept node
        with the lowest graph degree/connectivity in the topological manifold.
        """
        frontier_domains = [
            "laser", "holography", "superconductivity", "quantum_computing",
            "black_hole_thermodynamics", "general_relativity", "quantum_entanglement",
            "graphene", "optics", "fluid_dynamics", "astrophysics",
            "thermodynamics", "neural_networks", "electromagnetism", "nuclear_fusion"
        ]
        # 1. Frontier void check (unmapped domains)
        for dom in frontier_domains:
            dom_norm = dom.lower().replace(" ", "_")
            if dom_norm not in self.name_to_idx:
                return dom

        # 2. Topological graph connectivity check (lowest degree)
        with torch.no_grad():
            adj_n = self.adjacency[:self.num_concepts, :self.num_concepts]
            degrees = adj_n.sum(dim=-1)
            sparse_idx = torch.argmin(degrees).item()
            return self.concept_names[sparse_idx]

    def check_density(
        self,
        latent_vector: Optional[torch.Tensor] = None,
        threshold: float = 0.55,
        text: Optional[str] = None
    ) -> List[str]:
        """
        Tier 2 Contextual Auto-Fetch Trigger:
        Evaluates whether user utterance or latent vector references concepts that are
        sparse, unrepresented, or low-density in the 256D manifold.
        """
        sparse_hits: List[str] = []

        # 1. Semantic keyword density inspection from user text
        if text:
            # Extract content words (> 3 chars, letters and underscores)
            raw_toks = []
            curr_tok = []
            for ch in text.lower():
                if ch.isalpha() or ch == '_':
                    curr_tok.append(ch)
                else:
                    if curr_tok:
                        raw_toks.append("".join(curr_tok))
                        curr_tok = []
            if curr_tok:
                raw_toks.append("".join(curr_tok))
            tokens = [t for t in raw_toks if len(t) >= 4]
            stopwords = {
                "this", "that", "with", "from", "have", "were", "what", "when",
                "where", "which", "could", "would", "should", "their", "there",
                "about", "using", "design", "make", "create", "build", "system",
                "let's", "lets", "tell", "explain", "think", "please", "flux",
                "jane", "hello", "project", "want", "help", "need"
            }
            content_words = [t for t in tokens if t not in stopwords]
            for w in content_words:
                w_singular = w.rstrip("s") if w.endswith("s") and len(w) > 4 else w
                if w not in self.name_to_idx and w_singular not in self.name_to_idx:
                    # Concept is not mapped into Jane's manifold!
                    target = w_singular if len(w_singular) >= 4 else w
                    if target not in sparse_hits:
                        sparse_hits.append(target)

        # 2. Geometric continuous latent density inspection
        if latent_vector is not None and not sparse_hits:
            with torch.no_grad():
                z = latent_vector.squeeze().to(self.concept_nodes.device)
                if z.dim() == 1:
                    z_norm = F.normalize(z, p=2, dim=-1)
                    sims = F.cosine_similarity(z_norm.unsqueeze(0), self.concept_nodes[:self.num_concepts], dim=-1)
                    max_sim = sims.max().item()
                    if max_sim < threshold:
                        # Vector is located in a sparse geometric void
                        sparse_hits.append(self.find_sparse_cluster())

        return sparse_hits

    def ingest_text(self, raw_text: str, concept_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Wires raw text knowledge natively into the 256D space using Hebbian plasticity.
        """
        with torch.no_grad():
            c_name = concept_name
            if not c_name:
                raw_words = []
                curr_w = []
                for ch in raw_text.lower():
                    if ch.isalpha() or ch == '_':
                        curr_w.append(ch)
                    else:
                        if curr_w:
                            raw_words.append("".join(curr_w))
                            curr_w = []
                if curr_w:
                    raw_words.append("".join(curr_w))
                words = [t for t in raw_words if len(t) >= 4]
                c_name = words[0] if words else "novel_concept"

            c_name = c_name.lower().replace(" ", "_")
            # If new concept, add node
            torch.manual_seed(abs(hash(raw_text)) % (2**31 - 1))
            pseudo_vec = torch.randn(self.latent_dim, device=self.concept_nodes.device)
            pseudo_vec = F.normalize(pseudo_vec, p=2, dim=-1)

            if c_name not in self.name_to_idx:
                idx = self.add_node(c_name, pseudo_vec)
            else:
                idx = self.name_to_idx[c_name]
                # Hebbian blend with existing node
                self.concept_nodes[idx].copy_(
                    F.normalize(0.7 * self.concept_nodes[idx] + 0.3 * pseudo_vec, p=2, dim=-1)
                )

            # Reinforce self-loop and adjacent connections
            self.adjacency[idx, idx] = min(1.0, self.adjacency[idx, idx].item() + self.hebbian_rate)

            return {
                "concept": c_name,
                "text_length": len(raw_text),
                "node_norm": torch.norm(self.concept_nodes[idx]).item()
            }
