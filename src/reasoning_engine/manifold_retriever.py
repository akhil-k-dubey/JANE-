"""Manifold Knowledge Retriever for Project Jane.

Retrieves verified conceptual triples and principles from the 256D manifold
and knowledge graph to ground LLaMA (The Mouth) in Jane's actual mind.
Ensures LLaMA only verbalizes grounded facts, preventing ungrounded speculation.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from src.utils.logger import get_logger

logger = get_logger("manifold_retriever")


@dataclass
class GroundedManifoldContext:
    """Structured facts retrieved from Jane's relational manifold."""
    matched_concepts: List[str] = field(default_factory=list)
    matched_domains: List[str] = field(default_factory=list)
    triples: List[Dict[str, str]] = field(default_factory=list)
    principles: List[str] = field(default_factory=list)
    is_known: bool = False
    is_epistemic_gap: bool = False
    unmapped_concept: Optional[str] = None

    def format_for_prompt(self) -> str:
        """Formats retrieved manifold knowledge as a concise grounding block for LLaMA."""
        lines = []
        if self.matched_domains:
            lines.append(f"Knowledge Domains: {', '.join(self.matched_domains)}")
        if self.matched_concepts:
            lines.append(f"Mapped Concepts: {', '.join(self.matched_concepts)}")
        if self.triples:
            trip_strs = [f"{t['head']} -> {t['relation']} -> {t['tail']}" for t in self.triples[:6]]
            lines.append("Relational Triples: " + "; ".join(trip_strs))
        if self.principles:
            lines.append("Verified Domain Principles: " + "; ".join(self.principles[:2]))
        return "\n".join(lines) if lines else "No specific manifold triples matched."


class ManifoldRetriever:
    """
    Retrieves grounded facts from Jane's 256D manifold and symbolic knowledge graph.
    Enforces the 'Mouth Only' constraint by feeding verified facts into LLaMA.
    """

    DEFAULT_TRIPLES_PATH = "data/knowledge_graph/concept_triples.json"
    DEFAULT_PRINCIPLES_PATH = "data/knowledge_graph/physics_principles.json"

    def __init__(
        self,
        triples_path: Optional[str] = None,
        principles_path: Optional[str] = None,
        concept_names: Optional[List[str]] = None
    ):
        self.triples_path = triples_path or self.DEFAULT_TRIPLES_PATH
        self.principles_path = principles_path or self.DEFAULT_PRINCIPLES_PATH
        self.concept_names: List[str] = list(concept_names or [])
        self.triples: List[Dict[str, str]] = []
        self.principles: List[Dict[str, str]] = []
        self._triples_index: Dict[str, List[Dict[str, str]]] = {}
        self._principles_index: Dict[str, List[Dict[str, str]]] = {}
        self._concepts_index: Dict[str, List[str]] = {}
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        self.intent_manifold = TrueIntentManifold.get_instance()
        self.load_data()

    def _rebuild_indices(self):
        """Builds sub-linear O(1) inverted indices for concepts, triples, and principles."""
        # 1. Concept index
        self._concepts_index = {}
        for c in self.concept_names:
            c_clean = c.lower().replace("_", " ")
            for w in c_clean.split():
                if len(w) > 2:
                    self._concepts_index.setdefault(w, []).append(c)
            self._concepts_index.setdefault(c_clean, []).append(c)

        # 2. Triples index
        self._triples_index = {}
        for t in self.triples:
            head_words = t.get("head", "").lower().replace("_", " ").split()
            tail_words = t.get("tail", "").lower().replace("_", " ").split()
            for w in set(head_words + tail_words):
                if len(w) > 2:
                    self._triples_index.setdefault(w, []).append(t)

        # 3. Principles index
        self._principles_index = {}
        for p in self.principles:
            c_words = p.get("concept", "").lower().replace("_", " ").split()
            r_words = p.get("relation", "").lower().replace("_", " ").split()
            for w in set(c_words + r_words):
                if len(w) > 2:
                    self._principles_index.setdefault(w, []).append(p)

    def load_data(self):
        """Loads relational triples and principles from disk and updates inverted indices."""
        if os.path.exists(self.triples_path):
            try:
                with open(self.triples_path, "r", encoding="utf-8") as f:
                    self.triples = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to parse triples JSON from %s: %s", self.triples_path, e)
                self.triples = []
            except Exception as e:
                logger.error("Unexpected error reading triples from %s: %s", self.triples_path, e, exc_info=True)
                self.triples = []

        if os.path.exists(self.principles_path):
            try:
                with open(self.principles_path, "r", encoding="utf-8") as f:
                    self.principles = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to parse principles JSON from %s: %s", self.principles_path, e)
                self.principles = []
            except Exception as e:
                logger.error("Unexpected error reading principles from %s: %s", self.principles_path, e, exc_info=True)
                self.principles = []

        self._rebuild_indices()

    def set_known_concepts(self, concepts: List[str]):
        """Updates list of known concept nodes from the live manifold and rebuilds indices."""
        self.concept_names = list(concepts)
        self._rebuild_indices()

    def extract_keywords(self, text: str) -> List[str]:
        """Extracts significant search keywords from user prompt without procedural stopword sets."""
        cleaned = "".join(c if (c.isalnum() or c == '_' or c.isspace()) else " " for c in text.lower())
        tokens = cleaned.split()
        return [tok for tok in tokens if len(tok) > 2]

    def retrieve(self, query: str) -> GroundedManifoldContext:
        """
        Retrieves matching concepts, triples, and principles for a user query.
        Uses sub-linear O(1) inverted indices for instant retrieval even with 10,000+ entries.
        Also detects if the query is an unmapped technical inquiry (epistemic gap).
        """
        keywords = self.extract_keywords(query)
        matched_concepts: Set[str] = set()
        matched_domains: Set[str] = set()
        matched_triples: List[Dict[str, str]] = []
        matched_principles: List[str] = []

        # 1. Match against registered manifold concepts via inverted index O(1)
        for kw in keywords:
            if kw in self._concepts_index:
                matched_concepts.update(self._concepts_index[kw])

        # 2. Match against relational triples via inverted index O(1)
        for kw in keywords:
            if kw in self._triples_index:
                for t in self._triples_index[kw]:
                    matched_triples.append(t)
                    matched_concepts.add(t.get("head", ""))
                    matched_concepts.add(t.get("tail", ""))

        # 3. Match against principles via inverted index O(1)
        for kw in keywords:
            if kw in self._principles_index:
                for p in self._principles_index[kw]:
                    matched_principles.append(p.get("statement", ""))
                    matched_concepts.add(p.get("concept", ""))
                    if p.get("domain"):
                        matched_domains.add(p["domain"])

        # Deduplicate triples
        unique_triples = []
        seen_triples = set()
        for t in matched_triples:
            key = (t.get("head"), t.get("relation"), t.get("tail"))
            if key not in seen_triples:
                seen_triples.add(key)
                unique_triples.append(t)

        # Meta-query: if user asks about concepts or knowledge in general
        if not matched_concepts and self.intent_manifold.similarity(query, "concepts_inventory") >= 0.65:
            matched_concepts.update(self.concept_names[:15])

        is_known = bool(matched_concepts or unique_triples or matched_principles)

        # 4. Check for Epistemic Gap:
        # STRICT RULE: ONLY trigger when user is genuinely inquiring for a technical/factual
        # definition or explanation ("what is X", "explain X", "define X", "how does X work")
        # AND the candidate concept is unmapped in Jane's manifold.
        # Conversational praise, idioms, banter, and statements MUST NEVER trigger an epistemic gap.
        is_epistemic_gap = False
        unmapped_concept = None
        lower_q = query.strip().lower()

        if not is_known and keywords:
            definitional_triggers = (
                "what is ", "what are ", "what's ", "explain ", "define ",
                "definition of ", "how does ", "how do ", "tell me about ", "meaning of "
            )
            has_def_trigger = any(lower_q.startswith(tr) or f" {tr}" in lower_q for tr in definitional_triggers)
            is_question = lower_q.endswith("?") or any(lower_q.startswith(w + " ") for w in ("what", "how", "why", "explain", "define"))

            sim_inquiry = self.intent_manifold.similarity(query, "interrogative_inquiry")
            is_epistemic_inquiry = has_def_trigger or (is_question and sim_inquiry >= 0.58)

            # Exclude self-inquiries, capabilities, pleasantries, greetings, repairs, or evaluations
            is_excluded = (
                self.intent_manifold.is_self_query(query)
                or max(
                    self.intent_manifold.similarity(query, "pleasantry"),
                    self.intent_manifold.similarity(query, "greeting"),
                    self.intent_manifold.similarity(query, "evaluative_scale"),
                    self.intent_manifold.similarity(query, "conversational_repair"),
                    self.intent_manifold.similarity(query, "capabilities"),
                    self.intent_manifold.similarity(query, "language_capability")
                ) >= 0.55
            )

            if is_epistemic_inquiry and not is_excluded:
                functional_stop = {
                    "what", "whats", "is", "are", "was", "were", "explain", "explaining",
                    "define", "defining", "tell", "me", "about", "how", "does", "do", "did",
                    "work", "works", "a", "an", "the", "can", "you", "please", "meaning", "of"
                }
                cand_tokens = [k.strip("?.,!") for k in keywords if k.lower().strip("?.,!") not in functional_stop]
                candidate = " ".join(cand_tokens) if cand_tokens else None
                if candidate and not self.intent_manifold.is_self_query(candidate):
                    is_epistemic_gap = True
                    unmapped_concept = candidate

        return GroundedManifoldContext(
            matched_concepts=sorted(list(matched_concepts)),
            matched_domains=sorted(list(matched_domains)),
            triples=unique_triples[:8],
            principles=matched_principles[:3],
            is_known=is_known,
            is_epistemic_gap=is_epistemic_gap,
            unmapped_concept=unmapped_concept
        )
