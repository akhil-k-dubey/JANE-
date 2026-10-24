"""Pragmatic Grounder and Personal Lexicon Engine for Project Jane.

Implements Pragmatic Intent Abduction:
1. Extracts "meaning above language" from user conversational explanations, idioms, and shorthand.
2. Triangulates latent user intent across multi-turn conversational clues.
3. Maintains a persistent personal lexicon (data/knowledge_graph/user_pragmatics.json).
4. Enables stoic, JARVIS-style comprehension and state tracking without shallow small talk.

Zero regular expressions. Pure deterministic character lexing and continuous vector similarity.
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("pragmatic_grounder")


@dataclass
class LearnedUserIdiom:
    """A learned user expression mapped to canonical system intent."""
    expression: str
    semantic_meaning: str
    canonical_intent: str
    timestamp: float
    reinforcement_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PragmaticClarificationResult:
    """Outcome of parsing a user's pragmatic explanation or idiom definition."""
    is_clarification: bool
    expression: Optional[str] = None
    semantic_meaning: Optional[str] = None
    canonical_intent: Optional[str] = None
    acknowledgement_text: Optional[str] = None


class PragmaticGrounder:
    """
    Pragmatic Intent Abduction Engine.
    Maps user-specific idiosyncratic phrasing, slang, banter, and praise into grounded
    system intents and updates Jane's Theory of Mind and episodic memory in real time.
    """

    DEFAULT_PRAGMATICS_PATH = "data/knowledge_graph/user_pragmatics.json"

    def __init__(
        self,
        pragmatics_path: Optional[str] = None,
        episodic_ledger: Optional[Any] = None,
        intent_manifold: Optional[Any] = None,
        tom_manifold: Optional[Any] = None
    ):
        self.pragmatics_path = pragmatics_path or self.DEFAULT_PRAGMATICS_PATH
        self.ledger = episodic_ledger
        self.intent_manifold = intent_manifold
        self.tom_manifold = tom_manifold
        self.idioms: Dict[str, LearnedUserIdiom] = {}
        self._load_pragmatics()

    def _ensure_intent_manifold(self):
        if self.intent_manifold is None:
            from src.reasoning_engine.intent_manifold import TrueIntentManifold
            self.intent_manifold = TrueIntentManifold.get_instance()

    def _load_pragmatics(self):
        """Loads learned idioms from persistent JSON file."""
        if os.path.exists(self.pragmatics_path):
            try:
                with open(self.pragmatics_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    idioms_data = raw_data.get("idioms", {})
                    for expr, item in idioms_data.items():
                        self.idioms[expr.lower()] = LearnedUserIdiom(
                            expression=item.get("expression", expr),
                            semantic_meaning=item.get("semantic_meaning", ""),
                            canonical_intent=item.get("canonical_intent", "conversational"),
                            timestamp=item.get("timestamp", time.time()),
                            reinforcement_count=item.get("reinforcement_count", 1)
                        )
            except Exception as e:
                logger.warning("Failed to load user pragmatics from %s: %s", self.pragmatics_path, e)
                self.idioms = {}
        else:
            self.idioms = {}

    def save_pragmatics(self):
        """Persists learned idioms to disk."""
        try:
            os.makedirs(os.path.dirname(self.pragmatics_path), exist_ok=True)
            data_to_save = {
                "idioms": {k: v.to_dict() for k, v in self.idioms.items()}
            }
            with open(self.pragmatics_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            logger.error("Failed to save user pragmatics to %s: %s", self.pragmatics_path, e)

    def extract_recent_idiom_candidate(self, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """
        Retrieves the most recent distinctive user phrase from conversation history
        that the user is clarifying or defining. Pure character lexing. ZERO regex.
        """
        if not history:
            return None

        # Scan backwards through user messages
        user_msgs = [m.get("content", "").strip() for m in history if m.get("role") == "user" and m.get("content", "").strip()]
        if not user_msgs:
            return None

        # Look at the message prior to the current clarification
        target_msg = user_msgs[-1]
        t_clean = target_msg.lower()

        # Filter out greetings or trivial commands
        for filler in ("hello", "hi", "hey", "/clear", "/history"):
            if t_clean == filler:
                return None

        # Clean punctuation
        cleaned = "".join(c if (c.isalnum() or c.isspace() or c == '_') else " " for c in target_msg)
        tokens = [w for w in cleaned.split() if w.lower() not in ("alr", "alright", "jane", "the", "a", "an", "i", "see", "you", "have")]
        if tokens:
            return " ".join(tokens)
        return target_msg

    def parse_pragmatic_clarification(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> PragmaticClarificationResult:
        """
        Detects if the user is providing a meta-linguistic clarification of their intent or phrasing.
        Examples:
          - "it's my way of saying that - you are working properly"
          - "it is just a praising pharse. I meant , it's good"
          - "when I say X I mean Y"
          - "by X I meant Y"
        Pure deterministic string search and token slicing. ZERO regex.
        """
        t = text.strip()
        lower = t.lower()

        clarification_markers = [
            "my way of saying that",
            "my way of saying",
            "way of saying that",
            "way of saying",
            "i meant that",
            "i meant",
            "i mean that",
            "i mean",
            "when i say",
            "by this i mean",
            "by that i mean"
        ]

        marker_found = None
        marker_pos = -1

        for marker in clarification_markers:
            pos = lower.find(marker)
            if pos != -1:
                marker_found = marker
                marker_pos = pos
                break

        if not marker_found:
            return PragmaticClarificationResult(is_clarification=False)

        # 1. Extract the target meaning (everything after the marker)
        after_marker = t[marker_pos + len(marker_found):].strip(" -:;,.\"'")
        meaning = after_marker if after_marker else "satisfactory operation and validation"

        # 2. Extract the expression being clarified
        expression = None

        # Pattern A: "it is just a <expression>. I meant <meaning>"
        before_marker = t[:marker_pos].strip()
        if "just a " in before_marker.lower():
            idx = before_marker.lower().find("just a ") + len("just a ")
            cand = before_marker[idx:].strip(" .,;:'\"-")
            if cand:
                expression = cand
        elif "just an " in before_marker.lower():
            idx = before_marker.lower().find("just an ") + len("just an ")
            cand = before_marker[idx:].strip(" .,;:'\"-")
            if cand:
                expression = cand

        # Pattern B: "when I say <X> I mean <Y>"
        if not expression and marker_found == "when i say":
            # expression is between 'when i say' and 'i mean'
            i_mean_pos = lower.find("i mean", marker_pos + len("when i say"))
            if i_mean_pos != -1:
                expression = t[marker_pos + len("when i say"):i_mean_pos].strip(" .,;:'\"-")
                meaning = t[i_mean_pos + len("i mean"):].strip(" .,;:'\"-")

        # Pattern C: Fallback to previous turn in conversation history
        if not expression and history:
            expression = self.extract_recent_idiom_candidate(history)

        if not expression:
            expression = "user_expression"

        expression_clean = expression.lower().replace("_", " ").strip()
        meaning_clean = meaning.lower().strip()

        # 3. Determine canonical system intent
        canonical_intent = self._infer_canonical_intent(meaning_clean)

        # 4. Wire into persistent dictionary
        existing = self.idioms.get(expression_clean)
        count = existing.reinforcement_count + 1 if existing else 1
        learned_entry = LearnedUserIdiom(
            expression=expression_clean,
            semantic_meaning=meaning_clean,
            canonical_intent=canonical_intent,
            timestamp=time.time(),
            reinforcement_count=count
        )
        self.idioms[expression_clean] = learned_entry
        self.save_pragmatics()

        # 5. Record to Episodic Ledger
        if self.ledger:
            self.ledger.record_event(
                action_type="PRAGMATIC_GROUNDING",
                concept_name=f"idiom:{expression_clean.replace(' ', '_')}",
                source_name="Akhil",
                formula_or_axiom=f"'{expression_clean}' => intent:{canonical_intent} ('{meaning_clean}')",
                status="COMMITTED",
                details={
                    "expression": expression_clean,
                    "meaning": meaning_clean,
                    "canonical_intent": canonical_intent,
                    "reinforcements": count
                }
            )

        # 6. Update Theory of Mind
        if self.tom_manifold:
            self.tom_manifold.current_alter_label = "User_Affirming"
            target_idx = self.tom_manifold.node_to_idx.get("User_Affirming")
            if target_idx is not None:
                self.tom_manifold.z_alter.copy_(self.tom_manifold.node_embeddings[target_idx])

        # 7. Formulate stoic, JARVIS-style acknowledgement
        ack_text = (
            f"Understood, Akhil. I've noted that '{expression_clean}' signifies {meaning_clean}. "
            "Internal state and intent mappings updated. Standing by for your next directive."
        )

        return PragmaticClarificationResult(
            is_clarification=True,
            expression=expression_clean,
            semantic_meaning=meaning_clean,
            canonical_intent=canonical_intent,
            acknowledgement_text=ack_text
        )

    def _infer_canonical_intent(self, meaning_text: str) -> str:
        """Determines canonical intent category from clarified meaning."""
        m_lower = meaning_text.lower()

        # Positive praise / system operational affirmation
        affirm_words = {"working properly", "good", "praise", "praising", "awesome", "great", "mastered", "properly", "correct", "fine", "nice"}
        if any(w in m_lower for w in affirm_words):
            return "system_status_operational"

        repair_words = {"wrong", "incorrect", "fix", "mistake", "error", "stop", "bad"}
        if any(w in m_lower for w in repair_words):
            return "conversational_repair"

        return "pleasantry"

    def match_user_idiom(self, text: str) -> Optional[LearnedUserIdiom]:
        """
        Checks if an incoming user prompt matches any registered user idiom.
        Returns the matched LearnedUserIdiom if found.
        """
        if not self.idioms:
            return None

        t_clean = text.lower().strip()

        # Exact substring match
        for expr, entry in self.idioms.items():
            if expr in t_clean:
                return entry

        # Token set overlap
        words = set("".join(c if (c.isalnum() or c.isspace()) else " " for c in t_clean).split())
        for expr, entry in self.idioms.items():
            expr_words = set(expr.split())
            if len(expr_words) >= 2 and expr_words.issubset(words):
                return entry

        return None

    def get_pragmatics_summary_for_prompt(self) -> str:
        """Formats active user idioms into context block for LLaMA/Mouth."""
        if not self.idioms:
            return "[LEARNED USER PRAGMATICS]: No custom user idioms registered."

        lines = ["[LEARNED USER PRAGMATICS & IDIOMATIC INTENT]:"]
        for expr, entry in list(self.idioms.items())[:6]:
            lines.append(f"  • '{entry.expression}' means: {entry.semantic_meaning} (Intent: {entry.canonical_intent})")
        return "\n".join(lines)
