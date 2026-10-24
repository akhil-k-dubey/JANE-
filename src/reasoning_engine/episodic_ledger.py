"""Episodic Memory Ledger and Meta-Cognitive Provenance Tracker for Project Jane.

Maintains an immutable, time-indexed episodic action and belief stream:
1. Every discovery, ingestion, proof, and active learning step records an EpisodicEvent.
2. Tracks full source provenance: [arXiv], [Wikipedia], [CuriosityLoop], [UserConversation].
3. Grounds Jane's self-awareness dynamically without brittle hardcoded canned strings:
   When asked about her state, discoveries, or memory, she inspects her live ledger.
"""

import time
import json
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("episodic_ledger")


@dataclass
class EpisodicEvent:
    """A discrete episodic event representing an action taken or belief learned by Jane."""
    timestamp: float
    action_type: str        # "DISCOVERY", "INGEST_ARXIV", "INGEST_WIKIPEDIA", "TEACH_USER", "PROVE_SMT"
    concept_name: str
    source_name: str        # e.g. "arXiv:quant-ph/0306072", "Wikipedia:Ideal_gas_law", "CuriosityLoop", "Akhil"
    formula_or_axiom: str
    status: str             # "COMMITTED", "FALSIFIED", "REJECTED", "VERIFIED"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Formatted human-readable event description."""
        time_str = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return (
            f"[{time_str}] Action={self.action_type} | Source={self.source_name} | "
            f"Concept='{self.concept_name}' | Formula='{self.formula_or_axiom}' | Status={self.status}"
        )


class EpisodicLedger:
    """
    Persistent Episodic Memory Ledger for dynamic Meta-Cognition and Provenance Tracking.
    Stores episodic events to logs/episodic_ledger.jsonl and provides dynamic self-reflection.
    Capped ring buffer avoids memory leaks while preserving critical violations in an unbounded list.
    """

    DEFAULT_LOG_PATH = "logs/episodic_ledger.jsonl"

    def __init__(self, log_path: Optional[str] = None, max_events: int = 1000):
        self.log_path = log_path or self.DEFAULT_LOG_PATH
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.critical_events: List[EpisodicEvent] = []
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.load_ledger()

    @staticmethod
    def _is_critical_event(event: EpisodicEvent) -> bool:
        """Determines if an episodic event is a critical safety or verifier violation."""
        act = event.action_type.upper()
        stat = event.status.upper()
        if any(k in act for k in ("EMERGENCY", "HALT", "VIOLATION", "DISSONANCE", "WATCHDOG")):
            return True
        if stat in ("REJECTED", "HALTED", "FALSIFIED"):
            return True
        if event.details and event.details.get("critical"):
            return True
        return False

    def record_event(
        self,
        action_type: str,
        concept_name: str,
        source_name: str,
        formula_or_axiom: str,
        status: str = "COMMITTED",
        details: Optional[Dict[str, Any]] = None
    ) -> EpisodicEvent:
        """Appends a new episodic event to memory and disk."""
        evt = EpisodicEvent(
            timestamp=time.time(),
            action_type=action_type,
            concept_name=concept_name,
            source_name=source_name,
            formula_or_axiom=formula_or_axiom,
            status=status,
            details=details or {}
        )
        self.events.append(evt)
        if self._is_critical_event(evt):
            self.critical_events.append(evt)
        self._append_to_disk(evt)
        return evt

    def _append_to_disk(self, event: EpisodicEvent):
        """Appends event JSON line to disk."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except (OSError, IOError) as e:
            logger.error("Failed to append event to episodic ledger %s: %s", self.log_path, e, exc_info=True)

    def load_ledger(self):
        """Loads historical events from disk."""
        self.events.clear()
        self.critical_events.clear()
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                evt = EpisodicEvent(**data)
                                self.events.append(evt)
                                if self._is_critical_event(evt):
                                    self.critical_events.append(evt)
                            except Exception as e:
                                logger.warning("Corrupted JSON line in %s: %s", self.log_path, e)
            except (OSError, IOError) as e:
                logger.error("Failed to read episodic ledger %s: %s", self.log_path, e, exc_info=True)

    def get_recent_events(self, limit: int = 5) -> List[EpisodicEvent]:
        """Returns the most recent N episodic events."""
        ev_list = list(self.events)
        return ev_list[-limit:] if ev_list else []

    def get_critical_events(self) -> List[EpisodicEvent]:
        """Returns all recorded critical safety events and falsifications."""
        return list(self.critical_events)

    def get_provenance_stats(self) -> Dict[str, int]:
        """Calculates breakdown of concepts learned per source."""
        stats = {
            "arxiv": 0,
            "wikipedia": 0,
            "curiosity_loop": 0,
            "user_teaching": 0,
            "other": 0
        }
        user_name = "user"
        try:
            from src.reasoning_engine.self_model import SelfModelEngine
            user_name = SelfModelEngine().user_profile.get("name", "user").lower()
        except Exception:
            pass

        for e in self.events:
            if e.status != "COMMITTED":
                continue
            act = e.action_type.upper()
            src = e.source_name.lower()
            if "arxiv" in act or "arxiv" in src:
                stats["arxiv"] += 1
            elif "wiki" in act or "wiki" in src:
                stats["wikipedia"] += 1
            elif "discover" in act or "curiosity" in src:
                stats["curiosity_loop"] += 1
            elif "teach" in act or "user" in act or user_name in src or "user" in src or "human" in src or "teacher" in src:
                stats["user_teaching"] += 1
            else:
                stats["other"] += 1
        return stats

    def format_recent_ledger_for_context(self, limit: int = 4) -> str:
        """Formats recent episodic memory trace for dynamic prompt grounding."""
        all_recent = self.get_recent_events(limit=limit * 5)
        recent = [
            e for e in all_recent
            if e.action_type not in ("GENESIS_RESURFACE", "GENESIS_PRIORITY_SHIFT", "GENESIS_ANCHOR_CREATED")
            and e.status in ("COMMITTED", "VERIFIED")
        ][:limit]
        if not recent:
            return "No recent discovery or ingestion actions recorded in this session."
        lines = [f"• {e.to_summary()}" for e in recent]
        stats = self.get_provenance_stats()
        stats_str = f"Source Provenance: arXiv={stats['arxiv']}, Wikipedia={stats['wikipedia']}, Curiosity={stats['curiosity_loop']}, User={stats['user_teaching']}"
        return f"{stats_str}\n" + "\n".join(lines)
