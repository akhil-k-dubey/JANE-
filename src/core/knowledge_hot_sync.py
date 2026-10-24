"""Dynamic Knowledge & Manifold Synchronizer for Project Jane.

Monitors data/knowledge_graph/*.json in real-time. Automatically detects disk
modifications, updates in-memory indices, and refreshes the 256D manifold and
self-model without requiring a process restart. Zero regex.
"""

import os
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("knowledge_hot_sync")


@dataclass
class KnowledgeSyncReport:
    """Report detailing knowledge synchronization events."""
    synced: bool
    changed_files: List[str] = field(default_factory=list)
    total_concepts: int = 0
    total_triples: int = 0
    total_principles: int = 0
    elapsed_ms: float = 0.0
    message: str = "No changes detected."


class KnowledgeHotSyncEngine:
    """
    Watches data/knowledge_graph files for real-time changes and updates
    in-memory retriever and self-model structures instantly.
    """

    DEFAULT_WATCH_FILES = [
        "data/knowledge_graph/concept_triples.json",
        "data/knowledge_graph/physics_principles.json",
        "data/knowledge_graph/self_model.json"
    ]

    def __init__(self, watch_files: Optional[List[str]] = None):
        self.watch_files = watch_files or list(self.DEFAULT_WATCH_FILES)
        self._file_snapshots: Dict[str, Tuple[float, str]] = {}
        self._initialize_snapshots()

    def _compute_file_hash(self, path: str) -> str:
        """Computes SHA256 checksum of file contents."""
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return ""

    def _initialize_snapshots(self):
        """Records initial mtime and hash states for watched files."""
        for path in self.watch_files:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                h = self._compute_file_hash(path)
                self._file_snapshots[path] = (mtime, h)

    def check_for_changes(self) -> List[str]:
        """
        Checks watched files against existing snapshots.
        Returns list of modified or newly created file paths.
        """
        changed: List[str] = []
        for path in self.watch_files:
            if not os.path.exists(path):
                continue

            current_mtime = os.path.getmtime(path)
            prev = self._file_snapshots.get(path)

            if prev is None or current_mtime > prev[0]:
                current_hash = self._compute_file_hash(path)
                if prev is None or current_hash != prev[1]:
                    changed.append(path)
                    self._file_snapshots[path] = (current_mtime, current_hash)

        return changed

    def sync(self, pipeline: Any) -> KnowledgeSyncReport:
        """
        Detects disk updates and re-synchronizes the running Jane pipeline.
        Rebuilds inverted indices and updates concept vocabulary.
        """
        t0 = time.time()
        changed = self.check_for_changes()
        if not changed:
            return KnowledgeSyncReport(synced=False, elapsed_ms=0.0)

        # 1. Refresh ManifoldRetriever
        if hasattr(pipeline, "refresh_retriever"):
            pipeline.refresh_retriever()
        elif hasattr(pipeline, "retriever") and hasattr(pipeline.retriever, "load_data"):
            pipeline.retriever.load_data()

        # 2. Refresh SelfModelEngine
        if hasattr(pipeline, "self_model") and hasattr(pipeline.self_model, "load_model"):
            pipeline.self_model.load_model()

        # 3. Gather updated metrics
        num_concepts = 0
        if hasattr(pipeline, "hebbian_manifold") and hasattr(pipeline.hebbian_manifold, "num_concepts"):
            num_concepts = pipeline.hebbian_manifold.num_concepts
        elif hasattr(pipeline, "self_model"):
            num_concepts = pipeline.self_model.core_concepts_count

        num_triples = len(pipeline.retriever.triples) if hasattr(pipeline, "retriever") else 0
        num_principles = len(pipeline.retriever.principles) if hasattr(pipeline, "retriever") else 0
        elapsed = (time.time() - t0) * 1000.0

        msg = f"Synchronized {len(changed)} knowledge file(s) in {elapsed:.1f}ms: {', '.join([os.path.basename(p) for p in changed])}"
        logger.info(msg)

        return KnowledgeSyncReport(
            synced=True,
            changed_files=changed,
            total_concepts=num_concepts,
            total_triples=num_triples,
            total_principles=num_principles,
            elapsed_ms=elapsed,
            message=msg
        )
