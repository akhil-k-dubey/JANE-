"""Autonomous Idle Curiosity Daemon (Tier 1 Ingestion) for Project Jane.

Monitors the 256D continuous concept manifold during passive idle periods (when Akhil is not
actively conversing and compute is available):
1. Topological Void Detection:
   Identifies the most sparse or unrepresented knowledge cluster in the manifold.
2. Autonomous Ingestion:
   Queries scientific ingestors (Wikipedia / arXiv) to acquire new foundational concepts.
3. Native Hebbian Plasticity:
   Maps extracted knowledge and verified axioms directly into the 256D manifold.
4. Preemptible & Pausable:
   Cooperates with Tier 3 Manual Overrides by yielding immediately when high-priority
   admin learning commands are issued.
"""

import time
import threading
import logging
from typing import Optional, List, Dict, Any, Callable

from src.utils.logger import get_logger

logger = get_logger("curiosity_daemon")


class CuriosityDaemon:
    """
    Tier 1 Autonomous Background Curiosity Daemon.
    Fills sparse clusters of Jane's 256D manifold during idle periods.
    """

    def __init__(
        self,
        manifold: Any,
        wikipedia_ingestor: Any,
        arxiv_ingestor: Optional[Any] = None,
        check_interval_sec: float = 30.0,
        on_learn_callback: Optional[Callable[[str, Any], None]] = None
    ):
        self.manifold = manifold
        self.wiki = wikipedia_ingestor
        self.arxiv = arxiv_ingestor
        self.check_interval_sec = check_interval_sec
        self.on_learn_callback = on_learn_callback

        self.running = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Set means active (not paused)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.learned_history: List[Dict[str, Any]] = []

    def start(self):
        """Starts the background curiosity thread."""
        with self._lock:
            if self.running and self._thread is not None and self._thread.is_alive():
                return
            self.running = True
            self._stop_event.clear()
            self._pause_event.set()
            self._thread = threading.Thread(
                target=self._curiosity_loop,
                daemon=True,
                name="JaneCuriosityDaemon"
            )
            self._thread.start()
            logger.info("CuriosityDaemon started (Scan interval: %.1fs).", self.check_interval_sec)

    def stop(self, timeout: float = 2.0):
        """Stops the background curiosity thread gracefully."""
        self.running = False
        self._stop_event.set()
        self._pause_event.set()  # Unblock if waiting on pause
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("CuriosityDaemon stopped.")

    def pause(self):
        """Pauses the curiosity loop for high-priority Tier 3 manual override."""
        self._pause_event.clear()
        logger.info("CuriosityDaemon paused for manual override.")

    def resume(self):
        """Resumes the curiosity loop."""
        self._pause_event.set()
        logger.info("CuriosityDaemon resumed.")

    @property
    def is_paused(self) -> bool:
        """Returns True if the daemon is currently paused."""
        return not self._pause_event.is_set()

    @property
    def is_running(self) -> bool:
        """Returns True if background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Executes a single curiosity cycle:
        1. Find the most 'sparse' cluster in the 256D manifold.
        2. Autonomously fetch related topic summary.
        3. Map it natively into 256D space using Hebbian plasticity.
        """
        # 1. Find sparse cluster in 256D manifold
        if hasattr(self.manifold, "find_sparse_cluster"):
            target_concept = self.manifold.find_sparse_cluster()
        else:
            target_concept = "quantum_optics"

        print(f"[Curiosity Daemon] Jane is autonomously learning about: {target_concept}")
        logger.info("[Curiosity Daemon] Autonomous curiosity cycle: Target concept '%s'", target_concept)

        # 2. Autonomously fetch summary
        raw_text = ""
        report = None
        if hasattr(self.wiki, "fetch_summary"):
            raw_text = self.wiki.fetch_summary(target_concept)
        elif hasattr(self.wiki, "fetch_wikipedia_article"):
            art, _ = self.wiki.fetch_wikipedia_article(target_concept)
            raw_text = art.get("summary", "") if art else ""

        # 3. Map into 256D space using Hebbian plasticity
        ingest_res = {}
        if hasattr(self.manifold, "ingest_text"):
            ingest_res = self.manifold.ingest_text(raw_text, concept_name=target_concept)

        # Also trigger formal Wikipedia ingestion pass if supported
        if hasattr(self.wiki, "ingest_article"):
            try:
                report = self.wiki.ingest_article(target_concept)
            except Exception as e:
                logger.warning("Curiosity formal ingestion error for '%s': %s", target_concept, e)

        record = {
            "timestamp": time.time(),
            "concept": target_concept,
            "raw_text_length": len(raw_text),
            "ingest_result": ingest_res,
            "report": report
        }

        with self._lock:
            self.learned_history.append(record)

        if self.on_learn_callback:
            try:
                self.on_learn_callback(target_concept, record)
            except Exception as e:
                logger.error("Error invoking on_learn_callback: %s", e)

        return record

    def _curiosity_loop(self):
        """Runs while Jane is idle. Fills sparse regions of the manifold."""
        while self.running and not self._stop_event.is_set():
            # Wait if paused by manual override
            self._pause_event.wait()
            if self._stop_event.is_set() or not self.running:
                break

            try:
                self.step()
            except Exception as e:
                logger.error("Error in CuriosityDaemon cycle: %s", e, exc_info=True)

            # Sleep to save resources
            self._stop_event.wait(self.check_interval_sec)

    def get_learned_history(self) -> List[Dict[str, Any]]:
        """Returns history of concepts learned through curiosity."""
        with self._lock:
            return list(self.learned_history)

    def clear_learned_history(self):
        """Clears history of learned concepts."""
        with self._lock:
            self.learned_history.clear()
