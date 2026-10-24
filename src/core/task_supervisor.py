"""Asynchronous Background Task Supervisor for Project Jane.

Manages, tracks, and supervises real background worker threads (e.g., long-running
Wikipedia ingestion, ArXiv paper harvesting, or epistemic hypothesis exploration).
Provides deterministic, immutable live telemetry to prevent ungrounded LLM hallucination
regarding task execution status and progress metrics.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from src.utils.logger import get_logger

logger = get_logger("task_supervisor")


@dataclass
class BackgroundTaskRecord:
    task_id: str
    task_type: str  # e.g., "INGESTION", "HYPOTHESIS_SEARCH"
    description: str
    source: str     # e.g., "wikipedia", "arxiv", "epistemic_engine"
    domain: str
    target_count: int
    current_count: int = 0
    accepted_concepts: List[str] = field(default_factory=list)
    rejected_concepts: List[str] = field(default_factory=list)
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return max(0.0, end - self.start_time)

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        if self.current_count == 0 or self.status != "RUNNING":
            return None
        rate = self.current_count / max(0.1, self.elapsed_seconds)
        remaining_items = max(0, self.target_count - self.current_count)
        return remaining_items / max(1e-4, rate)

    def to_dict(self) -> Dict[str, Any]:
        rem = self.estimated_remaining_seconds
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "source": self.source,
            "domain": self.domain,
            "target_count": self.target_count,
            "current_count": self.current_count,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "estimated_remaining_seconds": round(rem, 1) if rem is not None else None,
            "status": self.status,
            "accepted_concepts": list(self.accepted_concepts[-5:]),
            "error_message": self.error_message
        }


class AsyncTaskSupervisor:
    """
    Central thread-safe supervisor for real background tasks in Project Jane.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, BackgroundTaskRecord] = {}
        self._task_counter = 0

    def spawn_wikipedia_ingestion(
        self,
        wikipedia_ingestor: Any,
        domain: str,
        target_count: int = 100,
        article_titles: Optional[List[str]] = None
    ) -> BackgroundTaskRecord:
        """
        Spawns a real asynchronous background worker thread to ingest concepts
        from Wikipedia up to target_count.
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"wiki_ingest_{self._task_counter}"
            record = BackgroundTaskRecord(
                task_id=task_id,
                task_type="INGESTION",
                description=f"Wikipedia Ingestion: {target_count} concepts in {domain}",
                source="wikipedia",
                domain=domain,
                target_count=target_count,
                status="RUNNING"
            )
            self._tasks[task_id] = record

        thread = threading.Thread(
            target=self._wikipedia_worker_loop,
            args=(record, wikipedia_ingestor, article_titles),
            daemon=True,
            name=f"JaneWorker-{task_id}"
        )
        thread.start()
        logger.info("Spawned async task %s: %s", task_id, record.description)
        return record

    def _wikipedia_worker_loop(
        self,
        record: BackgroundTaskRecord,
        wikipedia_ingestor: Any,
        article_titles: Optional[List[str]] = None
    ):
        """Worker thread body for continuous background ingestion."""
        # Curated seed lists for domain expansion
        math_seeds = [
            "Topology", "Differential_equation", "Abstract_algebra",
            "Linear_algebra", "Calculus", "Group_theory",
            "Manifold", "Tensor", "Probability_theory", "Fourier_transform",
            "Complex_analysis", "Knot_theory", "Category_theory", "Number_theory"
        ]
        physics_seeds = [
            "Quantum_entanglement", "Superconductivity", "Black_hole_thermodynamics",
            "General_relativity", "Thermodynamics", "Statistical_mechanics",
            "Electromagnetism", "Standard_Model", "Fluid_dynamics", "Plasma_physics"
        ]

        queue = []
        if article_titles:
            queue.extend(article_titles)
        elif any(k in record.domain.lower() for k in ("math", "algebra", "calculus", "geometry")):
            queue.extend(math_seeds)
        else:
            queue.extend(physics_seeds)

        seen_titles = set(queue)

        try:
            while record.current_count < record.target_count:
                if record.status != "RUNNING":
                    break

                if queue:
                    target_title = queue.pop(0)
                else:
                    # Dynamically detect topological void in the live manifold
                    if hasattr(wikipedia_ingestor, "manifold") and hasattr(wikipedia_ingestor.manifold, "find_sparse_cluster"):
                        target_title = wikipedia_ingestor.manifold.find_sparse_cluster()
                    else:
                        target_title = "quantum_optics"

                # Ingest article through Z3 verification gate
                try:
                    report = wikipedia_ingestor.ingest_article(target_title)
                    with self._lock:
                        for c in report.accepted_concepts:
                            if c not in record.accepted_concepts:
                                record.accepted_concepts.append(c)
                                record.current_count += 1
                        for r in report.rejected_concepts:
                            record.rejected_concepts.append(r.get("concept", "unknown"))

                    for c in report.accepted_concepts:
                        if c not in seen_titles:
                            seen_titles.add(c)
                            queue.append(c)
                except Exception as ex:
                    logger.warning("Error ingesting article %s in task %s: %s", target_title, record.task_id, ex)

                # Graceful pause between network / SMT requests
                time.sleep(0.5)

            with self._lock:
                record.status = "COMPLETED"
                record.end_time = time.time()
                logger.info("Task %s completed with %d concepts", record.task_id, record.current_count)

        except Exception as ex:
            with self._lock:
                record.status = "FAILED"
                record.error_message = str(ex)
                record.end_time = time.time()
                logger.error("Task %s failed: %s", record.task_id, ex, exc_info=True)

    def get_active_tasks(self) -> List[BackgroundTaskRecord]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == "RUNNING"]

    def get_most_recent_task(self) -> Optional[BackgroundTaskRecord]:
        with self._lock:
            if not self._tasks:
                return None
            return list(self._tasks.values())[-1]

    def get_task(self, task_id: str) -> Optional[BackgroundTaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """
        Returns structured telemetry for active and recent background tasks.
        Provides ground truth context to the mouth and prevents hallucinated metrics.
        """
        with self._lock:
            active = [t.to_dict() for t in self._tasks.values() if t.status == "RUNNING"]
            completed = [t.to_dict() for t in self._tasks.values() if t.status in ("COMPLETED", "FAILED")]

            lines = []
            if active:
                lines.append("[ACTIVE BACKGROUND TASKS - IMMUTABLE GROUND TRUTH TELEMETRY]:")
                for t in active:
                    rem_str = f"{t['estimated_remaining_seconds']}s remaining" if t['estimated_remaining_seconds'] is not None else "estimating..."
                    lines.append(
                        f"  • Task ID: {t['task_id']} ({t['description']})\n"
                        f"    - Status: {t['status']}\n"
                        f"    - Actual Ingested Concepts: {t['current_count']} / {t['target_count']}\n"
                        f"    - Elapsed Time: {t['elapsed_seconds']}s (est. {rem_str})\n"
                        f"    - Recent Verified Concepts: {', '.join(t['accepted_concepts']) if t['accepted_concepts'] else 'processing...'}"
                    )
            else:
                lines.append("[ACTIVE BACKGROUND TASKS]: No background ingestion tasks currently running.")

            if completed:
                last_comp = completed[-1]
                lines.append(
                    f"[RECENT COMPLETED TASK]: {last_comp['task_id']} finished with "
                    f"{last_comp['current_count']} concepts in {last_comp['elapsed_seconds']}s."
                )

            return {
                "has_active_tasks": len(active) > 0,
                "active_tasks": active,
                "completed_tasks": completed,
                "formatted_telemetry": "\n".join(lines)
            }
