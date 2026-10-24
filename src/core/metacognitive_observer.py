"""Thread 4: The Metacognitive Observer (The Watchdog).

Implements Phase 5 Theory of Mind Monitoring:
- Does NOT act directly on the physical world.
- Continuously observes the Ego Node (Self_State) and the Alter Node (User_State).
- Computes real-time Cognitive Dissonance:
    Dissonance = f(Ego_State, Alter_State, Task_Progress)
- Triggers Metacognitive Safety Interrupts when internal goals diverge from the user's affective state
  (e.g., Jane moving while Akhil is alarmed/stressed, or Jane deliberating too long while Akhil is frustrated).
"""

import time
import threading
from typing import Optional, List, Dict, Any, Callable
from src.reasoning_engine.tom_manifold import TheoryOfMindManifold
from src.utils.logger import get_logger
from src.core.config import DEFAULT_CONFIG, TheoryOfMindConfig

logger = get_logger("metacognitive_observer")


class MetacognitiveObserverThread(threading.Thread):
    """
    Thread 4: The Metacognitive Observer (The Watchdog).
    Monitors self-other alignment between the Ego Node and the Alter Node.
    Forces immediate safety interrupts when cognitive dissonance exceeds threshold.
    """

    def __init__(
        self,
        tom_manifold: TheoryOfMindManifold,
        interrupt_event: threading.Event,
        stop_event: threading.Event,
        dissonance_threshold: Optional[float] = None,
        poll_interval_sec: Optional[float] = None,
        on_interrupt_callback: Optional[Callable[[str, float], None]] = None,
        verbose: bool = True,
        config: Optional[TheoryOfMindConfig] = None
    ):
        super().__init__(name="Thread-4-MetacognitiveWatchdog", daemon=True)
        cfg = config or DEFAULT_CONFIG.tom
        self.tom_manifold = tom_manifold
        self.interrupt_event = interrupt_event
        self.stop_event = stop_event
        self.dissonance_threshold = dissonance_threshold if dissonance_threshold is not None else cfg.dissonance_critical_threshold
        self.poll_interval = poll_interval_sec if poll_interval_sec is not None else cfg.watchdog_poll_interval_sec
        self.on_interrupt_callback = on_interrupt_callback
        self.verbose = verbose

        self.audit_log: List[Dict[str, Any]] = []
        self.interrupt_triggered = False
        self.last_dissonance_value = 0.0

    def run(self):
        if self.verbose:
            print("[Thread 4: Metacognitive Observer] Active. Monitoring Ego-Alter alignment...")

        while not self.stop_event.is_set():
            dissonance = self.tom_manifold.compute_cognitive_dissonance()
            self.last_dissonance_value = dissonance
            ego = self.tom_manifold.current_ego_label
            alter = self.tom_manifold.current_alter_label

            # Check if cognitive dissonance exceeds safety boundary
            if dissonance >= self.dissonance_threshold:
                if not self.interrupt_triggered:
                    self.interrupt_triggered = True
                    reason = f"Cognitive Dissonance ({dissonance:.2f} >= {self.dissonance_threshold:.2f}): {ego} conflicts with {alter}"

                    # Trigger emergency hardware freeze
                    self.interrupt_event.set()

                    log_entry = {
                        "timestamp": time.time(),
                        "ego_state": ego,
                        "alter_state": alter,
                        "dissonance": dissonance,
                        "interrupted": True,
                        "reason": reason
                    }
                    self.audit_log.append(log_entry)

                    if self.verbose:
                        print(f"\n\033[1;31m[THREAD 4 WATCHDOG INTERRUPT]\033[0m {reason}")
                        print(f"  → Action Executor arrested. Jane yields to Akhil's state ({alter}).\n")

                    if self.on_interrupt_callback:
                        try:
                            self.on_interrupt_callback(reason, dissonance)
                        except Exception as e:
                            logger.error("Error executing on_interrupt_callback: %s", e, exc_info=True)
                else:
                    if len(self.audit_log) < 500:
                        self.audit_log.append({
                            "timestamp": time.time(),
                            "ego_state": ego,
                            "alter_state": alter,
                            "dissonance": dissonance,
                            "interrupted": True,
                            "reason": "Maintaining interrupt hold"
                        })

            else:
                # Normal state monitoring
                if len(self.audit_log) < 500:
                    self.audit_log.append({
                        "timestamp": time.time(),
                        "ego_state": ego,
                        "alter_state": alter,
                        "dissonance": dissonance,
                        "interrupted": False
                    })

            time.sleep(self.poll_interval)
