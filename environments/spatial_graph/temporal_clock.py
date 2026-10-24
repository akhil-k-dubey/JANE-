"""Continuous World Clock: 60Hz Real-Time Temporal Updating Engine.

Runs continuous temporal ticks (default 60Hz: dt = 1/60s) independently of deliberation.
Updates world physics, dynamic door timers, moving hazards, and pressure fluctuation.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any, Tuple


@dataclass
class TimedEvent:
    """A scheduled environmental event occurring at a specific continuous world timestamp."""
    trigger_time: float
    callback: Callable[[], None]
    description: str
    executed: bool = False


class ContinuousWorldClock:
    """
    60Hz Continuous World Clock.
    Drives real-time environmental dynamics asynchronously.
    Supports both real-time threaded execution and deterministic step-by-step ticking for tests.
    """

    def __init__(self, tick_rate_hz: float = 60.0):
        self.tick_rate_hz = tick_rate_hz
        self.dt = 1.0 / tick_rate_hz
        self.sim_time: float = 0.0

        self.is_running: bool = False
        self._stop_event = threading.Event()
        self._clock_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Scheduled events and tick listeners
        self.scheduled_events: List[TimedEvent] = []
        self.tick_listeners: List[Callable[[float, float], None]] = []

    def register_tick_listener(self, listener: Callable[[float, float], None]):
        """Registers a callback (current_time, dt) invoked on every clock tick."""
        with self._lock:
            self.tick_listeners.append(listener)

    def schedule_event(self, trigger_time: float, callback: Callable[[], None], description: str = ""):
        """Schedules an environmental event at a continuous timestamp."""
        with self._lock:
            self.scheduled_events.append(TimedEvent(
                trigger_time=trigger_time,
                callback=callback,
                description=description
            ))

    def step(self, delta_time: Optional[float] = None) -> float:
        """
        Advances the continuous clock by delta_time (defaults to 1/60s).
        Executes pending timed events and notifies tick listeners.
        Returns the new simulation time.
        """
        dt = delta_time if delta_time is not None else self.dt
        with self._lock:
            self.sim_time += dt
            current_t = self.sim_time

            # Fire scheduled events
            for event in self.scheduled_events:
                if not event.executed and current_t >= event.trigger_time:
                    event.executed = True
                    try:
                        event.callback()
                    except Exception as e:
                        print(f"[WorldClock ERROR] Event '{event.description}' failed: {e}")

            # Notify listeners
            listeners = list(self.tick_listeners)

        for listener in listeners:
            try:
                listener(current_t, dt)
            except Exception as e:
                print(f"[WorldClock ERROR] Tick listener failed: {e}")

        return current_t

    def start(self):
        """Starts real-time 60Hz background ticking."""
        if self.is_running:
            return

        self._stop_event.clear()
        self.is_running = True
        self._clock_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ContinuousWorldClock-60Hz"
        )
        self._clock_thread.start()

    def stop(self):
        """Stops the real-time background clock."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()
        if self._clock_thread and self._clock_thread.is_alive():
            self._clock_thread.join(timeout=1.0)
            self._clock_thread = None

    def _run_loop(self):
        """Internal real-time clock loop ticking at tick_rate_hz."""
        target_dt = self.dt
        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            self.step(target_dt)
            t_elapsed = time.perf_counter() - t_start
            sleep_time = target_dt - t_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def reset(self):
        """Resets the clock to t=0.0."""
        with self._lock:
            self.sim_time = 0.0
            self.scheduled_events.clear()
            self.tick_listeners.clear()
