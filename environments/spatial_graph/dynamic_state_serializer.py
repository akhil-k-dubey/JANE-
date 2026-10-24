"""Dynamic State Serializer: 60Hz Real-Time Clingo ASP State Streaming & Delta Detection.

Runs on a continuous clock (e.g. 60Hz).
Serializes live Python SpatialEnvironmentGraph states into symbolic Clingo facts and
detects instantaneous state deltas (e.g. door closing mid-step) to alert Jane's DNC tape.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Callable, Any
import clingo

from .environment import SpatialEnvironmentGraph, GraphNode, GraphEdge
from .temporal_clock import ContinuousWorldClock


@dataclass
class WorldStateDelta:
    """Represents a symbolic change in the environment detected by the 60Hz serializer."""
    sim_time: float
    added_facts: Set[str]
    removed_facts: Set[str]
    locked_doors_changed: List[Tuple[str, str, bool]]  # (source, target, is_locked)
    is_structural_change: bool = False                 # True if navigable topology was altered

    def description(self) -> str:
        changes = []
        for src, tgt, locked in self.locked_doors_changed:
            status = "CLOSED/LOCKED" if locked else "OPENED/UNLOCKED"
            changes.append(f"Door {src}->{tgt} {status}")
        return "; ".join(changes) if changes else f"Delta (+{len(self.added_facts)}, -{len(self.removed_facts)})"


class DynamicStateSerializer:
    """
    60Hz Continuous State Serializer.
    Continuously monitors the SpatialEnvironmentGraph, serializes live states into Clingo ASP,
    and publishes state deltas to Jane's reactive deliberation controller.
    """

    def __init__(
        self,
        env: SpatialEnvironmentGraph,
        clock: Optional[ContinuousWorldClock] = None
    ):
        self.env = env
        self.clock = clock
        self.last_facts: Set[str] = set()
        self.last_edge_locks: Dict[Tuple[str, str], bool] = {}
        self.delta_subscribers: List[Callable[[WorldStateDelta], None]] = []

        # Initialize baseline state
        self._sync_baseline()

        # Connect to continuous clock if provided
        if self.clock:
            self.clock.register_tick_listener(self.on_clock_tick)

    def _sync_baseline(self):
        """Captures initial baseline state facts and edge locks."""
        self.env.update_lock_conditions()
        self.last_facts = self.extract_current_symbolic_facts()
        self.last_edge_locks = {
            (e.source, e.target): e.is_locked for e in self.env.edges
        }

    def subscribe_delta(self, callback: Callable[[WorldStateDelta], None]):
        """Subscribes an agent controller to receive real-time state deltas."""
        self.delta_subscribers.append(callback)

    def extract_current_symbolic_facts(self) -> Set[str]:
        """Serializes current environment into normalized symbolic facts."""
        facts = set()

        # Nodes
        for nid, node in self.env.nodes.items():
            facts.add(f"node({nid})")
            facts.add(f"room_type({nid},{node.properties.get('type', 'room')})")
            p_val = round(node.pressure, 1)
            facts.add(f"pressure({nid},{p_val})")

        # Inventory
        for item in self.env.agent_inventory:
            facts.add(f"has_key({item})")

        # Edges & Locks
        for edge in self.env.edges:
            facts.add(f"edge({edge.source},{edge.target})")
            if edge.is_locked:
                facts.add(f"locked({edge.source},{edge.target})")
            else:
                facts.add(f"unlocked({edge.source},{edge.target})")
            if edge.required_key:
                facts.add(f"requires_key({edge.source},{edge.target},{edge.required_key})")

        # Agent State
        if self.env.agent_node:
            facts.add(f"agent_node({self.env.agent_node})")

        return facts

    def on_clock_tick(self, sim_time: float, dt: float) -> Optional[WorldStateDelta]:
        """
        Invoked on every 60Hz clock tick.
        Checks for dynamic environment mutations, compares with last state, and notifies subscribers.
        """
        self.env.update_lock_conditions()
        current_facts = self.extract_current_symbolic_facts()

        added = current_facts - self.last_facts
        removed = self.last_facts - current_facts

        # Check for edge lock changes specifically
        locked_doors_changed = []
        is_structural = False

        for edge in self.env.edges:
            key = (edge.source, edge.target)
            prev_locked = self.last_edge_locks.get(key, False)
            if edge.is_locked != prev_locked:
                locked_doors_changed.append((edge.source, edge.target, edge.is_locked))
                self.last_edge_locks[key] = edge.is_locked
                is_structural = True

        if added or removed or locked_doors_changed:
            delta = WorldStateDelta(
                sim_time=sim_time,
                added_facts=added,
                removed_facts=removed,
                locked_doors_changed=locked_doors_changed,
                is_structural_change=is_structural
            )

            self.last_facts = current_facts

            # Notify all subscribers (Jane's DNC tape monitor)
            for sub in self.delta_subscribers:
                try:
                    sub(delta)
                except Exception as e:
                    print(f"[DynamicStateSerializer ERROR] Delta subscriber failed: {e}")

            return delta

        return None

    def serialize_to_clingo_program(self, start_node: str, goal_node: str) -> str:
        """Generates ready-to-solve Clingo ASP program from the instantaneous 60Hz serialized state."""
        self.env.update_lock_conditions()
        facts = sorted(list(self.extract_current_symbolic_facts()))
        fact_lines = [f"{f}." for f in facts]
        fact_lines.append(f"start({start_node}).")
        fact_lines.append(f"goal({goal_node}).")

        axioms = """
        % Movement Axioms
        permitted(X, Y) :- edge(X, Y), not locked(X, Y).
        permitted(X, Y) :- edge(X, Y), locked(X, Y), requires_key(X, Y, K), has_key(K).

        reachable(X) :- start(X).
        reachable(Y) :- reachable(X), permitted(X, Y).

        goal_reached :- goal(G), reachable(G).
        violation(goal_unreachable) :- not goal_reached.
        :- violation(_).

        #show reachable/1.
        #show permitted/2.
        """
        return "\n".join(fact_lines) + "\n" + axioms
