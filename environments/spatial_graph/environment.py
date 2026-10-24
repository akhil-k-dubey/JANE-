"""Dynamic Spatial Graph Environment: Strictly-Typed Python Data Classes.

No hardcoded .lp files or static rules. The environment is pure Python with dynamic state:
- GraphNode: Rooms/zones with continuous coordinates, properties (temperature, pressure, items)
- GraphEdge: Topological transitions, dynamic locks, required keys, conditional locks
- SpatialEnvironmentGraph: Full dynamic state machine with action execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
import math


@dataclass
class GraphNode:
    """Represents a room or physical zone in the environment."""
    id: str
    coordinates: Tuple[float, float]
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def room_type(self) -> str:
        return self.properties.get("type", "room")

    @property
    def pressure(self) -> float:
        return float(self.properties.get("pressure", 1.0))

    @property
    def temperature(self) -> float:
        return float(self.properties.get("temperature", 20.0))

    @property
    def items(self) -> Set[str]:
        if "items" not in self.properties:
            self.properties["items"] = set()
        return self.properties["items"]


@dataclass
class GraphEdge:
    """Represents a navigable transition between two zones."""
    source: str
    target: str
    is_locked: bool = False
    required_key: Optional[str] = None
    lock_condition: Optional[str] = None  # e.g., "pressure > 40"
    distance: float = 1.0


@dataclass
class ContinuousObstacle:
    """Bounding box obstacle in continuous 2D space: (min_x, min_y, max_x, max_y)."""
    id: str
    bounds: Tuple[float, float, float, float]

    @property
    def min_x(self) -> float:
        return self.bounds[0]

    @property
    def min_y(self) -> float:
        return self.bounds[1]

    @property
    def max_x(self) -> float:
        return self.bounds[2]

    @property
    def max_y(self) -> float:
        return self.bounds[3]


class SpatialEnvironmentGraph:
    """
    Dynamic spatial environment state machine.
    Maintains:
    - Nodes (zones with coordinates and dynamic physical state)
    - Edges (connections with dynamic lock states)
    - Obstacles (continuous geometric boxes)
    - Agent State (current node, position, inventory)
    """

    def __init__(self, name: str = "Facility"):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.obstacles: List[ContinuousObstacle] = []
        self.agent_node: str = ""
        self.agent_pos: Tuple[float, float] = (0.0, 0.0)
        self.agent_inventory: Set[str] = set()
        self.global_state: Dict[str, Any] = {
            "reactor_power": 0.0,
            "alarm_active": False
        }

    def add_node(self, node_id: str, coordinates: Tuple[float, float], properties: Optional[Dict[str, Any]] = None) -> GraphNode:
        props = properties or {}
        node = GraphNode(id=node_id, coordinates=coordinates, properties=props)
        self.nodes[node_id] = node
        if not self.agent_node:
            self.agent_node = node_id
            self.agent_pos = coordinates
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        is_locked: bool = False,
        required_key: Optional[str] = None,
        lock_condition: Optional[str] = None,
        bidirectional: bool = True
    ):
        p1 = self.nodes[source].coordinates
        p2 = self.nodes[target].coordinates
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        edge = GraphEdge(
            source=source,
            target=target,
            is_locked=is_locked,
            required_key=required_key,
            lock_condition=lock_condition,
            distance=dist
        )
        self.edges.append(edge)
        if bidirectional:
            back_edge = GraphEdge(
                source=target,
                target=source,
                is_locked=is_locked,
                required_key=required_key,
                lock_condition=lock_condition,
                distance=dist
            )
            self.edges.append(back_edge)

    def add_obstacle(self, obstacle_id: str, bounds: Tuple[float, float, float, float]) -> ContinuousObstacle:
        obs = ContinuousObstacle(id=obstacle_id, bounds=bounds)
        self.obstacles.append(obs)
        return obs

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def get_edges_from(self, source_id: str) -> List[GraphEdge]:
        return [e for e in self.edges if e.source == source_id]

    def update_lock_conditions(self):
        """
        Dynamically updates lock status on edges based on environmental properties
        (e.g., venting gas drops pressure <= 40%, clearing lock condition).
        """
        for edge in self.edges:
            if edge.lock_condition:
                src_node = self.nodes.get(edge.source)
                tgt_node = self.nodes.get(edge.target)
                # Check pressure condition
                if "pressure > 40" in edge.lock_condition:
                    pressure = max(src_node.pressure if src_node else 0.0, tgt_node.pressure if tgt_node else 0.0)
                    edge.is_locked = (pressure > 40.0)

    # Agent Actions
    def move_to(self, target_node_id: str) -> Tuple[bool, str]:
        """Attempts to move agent from current node to target node."""
        if target_node_id not in self.nodes:
            return False, f"Target node '{target_node_id}' does not exist."

        if self.agent_node == target_node_id:
            return True, f"Agent already in {target_node_id}."

        self.update_lock_conditions()

        # Find matching edge
        edge = next((e for e in self.edges if e.source == self.agent_node and e.target == target_node_id), None)
        if not edge:
            return False, f"No topological edge between {self.agent_node} and {target_node_id}."

        if edge.is_locked:
            if edge.required_key and edge.required_key in self.agent_inventory:
                # Key unlocks door
                edge.is_locked = False
            else:
                reason = f"Door locked (requires '{edge.required_key}')" if edge.required_key else "Door locked by environmental safety interlock."
                return False, reason

        # Move agent
        self.agent_node = target_node_id
        self.agent_pos = self.nodes[target_node_id].coordinates
        return True, f"Successfully moved to {target_node_id}."

    def pickup_item(self, item_name: str) -> Tuple[bool, str]:
        """Picks up an item from the current node."""
        current = self.nodes[self.agent_node]
        if item_name in current.items:
            current.items.remove(item_name)
            self.agent_inventory.add(item_name)
            return True, f"Picked up '{item_name}'."
        return False, f"Item '{item_name}' not found in {self.agent_node}."

    def vent_gas(self, node_id: str) -> Tuple[bool, str]:
        """Vents gas in a room, dropping pressure to safe levels (15%)."""
        if node_id not in self.nodes:
            return False, f"Node '{node_id}' not found."
        node = self.nodes[node_id]
        old_p = node.properties.get("pressure", 100.0)
        node.properties["pressure"] = 15.0
        self.update_lock_conditions()
        return True, f"Vented gas in {node_id}. Pressure dropped from {old_p}% to 15.0%."

    def set_reactor_power(self, node_id: str, power_pct: float) -> Tuple[bool, str]:
        """Adjusts reactor power output (0-100%)."""
        if node_id not in self.nodes:
            return False, f"Node '{node_id}' not found."
        if not (0.0 <= power_pct <= 100.0):
            return False, f"Power setting {power_pct}% invalid; must be between 0.0% and 100.0%."
        self.global_state["reactor_power"] = power_pct
        self.nodes[node_id].properties["reactor_power"] = power_pct
        return True, f"Reactor power adjusted to {power_pct}%."

    @classmethod
    def create_facility_env(cls) -> "SpatialEnvironmentGraph":
        """
        Creates the canonical Project Jane Facility:
        - Lab (0, 0): contains 'reactor_key'
        - Hallway (5, 0): central hub
        - Airlock (5, 5): pressure 65% (locked by pressure > 40%)
        - Reactor Core (10, 5): locked by 'reactor_key'
        - Armory (0, 5): side room
        - Continuous obstacles between rooms (e.g. wall pillars at (2.5, 0.5, 3.5, 4.5))
        """
        env = cls(name="Apex_Research_Facility")

        # Nodes
        env.add_node("lab", coordinates=(0.0, 0.0), properties={"type": "lab", "items": {"reactor_key"}})
        env.add_node("hallway", coordinates=(5.0, 0.0), properties={"type": "corridor"})
        env.add_node("armory", coordinates=(0.0, 5.0), properties={"type": "armory", "items": {"plasma_torch"}})
        env.add_node("reactor_airlock", coordinates=(5.0, 5.0), properties={"type": "airlock", "pressure": 65.0})
        env.add_node("reactor_core", coordinates=(10.0, 5.0), properties={"type": "reactor", "reactor_power": 0.0})

        # Primary Edges
        env.add_edge("lab", "hallway", is_locked=False)
        env.add_edge("lab", "armory", is_locked=False)
        env.add_edge("hallway", "reactor_airlock", is_locked=True, lock_condition="pressure > 40")
        env.add_edge("reactor_airlock", "reactor_core", is_locked=True, required_key="reactor_key")

        # Continuous Obstacles (Central support pillar between lab and airlock)
        env.add_obstacle("central_pillar", bounds=(2.0, 2.0, 3.5, 4.0))

        env.agent_node = "lab"
        env.agent_pos = (0.0, 0.0)
        env.update_lock_conditions()
        return env

    @classmethod
    def create_temporal_facility_env(cls) -> "SpatialEnvironmentGraph":
        """
        Creates the temporal facility environment equipped with a secondary bypass route
        (Armory <-> Maintenance Tunnel <-> Reactor Core) for reactive mid-step replanning.
        """
        env = cls.create_facility_env()
        env.add_node("maintenance_tunnel", coordinates=(0.0, 8.0), properties={"type": "tunnel"})
        env.add_edge("armory", "maintenance_tunnel", is_locked=False)
        env.add_edge("maintenance_tunnel", "reactor_core", is_locked=False)
        return env


