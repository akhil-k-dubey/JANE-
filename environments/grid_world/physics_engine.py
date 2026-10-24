"""Hard Physics Rules for the 10x10 Sandboxed Spatial Grid World."""

from typing import Tuple, Set, Optional

class GridPhysicsEngine:
    """
    Enforces strict physical laws in the grid world:
    - Inviolable outer boundaries [0, 9] x [0, 9]
    - Impassable solid walls
    - Lethal hazard zones
    - Discrete 4-directional kinematics
    """

    ACTIONS = {
        0: (0, 1),   # Up / North
        1: (1, 0),   # Right / East
        2: (0, -1),  # Down / South
        3: (-1, 0)   # Left / West
    }

    def __init__(self, size: int = 10):
        self.size = size
        self.walls: Set[Tuple[int, int]] = set()
        self.hazards: Set[Tuple[int, int]] = set()

    def set_map(self, walls: Set[Tuple[int, int]], hazards: Set[Tuple[int, int]]):
        self.walls = set(walls)
        self.hazards = set(hazards)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and pos not in self.walls

    def step(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], bool, str]:
        """
        Executes a physics transition.
        Returns:
            next_pos: new position (bounces off walls / stays in place)
            is_hazard: True if stepped into a lethal hazard
            event: status string
        """
        dx, dy = self.ACTIONS.get(action, (0, 0))
        target_pos = (current_pos[0] + dx, current_pos[1] + dy)

        # 1. Boundary check (Energy wall bounce)
        if not (0 <= target_pos[0] < self.size and 0 <= target_pos[1] < self.size):
            return current_pos, False, "wall_bounce"

        # 2. Solid wall check
        if target_pos in self.walls:
            return current_pos, False, "wall_bounce"

        # 3. Hazard check
        if target_pos in self.hazards:
            return target_pos, True, "hazard_collision"

        return target_pos, False, "normal_move"
