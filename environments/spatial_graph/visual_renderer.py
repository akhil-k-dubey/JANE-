"""Synthetic Visual Camera & Floorplan Renderer for Spatial Graph.

Generates high-fidelity top-down RGB image tensors [3, H, W] from the live
SpatialEnvironmentGraph state for Jane's Vision Transformer.
Also supports synthetic visual hallucination injection to test neuro-symbolic safety boundaries.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import math
from .environment import SpatialEnvironmentGraph, ContinuousObstacle


class FacilityVisualRenderer:
    """
    Renders top-down camera feeds / floorplan schematics from live Python graph state.
    Produces pure PyTorch RGB image tensors in [0, 1] range.
    """

    def __init__(self, img_size: int = 128, max_coord: float = 12.0):
        self.img_size = img_size
        self.max_coord = max_coord

    def _coord_to_pixel(self, coord: Tuple[float, float]) -> Tuple[int, int]:
        """Converts continuous (x, y) coordinates to image pixel (row, col)."""
        x, y = coord
        col = int(max(0.0, min(float(self.img_size - 1), x / self.max_coord * (self.img_size - 1))))
        row = int(max(0.0, min(float(self.img_size - 1), y / self.max_coord * (self.img_size - 1))))
        return row, col

    def render_environment(
        self,
        env: SpatialEnvironmentGraph,
        inject_fake_door_through_obstacle: bool = False
    ) -> torch.Tensor:
        """
        Renders the current environment state into an RGB tensor [3, H, W].
        """
        H, W = self.img_size, self.img_size
        # Background: Dark high-tech facility floor
        img = torch.full((3, H, W), 0.08, dtype=torch.float32)

        # 1. Render Rooms as distinct colored zones
        room_colors = {
            "lab": torch.tensor([0.15, 0.35, 0.70]),           # High-tech Blue
            "hallway": torch.tensor([0.25, 0.25, 0.28]),       # Slate Corridor
            "armory": torch.tensor([0.45, 0.20, 0.55]),        # Purple Armory
            "reactor_airlock": torch.tensor([0.70, 0.45, 0.15]), # Caution Amber
            "reactor_core": torch.tensor([0.80, 0.15, 0.15])   # Glowing Red Core
        }

        room_radius = int(self.img_size * 0.12)
        for nid, node in env.nodes.items():
            r, c = self._coord_to_pixel(node.coordinates)
            color = room_colors.get(nid, torch.tensor([0.3, 0.3, 0.3]))
            # Fill circular room zone
            for dr in range(-room_radius, room_radius + 1):
                for dc in range(-room_radius, room_radius + 1):
                    if dr*dr + dc*dc <= room_radius*room_radius:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            img[:, nr, nc] = color

        # 2. Render Navigable Edges (Corridor paths)
        for edge in env.edges:
            p1 = self._coord_to_pixel(env.nodes[edge.source].coordinates)
            p2 = self._coord_to_pixel(env.nodes[edge.target].coordinates)
            # Draw connecting line
            steps = 40
            edge_color = torch.tensor([0.8, 0.2, 0.2]) if edge.is_locked else torch.tensor([0.2, 0.8, 0.3])
            for s in range(steps):
                t = s / steps
                lr = int(p1[0] * (1 - t) + p2[0] * t)
                lc = int(p1[1] * (1 - t) + p2[1] * t)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if 0 <= lr + dr < H and 0 <= lc + dc < W:
                            # Blend edge color
                            img[:, lr + dr, lc + dc] = 0.5 * img[:, lr + dr, lc + dc] + 0.5 * edge_color

        # 3. Render Solid Obstacles (Central support pillar)
        for obs in env.obstacles:
            r1, c1 = self._coord_to_pixel((obs.min_x, obs.min_y))
            r2, c2 = self._coord_to_pixel((obs.max_x, obs.max_y))
            min_r, max_r = min(r1, r2), max(r1, r2)
            min_c, max_c = min(c1, c2), max(c1, c2)
            # Solid dark-metal obstacle
            img[:, min_r:max_r+1, min_c:max_c+1] = torch.tensor([0.02, 0.02, 0.04]).view(3, 1, 1)

        # 4. Render Items (e.g. reactor_key: Golden Yellow spark)
        for nid, node in env.nodes.items():
            if "reactor_key" in node.items:
                r, c = self._coord_to_pixel(node.coordinates)
                key_color = torch.tensor([1.0, 0.85, 0.1])
                for dr in (-2, -1, 0, 1, 2):
                    for dc in (-2, -1, 0, 1, 2):
                        if 0 <= r + dr < H and 0 <= c + dc < W:
                            img[:, r + dr, c + dc] = key_color

        # 5. Render Agent (Cyan Beacon)
        ar, ac = self._coord_to_pixel(env.agent_pos)
        agent_color = torch.tensor([0.1, 0.95, 0.95])
        for dr in (-3, -2, -1, 0, 1, 2, 3):
            for dc in (-3, -2, -1, 0, 1, 2, 3):
                if dr*dr + dc*dc <= 9:
                    if 0 <= ar + dr < H and 0 <= ac + dc < W:
                        img[:, ar + dr, ac + dc] = agent_color

        # 6. Intentional Vision Hallucination Injection:
        # If enabled, paints a vivid bright-green "open door" right across the solid obstacle!
        if inject_fake_door_through_obstacle and env.obstacles:
            obs = env.obstacles[0]
            center_x = (obs.min_x + obs.max_x) * 0.5
            center_y = (obs.min_y + obs.max_y) * 0.5
            hr, hc = self._coord_to_pixel((center_x, center_y))
            hallucinated_door_color = torch.tensor([0.0, 1.0, 0.2])  # Glowing bright green
            for dr in range(-4, 5):
                for dc in range(-8, 9):
                    if 0 <= hr + dr < H and 0 <= hc + dc < W:
                        img[:, hr + dr, hc + dc] = hallucinated_door_color

        return img.clamp(0.0, 1.0)

    def render_ascii_hud(
        self,
        env: SpatialEnvironmentGraph,
        detected_objects: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Renders an ASCII HUD showing real-time facility floorplan,
        agent position, locks, and vision detections.
        """
        curr = env.agent_node
        p_airlock = env.nodes["reactor_airlock"].pressure if "reactor_airlock" in env.nodes else 0.0

        lab_tag = "[LAB *YOU*]" if curr == "lab" else "[  LAB  ]"
        hall_tag = "[HALLWAY *YOU*]" if curr == "hallway" else "[HALLWAY]"
        armory_tag = "[ARMORY *YOU*]" if curr == "armory" else "[ARMORY]"
        airlock_tag = f"[AIRLOCK *YOU* P:{p_airlock:.0f}%]" if curr == "reactor_airlock" else f"[AIRLOCK P:{p_airlock:.0f}%]"
        core_tag = "[CORE 🔒 *YOU*]" if curr == "reactor_core" else "[CORE 🔒]"

        # Interlock status
        lock_status = "LOCKED (Pressure > 40%)" if p_airlock > 40.0 else "UNLOCKED (Pressure Safe)"

        lines = [
            "  ┌─────────────────────────────────────────────────────────────────────────────┐",
            f"  │ 👁️ [JANE MULTIMODAL SIGHT HUD] Agent Location: {curr.upper():<20}       │",
            f"  │   {lab_tag} ─────── {hall_tag} ─────── {armory_tag}                   │",
            f"  │                         │                     │                             │",
            f"  │                    [■ PILLAR]                 ▼                             │",
            f"  │                         │          {airlock_tag} ── {core_tag}       │",
            f"  │                         └── Door: {lock_status:<38} │",
            "  │   Vision Stream: 128x128 RGB Camera -> ViT -> 256D Manifold Aligned         │"
        ]
        if detected_objects:
            obj_strs = [f"{d['concept']}(conf:{d.get('confidence', 0):.2f})" for d in detected_objects[:2]]
            lines.append(f"  │   Detected Features: {', '.join(obj_strs):<53} │")
        lines.append("  └─────────────────────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)
