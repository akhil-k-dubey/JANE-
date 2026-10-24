"""Jane's Agent Controller in the 10x10 Sandboxed Spatial Grid World."""

import os
from collections import deque
import torch
import torch.nn as nn
from typing import Tuple, List, Set, Dict, Any, Optional

from .physics_engine import GridPhysicsEngine
from .reward_function import GridRewardFunction


class JaneGridAgent:
    """
    Agent controller embodying Jane inside the 10x10 spatial grid.
    Encodes spatial coordinates into latent manifold concepts, executes policy actions,
    and supports deterministic optimal path planning.
    """

    def __init__(
        self,
        physics: Optional[GridPhysicsEngine] = None,
        reward_fn: Optional[GridRewardFunction] = None,
        latent_dim: int = 256
    ):
        self.physics = physics or GridPhysicsEngine()
        self.reward_fn = reward_fn or GridRewardFunction()
        self.latent_dim = latent_dim

        # Coordinate embedding to latent space
        self.pos_encoder = nn.Linear(4, latent_dim)
        self.policy_head = nn.Linear(latent_dim, 4)

    def load_checkpoint(self, checkpoint_path: str):
        """Loads trained RL weights if available."""
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.pos_encoder.load_state_dict(state["pos_encoder"])
            self.policy_head.load_state_dict(state["policy_head"])

    def get_observation_tensor(
        self,
        current_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        vec = [
            current_pos[0] / 9.0,
            current_pos[1] / 9.0,
            goal_pos[0] / 9.0,
            goal_pos[1] / 9.0
        ]
        return torch.tensor([vec], dtype=torch.float32, device=device)

    def select_action(self, obs_tensor: torch.Tensor, epsilon: float = 0.0) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 4, (1,)).item()
        with torch.no_grad():
            latent = torch.relu(self.pos_encoder(obs_tensor))
            logits = self.policy_head(latent)
            return logits.argmax(dim=-1).item()

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Computes an optimal, collision-free spatial path using BFS on the physics engine topology.
        """
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            if curr == goal:
                return path

            for action_idx, (dx, dy) in self.physics.ACTIONS.items():
                next_pos = (curr[0] + dx, curr[1] + dy)
                if (
                    self.physics.is_valid_position(next_pos)
                    and next_pos not in visited
                    and next_pos not in self.physics.hazards
                ):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return [start]

    def run_episode(
        self,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        max_steps: int = 50,
        epsilon: float = 0.0
    ) -> Dict[str, Any]:
        current_pos = start_pos
        path = [current_pos]
        total_reward = 0.0
        done = False

        for _ in range(max_steps):
            obs = self.get_observation_tensor(current_pos, goal_pos)
            action = self.select_action(obs, epsilon=epsilon)
            next_pos, is_hazard, event = self.physics.step(current_pos, action)
            reward, done = self.reward_fn.calculate_reward(next_pos, goal_pos, event, is_hazard)

            total_reward += reward
            current_pos = next_pos
            path.append(current_pos)

            if done:
                break

        return {
            "success": current_pos == goal_pos,
            "path": path,
            "total_reward": total_reward,
            "steps": len(path) - 1
        }
