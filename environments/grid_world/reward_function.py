"""Reinforcement Learning Reward Function for Grid World."""

from typing import Tuple

class GridRewardFunction:
    """
    RL scoring system designed to encourage:
    - Efficient goal navigation (+10.0)
    - Strong hazard avoidance (-10.0)
    - Minimization of redundant steps (-0.1 step penalty)
    - Disincentivizing wall collisions (-0.5 wall bump penalty)
    """

    def __init__(
        self,
        goal_reward: float = 10.0,
        hazard_penalty: float = -10.0,
        step_penalty: float = -0.1,
        wall_penalty: float = -0.5
    ):
        self.goal_reward = goal_reward
        self.hazard_penalty = hazard_penalty
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty

    def calculate_reward(
        self,
        current_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        event: str,
        is_hazard: bool
    ) -> Tuple[float, bool]:
        """
        Returns:
            reward: scalar reward
            done: episode termination flag
        """
        if current_pos == goal_pos:
            return self.goal_reward, True

        if is_hazard:
            return self.hazard_penalty, True

        if event == "wall_bounce":
            return self.step_penalty + self.wall_penalty, False

        return self.step_penalty, False
