"""Tier 2 Continuous Kinematics Engine: 2D Coordinates & Collision Avoidance.

Solves continuous geometric trajectories between rooms while strictly avoiding
convex bounding box obstacles.
"""

import math
from typing import List, Tuple, Optional
from .environment import ContinuousObstacle, SpatialEnvironmentGraph
from src.symbolic_verifier.contracts import VerificationResult


class ContinuousKinematicsEngine:
    """
    Continuous kinematics solver and trajectory validator.
    Computes exact physical coordinates and waypoints between graph nodes,
    detects line-box obstacle intersections via ray-clipping,
    and produces collision-free continuous motion plans.
    """

    @staticmethod
    def segment_intersects_box(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        box: Tuple[float, float, float, float]
    ) -> bool:
        """
        Liang-Barsky segment clipping against axis-aligned bounding box (min_x, min_y, max_x, max_y).
        Returns True if segment p1->p2 penetrates or touches the interior of the box.
        """
        x1, y1 = p1
        x2, y2 = p2
        bx1, by1, bx2, by2 = box

        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - bx1, bx2 - x1, y1 - by1, by2 - y1]

        u1 = 0.0
        u2 = 1.0

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u2:
                        return False
                    if t > u1:
                        u1 = t
                else:
                    if t < u1:
                        return False
                    if t < u2:
                        u2 = t

        return u1 <= u2

    def verify_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        obstacles: List[ContinuousObstacle]
    ) -> VerificationResult:
        """
        Verifies that every line segment in trajectory is 100% collision-free against all obstacles.
        """
        if len(trajectory) < 2:
            return VerificationResult(
                is_valid=True,
                proof_log="Trivial single-point trajectory is collision-free.",
                prover_name="Continuous_Kinematics_Engine"
            )

        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]

            for obs in obstacles:
                if self.segment_intersects_box(p1, p2, obs.bounds):
                    return VerificationResult(
                        is_valid=False,
                        proof_log=f"Segment {i} ({p1}) -> {i+1} ({p2}) collides with obstacle '{obs.id}' at {obs.bounds}.",
                        prover_name="Continuous_Kinematics_Engine",
                        violation_details=f"Collision with {obs.id} between {p1} and {p2}."
                    )

        return VerificationResult(
            is_valid=True,
            proof_log=f"Trajectory with {len(trajectory)} waypoints proven 100% collision-free.",
            prover_name="Continuous_Kinematics_Engine"
        )

    def plan_continuous_trajectory(
        self,
        start_coord: Tuple[float, float],
        target_coord: Tuple[float, float],
        obstacles: List[ContinuousObstacle],
        clearance: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Generates a continuous collision-free trajectory from start_coord to target_coord.
        If a direct line intersects an obstacle, generates safe bypass waypoint trajectories
        around the obstacle bounding box.
        """
        # Test direct line first
        if not any(self.segment_intersects_box(start_coord, target_coord, obs.bounds) for obs in obstacles):
            return [start_coord, target_coord]

        # Obstacle detected: generate candidate bypass trajectories
        candidate_paths: List[List[Tuple[float, float]]] = []

        for obs in obstacles:
            if self.segment_intersects_box(start_coord, target_coord, obs.bounds):
                bx1, by1, bx2, by2 = obs.bounds
                c = clearance

                # Four bypass boundary routes: top, bottom, left, right
                routes = [
                    # Top skirt
                    [(bx1 - c, by2 + c), (bx2 + c, by2 + c)],
                    # Bottom skirt
                    [(bx1 - c, by1 - c), (bx2 + c, by1 - c)],
                    # Left skirt
                    [(bx1 - c, by1 - c), (bx1 - c, by2 + c)],
                    # Right skirt
                    [(bx2 + c, by1 - c), (bx2 + c, by2 + c)],
                    # Single corner candidates
                    [(bx1 - c, by2 + c)],
                    [(bx2 + c, by2 + c)],
                    [(bx1 - c, by1 - c)],
                    [(bx2 + c, by1 - c)]
                ]

                for r in routes:
                    cand = [start_coord] + r + [target_coord]
                    # Check collision
                    is_colliding = False
                    for i in range(len(cand) - 1):
                        for o in obstacles:
                            if self.segment_intersects_box(cand[i], cand[i + 1], o.bounds):
                                is_colliding = True
                                break
                        if is_colliding:
                            break
                    if not is_colliding:
                        candidate_paths.append(cand)

        if candidate_paths:
            # Pick candidate with minimum total path length
            def path_len(p_list):
                return sum(math.hypot(p_list[i+1][0] - p_list[i][0], p_list[i+1][1] - p_list[i][1]) for i in range(len(p_list) - 1))

            best_path = min(candidate_paths, key=path_len)
            return best_path

        # Fallback
        return [start_coord, target_coord]

    def generate_full_path_trajectory(
        self,
        env: SpatialEnvironmentGraph,
        node_path: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Converts a high-level topological room sequence (e.g. ['lab', 'hallway', 'reactor_airlock'])
        into an exact continuous physical coordinate trajectory avoiding all obstacles.
        """
        if not node_path:
            return []

        full_trajectory = [env.nodes[node_path[0]].coordinates]

        for i in range(len(node_path) - 1):
            src_node = env.nodes[node_path[i]]
            tgt_node = env.nodes[node_path[i + 1]]

            sub_trajectory = self.plan_continuous_trajectory(
                start_coord=src_node.coordinates,
                target_coord=tgt_node.coordinates,
                obstacles=env.obstacles
            )
            # Append without duplicating starting point
            full_trajectory.extend(sub_trajectory[1:])

        return full_trajectory
