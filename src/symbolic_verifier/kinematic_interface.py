"""Tier 2 Continuous Kinematics & Geometric Collision Solver."""

from typing import Tuple, List, Optional
from .contracts import KinematicTrajectoryHypothesis, VerificationResult


class ContinuousKinematicsVerifier:
    """
    Tier 2 Spatial Verifier: Continuous Kinematics & Collision Geometry.
    Performs continuous segment-box intersection tests across continuous Euclidean coordinates.
    Guarantees no state-space explosion because it solves in continuous space in O(N * M) time.
    """

    @staticmethod
    def _line_intersects_box(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        box: Tuple[float, float, float, float]
    ) -> bool:
        """
        Liang-Barsky / Cohen-Sutherland line clipping test to detect if line segment
        (p1 -> p2) penetrates bounding box (min_x, min_y, max_x, max_y).
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
            if abs(p[i]) < 1e-9:
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

    def verify_trajectory(self, hypothesis: KinematicTrajectoryHypothesis) -> VerificationResult:
        """
        Verifies that a continuous trajectory [p_0, p_1, ..., p_n] is collision-free
        against convex obstacle bounding boxes.
        """
        path = [hypothesis.start_pos] + hypothesis.waypoints + [hypothesis.target_pos]
        min_gx, min_gy, max_gx, max_gy = hypothesis.bounds

        for i, pt in enumerate(path):
            x, y = pt
            # Boundary check
            if not (min_gx <= x <= max_gx and min_gy <= y <= max_gy):
                return VerificationResult(
                    is_valid=False,
                    proof_log=f"Waypoint {i} ({x:.2f}, {y:.2f}) out of continuous bounds.",
                    prover_name="Continuous_Kinematics",
                    violation_details=f"Spatial boundary violation at step {i}."
                )

        # Continuous collision check for every path segment against all obstacles
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            for obs_idx, obs in enumerate(hypothesis.obstacles):
                if self._line_intersects_box(p1, p2, obs):
                    return VerificationResult(
                        is_valid=False,
                        proof_log=f"Segment {i} -> {i+1} intersects obstacle {obs_idx} {obs}.",
                        prover_name="Continuous_Kinematics",
                        violation_details=f"Collision detected between {p1} and {p2} with obstacle {obs}."
                    )

        return VerificationResult(
            is_valid=True,
            proof_log=f"Continuous trajectory of {len(path)} points proven 100% collision-free.",
            prover_name="Continuous_Kinematics"
        )
