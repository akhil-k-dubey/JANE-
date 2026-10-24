"""Tier 1 Clingo ASP Solver: High-Level Discrete Logic & Topological Permissions."""

import clingo
from typing import Tuple, Optional
from .contracts import SpatialHypothesis, SpatialLogicHypothesis, VerificationResult


class ClingoSpatialVerifier:
    """
    Tier 1 Spatial Verifier: Clingo Answer Set Programming (ASP).
    Restricted to discrete logical permissions and topological graph transitions
    (rooms, zones, keys, locks) to mathematically prevent state-space explosion.
    """

    LOGIC_ASP_PROGRAM = """
    #show violation/2.

    % Fallback declarations to prevent ungrounded predicate warnings
    locked_zone(null_zone) :- #false.
    locked_transition(null_zone, null_zone) :- #false.
    key_held_zone(null_zone) :- #false.
    key_held_transition(null_zone, null_zone) :- #false.

    % Topological lock determination
    is_locked(A, B) :- transition(A, B), locked_zone(B).
    is_locked(A, B) :- transition(A, B), locked_transition(A, B).

    % Key clearance determination
    has_key(A, B) :- transition(A, B), key_held_zone(B).
    has_key(A, B) :- transition(A, B), key_held_transition(A, B).

    % Permission rules
    permitted_move(A, B) :- transition(A, B), not is_locked(A, B).
    permitted_move(A, B) :- transition(A, B), is_locked(A, B), has_key(A, B).

    % Violations
    violation(unauthorized_transition, I) :- step(I, A), step(I+1, B), not permitted_move(A, B).
    violation(invalid_start, 0) :- step(0, A), start(S), A != S.
    violation(goal_unreached, Last) :- max_step(Last), step(Last, A), goal(G), A != G.

    :- violation(_, _).
    """

    GRID_ASP_PROGRAM = """
    #show violation/2.

    % Fallback declarations to prevent ungrounded predicate warnings
    wall(-1, -1) :- #false.
    hazard(-1, -1) :- #false.

    violation(wall_collision, I)    :- step(I, X, Y), wall(X, Y).
    violation(hazard_entry, I)      :- step(I, X, Y), hazard(X, Y).
    violation(out_of_bounds, I)     :- step(I, X, Y), X < 0.
    violation(out_of_bounds, I)     :- step(I, X, Y), Y < 0.
    violation(out_of_bounds, I)     :- step(I, X, Y), grid_size(N), X >= N.
    violation(out_of_bounds, I)     :- step(I, X, Y), grid_size(N), Y >= N.
    violation(discontinuous_step, I):- step(I, X1, Y1), step(I+1, X2, Y2), |X1 - X2| + |Y1 - Y2| != 1.

    violation(invalid_start, 0)     :- step(0, X, Y), start(SX, SY), X != SX.
    violation(invalid_start, 0)     :- step(0, X, Y), start(SX, SY), Y != SY.
    violation(goal_unreached, Last) :- max_step(Last), step(Last, X, Y), goal(GX, GY), X != GX.
    violation(goal_unreached, Last) :- max_step(Last), step(Last, X, Y), goal(GX, GY), Y != GY.

    :- violation(_, _).
    """

    def verify_logic_permissions(self, hypothesis: SpatialLogicHypothesis) -> VerificationResult:
        """
        Tier 1: Proves logical permission and reachability across an abstracted zone graph.
        Solves in O(1) or small polynomial time without state-space explosion.
        """
        ctl = clingo.Control()
        ctl.add("base", [], self.LOGIC_ASP_PROGRAM)

        facts = []
        facts.append(f"start({hypothesis.start_zone}).")
        facts.append(f"goal({hypothesis.target_zone}).")
        facts.append(f"max_step({len(hypothesis.zone_sequence) - 1}).")

        for idx, z in enumerate(hypothesis.zone_sequence):
            facts.append(f"step({idx}, {z}).")

        for za, zb in hypothesis.allowed_transitions:
            facts.append(f"transition({za}, {zb}).")

        for lk in hypothesis.locked_doors:
            if isinstance(lk, (tuple, list)) and len(lk) == 2:
                facts.append(f"locked_transition({lk[0]}, {lk[1]}).")
            else:
                facts.append(f"locked_zone({lk}).")

        for kh in hypothesis.agent_keys:
            if isinstance(kh, (tuple, list)) and len(kh) == 2:
                facts.append(f"key_held_transition({kh[0]}, {kh[1]}).")
            else:
                facts.append(f"key_held_zone({kh}).")

        ctl.add("base", [], " ".join(facts))

        try:
            ctl.ground([("base", [])])
            result = ctl.solve()

            if result.satisfiable:
                return VerificationResult(
                    is_valid=True,
                    proof_log=f"Clingo ASP proved topological zone sequence '{hypothesis.start_zone} -> {hypothesis.target_zone}' is fully permitted.",
                    prover_name="Clingo_Logical_ASP"
                )
            else:
                return VerificationResult(
                    is_valid=False,
                    proof_log="Clingo ASP proved UNSAT: Unauthorized zone transition or locked door without key.",
                    prover_name="Clingo_Logical_ASP",
                    violation_details="Logical permission denied on zone transition sequence."
                )
        except Exception as e:
            return VerificationResult(
                is_valid=False,
                proof_log=f"Clingo logic solver error: {str(e)}",
                prover_name="Clingo_Logical_ASP",
                violation_details=str(e)
            )

    def verify_spatial(self, hypothesis: SpatialHypothesis) -> VerificationResult:
        """
        Verifies discrete spatial grid trajectory using Clingo ASP.
        """
        if not hypothesis.path:
            return VerificationResult(
                is_valid=False,
                proof_log="Empty trajectory provided.",
                prover_name="Clingo_ASP",
                violation_details="No steps in path."
            )

        ctl = clingo.Control()
        ctl.add("base", [], self.GRID_ASP_PROGRAM)

        facts = []
        facts.append(f"grid_size({hypothesis.grid_size}).")
        facts.append(f"start({hypothesis.start[0]}, {hypothesis.start[1]}).")
        facts.append(f"goal({hypothesis.goal[0]}, {hypothesis.goal[1]}).")
        facts.append(f"max_step({len(hypothesis.path) - 1}).")

        for idx, (x, y) in enumerate(hypothesis.path):
            facts.append(f"step({idx}, {x}, {y}).")

        for wx, wy in hypothesis.walls:
            facts.append(f"wall({wx}, {wy}).")

        for hx, hy in hypothesis.hazards:
            facts.append(f"hazard({hx}, {hy}).")

        ctl.add("base", [], " ".join(facts))

        try:
            ctl.ground([("base", [])])
            solve_result = ctl.solve()

            if solve_result.satisfiable:
                return VerificationResult(
                    is_valid=True,
                    proof_log=f"Trajectory formally proven collision-free and topologically valid by Clingo ASP (Steps: {len(hypothesis.path)}).",
                    prover_name="Clingo_ASP"
                )
            else:
                return VerificationResult(
                    is_valid=False,
                    proof_log="Clingo ASP solver proved UNSAT: Trajectory violates spatial rules.",
                    prover_name="Clingo_ASP",
                    violation_details="Trajectory collides with walls/hazards, exceeds boundaries, or is discontinuous."
                )

        except Exception as e:
            return VerificationResult(
                is_valid=False,
                proof_log=f"Clingo solver error: {str(e)}",
                prover_name="Clingo_ASP",
                violation_details=str(e)
            )
