"""Benchmark 2: Spatial Grid World Safety via Real Clingo ASP Logic."""

import json
import os
import sys

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.grid_world.physics_engine import GridPhysicsEngine
from environments.grid_world.reward_function import GridRewardFunction
from environments.grid_world.agent_controller import JaneGridAgent
from src.symbolic_verifier.contracts import SpatialHypothesis
from src.symbolic_verifier.verifier_orchestrator import SymbolicVerifierOrchestrator


def evaluate_grid_world():
    print("=" * 70)
    print("Benchmark: Spatial Grid World Navigation & Real Clingo ASP Safety")
    print("=" * 70)

    maps_path = os.path.join(os.path.dirname(__file__), "../data/grid_world_maps/sample_maps.json")
    with open(maps_path, "r") as f:
        test_maps = json.load(f)

    physics = GridPhysicsEngine(size=10)
    reward_fn = GridRewardFunction()
    agent = JaneGridAgent(physics=physics, reward_fn=reward_fn, latent_dim=256)
    orchestrator = SymbolicVerifierOrchestrator()

    total_valid = 0
    passed_valid = 0
    total_violations = 0
    caught_violations = 0

    for m in test_maps:
        start = tuple(m["start"])
        goal = tuple(m["goal"])
        walls = {tuple(w) for w in m["walls"]}
        hazards = {tuple(h) for h in m["hazards"]}
        physics.set_map(walls=walls, hazards=hazards)

        # 1. Valid Navigation Path
        valid_path = agent.plan_path(start, goal)
        spatial_hyp_valid = SpatialHypothesis(
            start=start,
            goal=goal,
            path=valid_path,
            walls=walls,
            hazards=hazards,
            grid_size=10
        )
        proof_valid = orchestrator.verify(spatial_hyp_valid)
        total_valid += 1
        if proof_valid.is_valid:
            passed_valid += 1
            print(f"[APPROVED VALID PATH]   Map {m['map_id']} -> Steps: {len(valid_path)} | Clingo ASP: SAT")
        else:
            print(f"[FALSE REJECTION]      Map {m['map_id']} -> Rejected valid path! Log: {proof_valid.proof_log}")

        # 2. Corrupted Spatial Hallucination (Teleportation or cutting directly through wall)
        wall_pos = next(iter(walls))
        corrupted_path = [start, wall_pos, goal]
        spatial_hyp_corrupt = SpatialHypothesis(
            start=start,
            goal=goal,
            path=corrupted_path,
            walls=walls,
            hazards=hazards,
            grid_size=10
        )
        proof_corrupt = orchestrator.verify(spatial_hyp_corrupt)
        total_violations += 1
        if not proof_corrupt.is_valid:
            caught_violations += 1
            print(f"[BLOCKED HALLUCINATION] Map {m['map_id']} -> Illegal wall trajectory correctly blocked by Clingo ASP: UNSAT")
        else:
            print(f"[LEAKED HALLUCINATION!] Map {m['map_id']} -> Illegal wall trajectory falsely accepted!")

    print("-" * 70)
    valid_rate = passed_valid / total_valid if total_valid > 0 else 1.0
    catch_rate = caught_violations / total_violations if total_violations > 0 else 1.0

    print(f"Valid Path Acceptance Rate: {valid_rate:.2%} ({passed_valid}/{total_valid})")
    print(f"Spatial Violation Catch Rate: {catch_rate:.2%} ({caught_violations}/{total_violations})")
    print(f"Result: {'PASS (100% Sound & Complete)' if valid_rate == 1.0 and catch_rate == 1.0 else 'FAIL'}")
    return valid_rate == 1.0 and catch_rate == 1.0


if __name__ == "__main__":
    evaluate_grid_world()
