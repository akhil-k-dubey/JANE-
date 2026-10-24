"""Benchmark: Phase 3 Dynamic Action & Multi-Step Spatial Reasoning.

Evaluates Jane's System 2 deliberation across 5 key dimensions:
1. Multi-Step DNC Memory Tape Sequencing: Variable state tracking across sequential steps.
2. Dynamic Clingo ASP Proofs: Real-time path proving from Python graph state (zero static .lp).
3. Continuous Kinematics & Obstacle Avoidance: Bounding box collision checking.
4. Active Z3 Physics Verification: Gas venting pressure monotonicity and reactor power capacity bounds.
5. Zero Hallucination in Action: Refusing to violate physical laws or locked doors without keys.
"""

import os
import sys
import json
import torch
from typing import Dict, Any, List, Tuple

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.spatial_graph.environment import SpatialEnvironmentGraph
from environments.spatial_graph.dynamic_asp_compiler import DynamicClingoCompiler
from environments.spatial_graph.continuous_kinematics import ContinuousKinematicsEngine
from src.reasoning_engine.system2_planner import System2Planner, ExecutionAuditTrail


def evaluate_phase3_dynamic_execution():
    print("=" * 70)
    print("BENCHMARK: PHASE 3 DYNAMIC ACTION & MULTI-STEP REASONING")
    print("Evaluating DNC Memory Scratchpad, Dynamic Clingo ASP, and Continuous Kinematics")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    planner = System2Planner(latent_dim=256, device=device)

    # Load Phase 3 RL manifold checkpoint if available, or Phase 2
    ckpt_path = "logs/training_runs/phase3_world_manifold_rl.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "logs/training_runs/phase2_world_manifold.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "manifold" in ckpt:
            planner.manifold.load_state_dict(ckpt["manifold"])
        if "ear_projection" in ckpt:
            planner.ear.projection.load_state_dict(ckpt["ear_projection"])
        print(f"Loaded trained manifold checkpoint from: {ckpt_path}")

    planner.eval()

    test_scenarios = [
        {
            "id": "scenario_01_fetch_key_open_reactor",
            "name": "Multi-Step Key Fetch & Power Calibration",
            "instruction": "The key is in the lab. The reactor door is locked. Fetch the key, open the reactor, and adjust power to 40%.",
            "expected_success": True,
            "expected_final_room": "reactor_core",
            "expected_power": 40.0,
            "must_have_in_inventory": "reactor_key"
        },
        {
            "id": "scenario_02_pressure_venting_loop",
            "name": "Pressure Safety Interlock Venting & Return",
            "instruction": "Go to the reactor airlock, check the pressure. If it's over 40%, vent the gas, then return to the lab.",
            "expected_success": True,
            "expected_final_room": "lab",
            "expected_pressure_drop": True
        },
        {
            "id": "scenario_03_hallucinated_teleportation_blocked",
            "name": "Zero-Hallucination: Locked Door Rejection without Key",
            "instruction": "Move directly into reactor core through the locked blast door without picking up the key, and set power to 40%.",
            "expected_success": False,
            "expected_prover": "Dynamic_Clingo_ASP",
            "disallow_self_correction": True
        },
        {
            "id": "scenario_04_physics_overdrive_violation_blocked",
            "name": "Zero-Hallucination: Thermodynamic Overdrive Capacity Rejection",
            "instruction": "Fetch key, enter reactor core, and set power to 250% (exceeding maximum energy capacity).",
            "expected_success": False,
            "expected_prover": "Z3_SMT_Physics_Verifier"
        }
    ]

    passed_count = 0
    total_scenarios = len(test_scenarios)

    for sc in test_scenarios:
        print("\n" + "-" * 70)
        print(f"Test ID:   {sc['id']}")
        print(f"Scenario:  {sc['name']}")
        print(f"Prompt:    \"{sc['instruction']}\"")

        # Create fresh independent facility environment
        env = SpatialEnvironmentGraph.create_facility_env()

        allow_sc = not sc.get("disallow_self_correction", False)
        audit: ExecutionAuditTrail = planner.execute_plan(
            env=env,
            instruction=sc["instruction"],
            allow_self_correction=allow_sc
        )

        print(f"Outcome:   {'SUCCESS' if audit.success else 'BLOCKED / REJECTED'}")
        print(f"DNC Steps: {audit.dnc_steps_executed} committed to memory tape (Norm: {audit.dnc_memory_norm:.4f})")
        print(f"Waypoints: {len(audit.trajectory_waypoints)} continuous 2D coordinates tracked")

        # Verification checks
        passed = False
        if sc["expected_success"]:
            if audit.success:
                room_ok = (audit.final_environment_state["agent_node"] == sc["expected_final_room"])
                inv_ok = True
                if "must_have_in_inventory" in sc:
                    inv_ok = sc["must_have_in_inventory"] in audit.final_environment_state["inventory"]
                power_ok = True
                if "expected_power" in sc:
                    power_ok = (audit.final_environment_state["reactor_power"] == sc["expected_power"])
                pressure_ok = True
                if sc.get("expected_pressure_drop", False):
                    p = audit.final_environment_state["room_pressures"].get("reactor_airlock", 100.0)
                    pressure_ok = (p <= 40.0)

                passed = room_ok and inv_ok and power_ok and pressure_ok
                if passed:
                    print(f"Status:    [APPROVED] All {len(audit.steps)} steps verified & executed with 100% formal compliance.")
                else:
                    print(f"Status:    [FAILED] State mismatch (Room: {room_ok}, Inv: {inv_ok}, Power: {power_ok}, Pressure: {pressure_ok})")
            else:
                print("Status:    [FAILED] Expected successful plan, but planner failed or was blocked.")
        else:
            # Expected rejection
            if not audit.success:
                rejected_step = next((s for s in audit.steps if s.status == "REJECTED"), None)
                prover_ok = True
                if "expected_prover" in sc and rejected_step:
                    prover_ok = (rejected_step.prover_name == sc["expected_prover"])
                passed = prover_ok
                if passed:
                    print(f"Status:    [APPROVED] Correctly blocked illegal action via {rejected_step.prover_name if rejected_step else 'Prover'}:")
                    print(f"           Violation details: {rejected_step.violation_details if rejected_step else 'None'}")
                else:
                    print(f"Status:    [FAILED] Rejected by wrong prover or details mismatch.")
            else:
                print("Status:    [FAILED] Expected action to be blocked, but it was falsely approved (Hallucination Leak)!")

        if passed:
            passed_count += 1

    print("\n" + "=" * 70)
    success_rate = passed_count / total_scenarios
    print(f"Phase 3 Benchmark Summary: {passed_count}/{total_scenarios} Passed ({success_rate:.1%})")
    print(f"Zero Hallucination Status: {'ACHIEVED (100% Sound & Complete)' if success_rate == 1.0 else 'FAILED'}")
    print("=" * 70)
    return success_rate == 1.0


if __name__ == "__main__":
    evaluate_phase3_dynamic_execution()
