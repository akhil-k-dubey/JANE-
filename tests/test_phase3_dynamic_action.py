"""Comprehensive Unit & System Tests for Project Jane Phase 3 (Dynamic Action & Multi-Step Reasoning)."""

import pytest
import torch

from environments.spatial_graph.environment import (
    GraphNode,
    GraphEdge,
    ContinuousObstacle,
    SpatialEnvironmentGraph
)
from environments.spatial_graph.dynamic_asp_compiler import DynamicClingoCompiler
from environments.spatial_graph.continuous_kinematics import ContinuousKinematicsEngine
from src.reasoning_engine.system2_planner import System2Planner, PlanStep


def test_spatial_environment_graph_dynamic_state_machine():
    env = SpatialEnvironmentGraph.create_facility_env()

    # 1. Initial State: Agent is in lab, airlock pressure is 65% (locked), reactor is locked
    assert env.agent_node == "lab"
    assert env.nodes["reactor_airlock"].pressure == 65.0
    airlock_edge = next(e for e in env.edges if e.source == "hallway" and e.target == "reactor_airlock")
    assert airlock_edge.is_locked is True

    # 2. Moving to hallway is permitted (unlocked)
    ok, msg = env.move_to("hallway")
    assert ok is True
    assert env.agent_node == "hallway"

    # 3. Moving to airlock fails before venting gas
    ok, msg = env.move_to("reactor_airlock")
    assert ok is False
    assert "locked" in msg.lower()

    # 4. Vent gas: pressure drops to 15%, clearing the dynamic lock condition
    ok_vent, msg_vent = env.vent_gas("reactor_airlock")
    assert ok_vent is True
    assert env.nodes["reactor_airlock"].pressure == 15.0
    assert airlock_edge.is_locked is False

    # 5. Moving to airlock now succeeds!
    ok_move, msg_move = env.move_to("reactor_airlock")
    assert ok_move is True
    assert env.agent_node == "reactor_airlock"

    # 6. Moving to reactor core fails without key
    ok_core, msg_core = env.move_to("reactor_core")
    assert ok_core is False
    assert "requires 'reactor_key'" in msg_core

    # 7. Pickup key in lab and unlock reactor
    env.agent_inventory.add("reactor_key")
    ok_core_with_key, _ = env.move_to("reactor_core")
    assert ok_core_with_key is True
    assert env.agent_node == "reactor_core"


def test_dynamic_asp_compiler_runtime_generation():
    compiler = DynamicClingoCompiler()
    env = SpatialEnvironmentGraph.create_facility_env()

    # 1. Compile state into runtime Clingo code (zero static .lp files)
    asp_code = compiler.compile_state(env, start_node="lab", goal_node="reactor_core")
    assert "node(lab)." in asp_code
    assert "node(reactor_core)." in asp_code
    assert "locked(reactor_airlock, reactor_core)." in asp_code
    assert "requires_key(reactor_airlock, reactor_core, reactor_key)." in asp_code

    # 2. Query Clingo: reaching reactor_core from lab without key is UNSAT
    proof_unreachable = compiler.verify_reachability(env, start_node="lab", goal_node="reactor_core")
    assert proof_unreachable.is_valid is False
    assert "UNSAT" in proof_unreachable.proof_log

    # 3. Dynamic State Mutation: Add key and vent gas
    env.agent_inventory.add("reactor_key")
    env.vent_gas("reactor_airlock")

    # 4. Re-compile: locked rule disappears or is unlocked by has_key
    proof_reachable = compiler.verify_reachability(env, start_node="lab", goal_node="reactor_core")
    assert proof_reachable.is_valid is True
    assert "proven reachable" in proof_reachable.proof_log.lower()

    # 5. Discrete shortest path finding
    path = compiler.find_path(env, start_node="lab", goal_node="reactor_core")
    assert path == ["lab", "hallway", "reactor_airlock", "reactor_core"]


def test_continuous_kinematics_obstacle_avoidance():
    engine = ContinuousKinematicsEngine()
    obstacles = [ContinuousObstacle(id="pillar", bounds=(2.0, 2.0, 4.0, 4.0))]

    # 1. Direct collision: segment from (1.0, 3.0) to (5.0, 3.0) cuts straight through x in [2, 4] at y=3
    colliding_traj = [(1.0, 3.0), (5.0, 3.0)]
    proof_colliding = engine.verify_trajectory(colliding_traj, obstacles)
    assert proof_colliding.is_valid is False
    assert "collides with obstacle" in proof_colliding.proof_log

    # 2. Collision-free trajectory: clear line
    safe_traj = [(0.0, 0.0), (5.0, 0.0)]
    proof_safe = engine.verify_trajectory(safe_traj, obstacles)
    assert proof_safe.is_valid is True

    # 3. Automatic detour waypoint planning around obstacle
    detour_traj = engine.plan_continuous_trajectory(
        start_coord=(1.0, 3.0),
        target_coord=(5.0, 3.0),
        obstacles=obstacles
    )
    assert len(detour_traj) >= 3  # Start -> Corner Waypoint -> Target
    proof_detour = engine.verify_trajectory(detour_traj, obstacles)
    assert proof_detour.is_valid is True


def test_dnc_scratchpad_memory_sequencing():
    planner = System2Planner(latent_dim=256, memory_slots=16, memory_dim=32, device="cpu")
    env = SpatialEnvironmentGraph.create_facility_env()

    # Multi-step command
    task = "The key is in the lab. Fetch the key, open the reactor, and adjust power to 40%."
    audit = planner.execute_plan(env, task, allow_self_correction=True)

    assert audit.success is True
    assert audit.dnc_steps_executed >= 3
    # Memory tape was actively modified
    assert audit.dnc_memory_norm > 0.0
    assert audit.final_environment_state["reactor_power"] == 40.0
    assert "reactor_key" in audit.final_environment_state["inventory"]


def test_zero_hallucination_physics_boundary_rejection():
    planner = System2Planner(latent_dim=256, device="cpu")
    env = SpatialEnvironmentGraph.create_facility_env()

    # Command violating thermodynamic capacity (setting power to 250%)
    invalid_task = "Fetch key, enter reactor core, and set power to 250% (exceeding maximum energy capacity)."
    audit = planner.execute_plan(env, invalid_task, allow_self_correction=True)

    # Must be blocked by Z3 verifier
    assert audit.success is False
    rejected_step = next(s for s in audit.steps if s.status == "REJECTED")
    assert rejected_step.action_type == "SET_POWER"
    assert "Z3_SMT" in rejected_step.prover_name
    assert "exceeds" in rejected_step.violation_details.lower()
