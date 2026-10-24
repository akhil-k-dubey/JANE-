"""Reactive Temporal Loop: 60Hz Continuous World Clock & Reactive DNC Replanning.

Implements Real-Time Temporal Updating for Embodied Jane:
1. Continuous World Clock (60Hz / dt = 1/60s) driving dynamic environmental physics independently of deliberation.
2. Dynamic State Serializer: Continuously converts live graph states into symbolic Clingo facts and detects state deltas.
3. Mid-Stride Kinetic Motion: Moves Jane continuously along physical waypoints (x, y) rather than instant teleportation.
4. DNC Memory Tape Delta Invalidation: When a door locks mid-step, DNC registers the state contradiction,
   instantaneously freezes kinetic motion, and reverts to System 2 reasoning.
5. Reactive System 2 Replanning: Dynamic Clingo ASP compiler solves an alternative route from current coordinates,
   updates the DNC memory tape, and safely completes execution.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
import torch
import torch.nn as nn

from environments.spatial_graph.environment import SpatialEnvironmentGraph, GraphNode, GraphEdge
from environments.spatial_graph.temporal_clock import ContinuousWorldClock, TimedEvent
from environments.spatial_graph.dynamic_state_serializer import DynamicStateSerializer, WorldStateDelta
from environments.spatial_graph.continuous_kinematics import ContinuousKinematicsEngine
from environments.spatial_graph.dynamic_asp_compiler import DynamicClingoCompiler
from src.reasoning_engine.system2_planner import PlanStep, System2Planner, ExecutionAuditTrail
from src.reasoning_engine.dnc_memory import DNCMemoryTape


@dataclass
class ReactiveExecutionAudit:
    """Detailed explainable audit trail for real-time temporal reactive execution."""
    task_description: str
    success: bool
    halted_mid_step: bool
    replan_triggered: bool
    halt_position: Optional[Tuple[float, float]]
    halt_reason: Optional[str]
    initial_plan: List[PlanStep]
    reactive_plan: List[PlanStep]
    dnc_memory_norm: float
    total_sim_time: float
    inner_monologue_logs: List[str]
    trajectory_waypoints: List[Tuple[float, float]]
    final_environment_state: Dict[str, Any]


class ReactiveTemporalEngine:
    """
    Reactive Temporal Embodiment Engine.
    Coordinates continuous 60Hz world updates, mid-step motion, and dynamic Clingo replanning.
    """

    def __init__(
        self,
        planner: Optional[System2Planner] = None,
        latent_dim: int = 256,
        tick_rate_hz: float = 60.0,
        walking_speed: float = 2.5,  # 2.5 units per second
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.tick_rate_hz = tick_rate_hz
        self.walking_speed = walking_speed
        self.dt = 1.0 / tick_rate_hz

        # Cognitive & Proving components
        self.planner = planner or System2Planner(latent_dim=latent_dim, device=device)
        self.clingo_compiler = DynamicClingoCompiler()
        self.kinematics_engine = ContinuousKinematicsEngine()
        self.dnc_tape = self.planner.dnc_tape

    def execute_with_temporal_updating(
        self,
        env: SpatialEnvironmentGraph,
        instruction: str,
        scheduled_events: Optional[List[Tuple[float, str, Any]]] = None,
        max_sim_seconds: float = 10.0
    ) -> ReactiveExecutionAudit:
        """
        Executes an embodied instruction on the continuous 60Hz temporal clock.
        Monitors for real-time environmental events (e.g. door closing mid-step),
        registers state deltas on DNC memory tape, halts, and replans reactively.

        Args:
            env: Live SpatialEnvironmentGraph
            instruction: Human goal (e.g. "Navigate to reactor_core and set power to 40%")
            scheduled_events: List of (trigger_time, event_type, details) e.g. (0.5, "LOCK_DOOR", ("hallway", "reactor_airlock"))
            max_sim_seconds: Safety limit for simulation time

        Returns:
            ReactiveExecutionAudit with full timestamps, waypoints, and DNC memory state.
        """
        inner_monologue: List[str] = []
        inner_monologue.append(f"[Step 3 Temporal Engine] Initializing 60Hz clock (dt={self.dt:.4f}s).")

        # 1. Initialize Continuous Clock and Dynamic State Serializer
        clock = ContinuousWorldClock(tick_rate_hz=self.tick_rate_hz)
        serializer = DynamicStateSerializer(env=env, clock=clock)

        # Register external scheduled events on the clock
        if scheduled_events:
            for trig_time, ev_type, details in scheduled_events:
                if ev_type == "LOCK_DOOR":
                    src, tgt = details
                    def make_lock_cb(s=src, t=tgt):
                        def cb():
                            edge = next((e for e in env.edges if e.source == s and e.target == t), None)
                            if edge:
                                edge.is_locked = True
                                inner_monologue.append(f"  [Clock Event @ t={clock.sim_time:.2f}s] Security Lockdown! Door '{s}' -> '{t}' slammed shut and locked.")
                        return cb
                    clock.schedule_event(trig_time, make_lock_cb(), description=f"Lock door {src}->{tgt}")

                elif ev_type == "VENT_PRESSURE":
                    r_id, new_p = details
                    def make_press_cb(r=r_id, p=new_p):
                        def cb():
                            env.nodes[r].properties["pressure"] = p
                            inner_monologue.append(f"  [Clock Event @ t={clock.sim_time:.2f}s] Pressure changed in '{r}' to {p}%.")
                        return cb
                    clock.schedule_event(trig_time, make_press_cb(), description=f"Pressure in {r_id} to {new_p}")

        # Track state deltas via serializer subscription
        latest_delta: Optional[WorldStateDelta] = None
        def delta_listener(delta: WorldStateDelta):
            nonlocal latest_delta
            latest_delta = delta
        serializer.subscribe_delta(delta_listener)

        # 2. Initial System 2 Plan Deliberation (Frozen in DNC memory)
        initial_plan = self.planner.parse_instruction_to_plan(instruction)
        inner_monologue.append(f"[System 2 Deliberation] Initial plan created with {len(initial_plan)} steps.")
        for s in initial_plan:
            inner_monologue.append(f"  Plan Step {s.step_index}: {s.action_type} -> {s.target}")

        active_plan = list(initial_plan)
        executed_steps: List[PlanStep] = []
        all_waypoints: List[Tuple[float, float]] = [env.agent_pos]

        # Initialize DNC Memory Tape
        memory = self.dnc_tape.init_memory(1, device=self.device)

        halted_mid_step = False
        replan_triggered = False
        halt_pos: Optional[Tuple[float, float]] = None
        halt_reason: Optional[str] = None

        current_step_idx = 0
        while current_step_idx < len(active_plan) and clock.sim_time < max_sim_seconds:
            step = active_plan[current_step_idx]

            if step.action_type == "NAVIGATE":
                target_room = step.target
                start_room = env.agent_node

                if start_room == target_room:
                    step.status = "EXECUTED"
                    executed_steps.append(step)
                    current_step_idx += 1
                    continue

                inner_monologue.append(f"\n[Executing Step {step.step_index}] NAVIGATE '{start_room}' -> '{target_room}' (Continuous Motion)")

                # If target room is not directly adjacent (multi-hop path), expand into atomic room hops
                is_direct_edge = any(e.source == start_room and e.target == target_room for e in env.edges)
                if not is_direct_edge:
                    node_path = self.clingo_compiler.find_path(env, start_room, target_room)
                    if node_path and len(node_path) > 2:
                        inner_monologue.append(f"  [Topological Expansion] Decomposed multi-hop path: {' -> '.join(node_path)}")
                        sub_steps = [PlanStep(step_index=0, action_type="NAVIGATE", target=r) for r in node_path[1:]]
                        active_plan = executed_steps + sub_steps + active_plan[current_step_idx + 1:]
                        for idx, s in enumerate(active_plan):
                            s.step_index = idx + 1
                        continue

                # Verify topological reachability before setting foot
                clingo_proof = self.clingo_compiler.verify_reachability(env, start_room, target_room)
                if not clingo_proof.is_valid:
                    inner_monologue.append(f"  [Topological Check Failed] Edge '{start_room}' -> '{target_room}' is blocked: {clingo_proof.violation_details}")
                    step.status = "BLOCKED"
                    # Trigger self-correction / alternate path
                    new_path = self.clingo_compiler.find_path(env, start_room, target_room)
                    if new_path and len(new_path) > 1:
                        inner_monologue.append(f"  [Reactive Rerouting] Found alternative topological path: {' -> '.join(new_path)}")
                        # Replace step with multi-room path
                        sub_steps = [PlanStep(step_index=0, action_type="NAVIGATE", target=r) for r in new_path[1:]]
                        active_plan = executed_steps + sub_steps + active_plan[current_step_idx + 1:]
                        for idx, s in enumerate(active_plan):
                            s.step_index = idx + 1
                        replan_triggered = True
                        continue
                    else:
                        break

                # Generate continuous geometric waypoints
                start_coord = env.agent_pos
                target_coord = env.nodes[target_room].coordinates
                segment_dist = math.hypot(target_coord[0] - start_coord[0], target_coord[1] - start_coord[1])
                total_duration = max(0.1, segment_dist / self.walking_speed)
                steps_for_segment = max(1, int(total_duration * self.tick_rate_hz))

                # Step along continuous trajectory micro-steps on 60Hz clock
                step_interrupted = False
                for micro_step in range(1, steps_for_segment + 1):
                    # Advance 60Hz world clock
                    clock.step(self.dt)

                    # Interpolate continuous coordinates
                    alpha = micro_step / float(steps_for_segment)
                    current_x = start_coord[0] + alpha * (target_coord[0] - start_coord[0])
                    current_y = start_coord[1] + alpha * (target_coord[1] - start_coord[1])
                    env.agent_pos = (round(current_x, 3), round(current_y, 3))
                    all_waypoints.append(env.agent_pos)

                    # Check for mid-step state delta from 60Hz serializer
                    if latest_delta and latest_delta.is_structural_change:
                        # Check if our current edge or destination door locked
                        for src, tgt, is_locked in latest_delta.locked_doors_changed:
                            if is_locked and ((src == start_room and tgt == target_room) or (tgt == target_room)):
                                step_interrupted = True
                                halt_reason = f"Door '{src}' -> '{tgt}' locked mid-step at t={clock.sim_time:.2f}s!"
                                halt_pos = env.agent_pos
                                inner_monologue.append(f"  \033[91m[MID-STRIDE HALT @ t={clock.sim_time:.2f}s]\033[0m {halt_reason}")
                                inner_monologue.append(f"  Physical Motion Frozen instantly at continuous coordinate {halt_pos}.")
                                latest_delta = None
                                break

                    if step_interrupted:
                        break

                if step_interrupted:
                    halted_mid_step = True
                    step.status = "HALTED_MID_STRIDE"
                    step.violation_details = halt_reason

                    # Write surprise / invalidation delta into DNC Memory Tape
                    surprise_vec = torch.randn(1, self.latent_dim, device=self.device) * 0.5
                    memory, _ = self.dnc_tape.write(memory, surprise_vec)
                    inner_monologue.append(f"  [DNC Memory Scratchpad] Registered State Delta contradiction (Tape Memory Norm: {torch.norm(memory):.4f}).")

                    # Revert to System 2 Reasoning Loop from current topological zone
                    inner_monologue.append("  [System 2 Reversion] Initiating Dynamic Clingo Replanning from current state...")
                    final_goal = active_plan[-1].target if active_plan else "reactor_core"

                    # Solve new path from start_room to final_goal on updated environment
                    replan_path = self.clingo_compiler.find_path(env, start_room, final_goal)
                    if replan_path and len(replan_path) > 1:
                        inner_monologue.append(f"  \033[92m[Clingo ASP Proved New Path]\033[0m Reactive detour discovered: {' -> '.join(replan_path)}")
                        replan_steps = [PlanStep(step_index=0, action_type="NAVIGATE", target=r) for r in replan_path[1:]]
                        # Preserve remaining non-navigate steps (e.g. set power)
                        remaining_non_nav = [s for s in active_plan[current_step_idx + 1:] if s.action_type != "NAVIGATE"]
                        active_plan = executed_steps + [step] + replan_steps + remaining_non_nav
                        for idx, s in enumerate(active_plan):
                            s.step_index = idx + 1
                        current_step_idx = len(executed_steps) + 1
                        replan_triggered = True
                        continue
                    else:
                        inner_monologue.append("  [Replanning Exhausted] No alternative topological route permitted under current constraints.")
                        break
                else:
                    # Successfully arrived at target room
                    env.agent_node = target_room
                    step.status = "EXECUTED"
                    executed_steps.append(step)
                    inner_monologue.append(f"  [Arrived] Agent safely entered room '{target_room}' at {env.agent_pos}.")
                    current_step_idx += 1

            elif step.action_type == "PICKUP":
                clock.step(0.1)
                env.pickup_item(step.target)
                step.status = "EXECUTED"
                executed_steps.append(step)
                inner_monologue.append(f"[Executed Step {step.step_index}] PICKUP item '{step.target}'. Inventory: {list(env.agent_inventory)}")
                current_step_idx += 1

            elif step.action_type == "VENT_GAS":
                clock.step(0.15)
                env.vent_gas(step.target)
                step.status = "EXECUTED"
                executed_steps.append(step)
                inner_monologue.append(f"[Executed Step {step.step_index}] VENT_GAS in '{step.target}'. Pressure stabilized.")
                current_step_idx += 1

            elif step.action_type == "SET_POWER":
                clock.step(0.1)
                p_pct = float(step.parameters.get("power_percent", 40.0))
                env.set_reactor_power(step.target, p_pct)
                step.status = "EXECUTED"
                executed_steps.append(step)
                inner_monologue.append(f"[Executed Step {step.step_index}] SET_POWER '{step.target}' to {p_pct}%.")
                current_step_idx += 1

            else:
                clock.step(self.dt)
                step.status = "EXECUTED"
                executed_steps.append(step)
                current_step_idx += 1

        all_success = all(s.status == "EXECUTED" for s in executed_steps if s.status != "HALTED_MID_STRIDE") and (not active_plan or executed_steps[-1].status == "EXECUTED")

        return ReactiveExecutionAudit(
            task_description=instruction,
            success=all_success,
            halted_mid_step=halted_mid_step,
            replan_triggered=replan_triggered,
            halt_position=halt_pos,
            halt_reason=halt_reason,
            initial_plan=initial_plan,
            reactive_plan=active_plan,
            dnc_memory_norm=torch.norm(memory).item(),
            total_sim_time=round(clock.sim_time, 3),
            inner_monologue_logs=inner_monologue,
            trajectory_waypoints=all_waypoints,
            final_environment_state={
                "agent_node": env.agent_node,
                "agent_pos": env.agent_pos,
                "inventory": list(env.agent_inventory),
                "reactor_power": env.global_state.get("reactor_power", 0.0)
            }
        )
