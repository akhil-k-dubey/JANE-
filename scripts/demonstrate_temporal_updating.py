#!/usr/bin/env python3
"""Project Jane — Step 3 Demonstration: Real-Time Temporal Updating.

Demonstrates:
1. 60Hz Continuous World Clock (world states ticking while Jane is in motion).
2. Continuous Kinematic Motion (Jane traverses physical coordinates, not instant teleportation).
3. Mid-Stride Invariant Violation: A security blast door locks while Jane is walking toward it.
4. DNC Memory Tape Delta Invalidation: DNC registers the contradiction and arrests kinetic motion.
5. Reactive System 2 Replanning: Dynamic Clingo ASP solves an alternate route (via maintenance tunnel),
   splices the new verified path into DNC memory, and completes the goal without collision.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.spatial_graph.environment import SpatialEnvironmentGraph
from src.core.reactive_temporal_loop import ReactiveTemporalEngine


def main():
    print("=" * 76)
    print("   PROJECT JANE — STEP 3: REAL-TIME TEMPORAL UPDATING (60Hz CLOCK)")
    print("=" * 76)
    print("  • Engine:             Continuous 60Hz World Clock & Dynamic State Serializer")
    print("  • Physical Motion:    Continuous 2D waypoints & velocity interpolation")
    print("  • Invariant Monitor:  Clingo state stream checks permissions at every tick")
    print("  • Reactive Behavior:  Mid-stride halt on door closure -> DNC delta -> Clingo replan")
    print("=" * 76)

    # 1. Create Facility Environment with bypass tunnel
    env = SpatialEnvironmentGraph.create_temporal_facility_env()
    # Ensure airlock door is initially open so Jane plans to use it
    airlock_edge = next((e for e in env.edges if e.source == "hallway" and e.target == "reactor_airlock"), None)
    if airlock_edge:
        airlock_edge.is_locked = False
        airlock_edge.lock_condition = None

    back_airlock = next((e for e in env.edges if e.source == "reactor_airlock" and e.target == "hallway"), None)
    if back_airlock:
        back_airlock.is_locked = False
        back_airlock.lock_condition = None

    # Agent holds reactor key so primary airlock route is fully permitted
    env.agent_inventory.add("reactor_key")

    print(f"\n[Environment Setup]")
    print(f"  Agent Starting Room:       '{env.agent_node}' at coordinate {env.agent_pos}")
    print(f"  Target Destination:        'reactor_core'")
    print(f"  Scheduled Clock Event:     Door 'hallway' -> 'reactor_airlock' will SLAM SHUT at t=2.40s mid-stride!\n")


    # 2. Configure Reactive Temporal Engine
    engine = ReactiveTemporalEngine(
        tick_rate_hz=60.0,
        walking_speed=3.0,
        device="cpu"
    )

    # Schedule mid-stride door closure event at t = 2.40s
    # Jane reaches hallway at t=1.67s, and is mid-step to airlock at t=2.40s (around (5.0, 2.2))
    scheduled_events = [
        (2.40, "LOCK_DOOR", ("hallway", "reactor_airlock"))
    ]

    instruction = "Navigate to hallway, then go to reactor_core and set power to 40%"

    print("-" * 76)
    print(">>> INITIATING REAL-TIME TEMPORAL EXECUTION...")
    print("-" * 76)

    start_real = time.time()
    audit = engine.execute_with_temporal_updating(
        env=env,
        instruction=instruction,
        scheduled_events=scheduled_events,
        max_sim_seconds=15.0
    )

    elapsed_real = time.time() - start_real

    # Print Monologue & Audit
    print("\n--- CONTINUOUS TEMPORAL AUDIT TRAIL ---")
    for log in audit.inner_monologue_logs:
        print(log)

    print("\n" + "=" * 76)
    print("                    STEP 3 VERIFICATION RESULTS")
    print("=" * 76)
    print(f"  Execution Status:          {'SUCCESS (Goal Reached Reactively)' if audit.success else 'FAILED'}")
    print(f"  Halted Mid-Step:           {audit.halted_mid_step} (Instant hardware freeze triggered)")
    if audit.halt_position:
        print(f"  Kinetic Freeze Coordinate: {audit.halt_position}")
    print(f"  Reactive Replan Triggered: {audit.replan_triggered} (Dynamic Clingo ASP path re-solved)")
    print(f"  DNC Memory Tape Norm:      {audit.dnc_memory_norm:.4f} (Contradiction registered on tape)")
    print(f"  Continuous Sim Time:       {audit.total_sim_time} seconds (Ticked at 60Hz)")
    print(f"  Wallclock Real Time:       {elapsed_real:.3f} seconds")
    print(f"  Final Agent Room:          '{audit.final_environment_state['agent_node']}' at {audit.final_environment_state['agent_pos']}")
    print(f"  Reactor Power Setting:     {audit.final_environment_state['reactor_power']}%")
    print(f"  Total Trajectory Steps:    {len(audit.trajectory_waypoints)} continuous coordinates")
    print("=" * 76)


if __name__ == "__main__":
    main()
