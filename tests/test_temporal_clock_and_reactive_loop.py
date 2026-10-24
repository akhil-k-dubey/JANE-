"""Unit Tests for Step 3: Real-Time Temporal Updating (60Hz Clock & Reactive DNC Replanning)."""

import pytest
import math
import time
from environments.spatial_graph.environment import SpatialEnvironmentGraph
from environments.spatial_graph.temporal_clock import ContinuousWorldClock
from environments.spatial_graph.dynamic_state_serializer import DynamicStateSerializer, WorldStateDelta
from src.core.reactive_temporal_loop import ReactiveTemporalEngine


class TestContinuousWorldClock:

    def test_clock_stepping_and_events(self):
        clock = ContinuousWorldClock(tick_rate_hz=60.0)
        assert clock.dt == pytest.approx(1.0 / 60.0, abs=1e-5)

        event_fired = False
        def on_event():
            nonlocal event_fired
            event_fired = True

        clock.schedule_event(trigger_time=0.10, callback=on_event, description="Test event")

        # Step until 0.09s: should not have fired
        for _ in range(5):  # 5 * (1/60) = 0.0833s
            clock.step()
        assert not event_fired

        # Step across 0.10s threshold
        for _ in range(3):
            clock.step()
        assert event_fired
        assert clock.sim_time >= 0.10

    def test_clock_tick_listener(self):
        clock = ContinuousWorldClock(tick_rate_hz=60.0)
        ticks_heard = 0

        def on_tick(t, dt):
            nonlocal ticks_heard
            ticks_heard += 1

        clock.register_tick_listener(on_tick)
        for _ in range(10):
            clock.step()

        assert ticks_heard == 10


class TestDynamicStateSerializer:

    def test_serializer_delta_detection_on_door_lock(self):
        env = SpatialEnvironmentGraph.create_facility_env()
        clock = ContinuousWorldClock(tick_rate_hz=60.0)
        serializer = DynamicStateSerializer(env=env, clock=clock)

        deltas_received = []
        serializer.subscribe_delta(lambda d: deltas_received.append(d))

        # Initial baseline tick: no change
        clock.step()
        assert len(deltas_received) == 0

        # Change door state dynamically
        edge = next(e for e in env.edges if e.source == "lab" and e.target == "hallway")
        edge.is_locked = True

        # Next clock tick must detect this structural delta
        delta = serializer.on_clock_tick(sim_time=clock.sim_time, dt=clock.dt)
        assert delta is not None
        assert delta.is_structural_change
        assert ("lab", "hallway", True) in delta.locked_doors_changed
        assert len(deltas_received) == 1

    def test_clingo_program_serialization(self):
        env = SpatialEnvironmentGraph.create_facility_env()
        serializer = DynamicStateSerializer(env=env)
        program = serializer.serialize_to_clingo_program(start_node="lab", goal_node="reactor_core")

        assert "node(lab)." in program
        assert "node(reactor_core)." in program
        assert "start(lab)." in program
        assert "goal(reactor_core)." in program
        assert "permitted(X, Y) :-" in program


class TestReactiveTemporalEngine:

    def test_mid_stride_door_lock_and_clingo_replan(self):
        env = SpatialEnvironmentGraph.create_temporal_facility_env()

        # Unlock airlock initially so primary path is hallway -> airlock
        airlock_edge = next(e for e in env.edges if e.source == "hallway" and e.target == "reactor_airlock")
        airlock_edge.is_locked = False
        airlock_edge.lock_condition = None
        back_airlock = next(e for e in env.edges if e.source == "reactor_airlock" and e.target == "hallway")
        back_airlock.is_locked = False
        back_airlock.lock_condition = None

        # Give agent key
        env.agent_inventory.add("reactor_key")

        engine = ReactiveTemporalEngine(tick_rate_hz=60.0, walking_speed=3.0, device="cpu")

        # Schedule door closure mid-stride at t=2.40s
        scheduled_events = [
            (2.40, "LOCK_DOOR", ("hallway", "reactor_airlock"))
        ]

        instruction = "Navigate to reactor_core and set power to 40%"

        audit = engine.execute_with_temporal_updating(
            env=env,
            instruction=instruction,
            scheduled_events=scheduled_events,
            max_sim_seconds=15.0
        )

        assert audit.success
        assert audit.halted_mid_step
        assert audit.replan_triggered
        assert audit.halt_position is not None
        # Agent halted midway along hallway (x=5.0, y ~ 2.25)
        assert audit.halt_position[0] == pytest.approx(5.0, abs=0.1)
        assert 1.0 <= audit.halt_position[1] <= 4.0

        # DNC tape norm increased due to contradiction write
        assert audit.dnc_memory_norm > 0.05

        # Successfully arrived at reactor_core via maintenance tunnel detour
        assert audit.final_environment_state["agent_node"] == "reactor_core"
        assert audit.final_environment_state["reactor_power"] == 40.0
        assert len(audit.trajectory_waypoints) > 200
