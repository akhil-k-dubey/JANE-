"""Unit Tests for Phase 5: Theory of Mind (ToM) Cognitive Architecture."""

import pytest
import os
import threading
import time
import torch

from src.reasoning_engine.tom_manifold import TheoryOfMindManifold
from src.core.metacognitive_observer import MetacognitiveObserverThread


@pytest.fixture
def tmp_tom_path(tmp_path):
    return str(tmp_path / "test_tom_manifold.pt")


class TestTheoryOfMindManifold:

    def test_ego_and_alter_node_initialization(self, tmp_tom_path):
        tom = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)
        assert tom.current_ego_label == "Self_Idle"
        assert tom.current_alter_label == "User_Calm"
        assert torch.norm(tom.z_ego) > 0.9
        assert torch.norm(tom.z_alter) > 0.9

    def test_ego_proprioception_updates(self, tmp_tom_path):
        tom = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)

        info_nav = tom.update_ego_state("NAVIGATE", target="reactor_airlock", status="EXECUTING")
        assert info_nav["ego_label"] == "Self_Navigating"
        assert tom.current_ego_label == "Self_Navigating"

        info_vent = tom.update_ego_state("VENT_GAS", target="reactor_airlock", status="EXECUTING")
        assert info_vent["ego_label"] == "Self_Venting"

        info_halt = tom.update_ego_state("EMERGENCY_HALT", status="HALTED")
        assert info_halt["ego_label"] == "Self_Halted"

    def test_alter_user_state_tracking(self, tmp_tom_path):
        tom = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)

        # 1. Calm reading
        res_calm = tom.update_alter_state(audio_rms=0.026, speech_text_or_intent="reading quiet")
        assert res_calm["alter_label"] in ("User_Calm", "User_Learning")

        # 2. Loud teaching
        res_teach = tom.update_alter_state(audio_rms=0.065, speech_text_or_intent="lecturing about physics")
        assert res_teach["alter_label"] == "User_Teaching"

        # 3. Emergency Alarm
        res_alarm = tom.update_alter_state(audio_rms=0.090, speech_text_or_intent="STOP! HALT!")
        assert res_alarm["alter_label"] in ("User_Alarm", "User_Stress")

        # 4. Affirmation
        res_affirm = tom.update_alter_state(audio_rms=0.035, speech_text_or_intent="good job, perfect")
        assert res_affirm["alter_label"] == "User_Affirming"

    def test_cognitive_dissonance_computation(self, tmp_tom_path):
        tom = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)

        # Baseline: Idle + Calm -> Very low dissonance
        tom.update_ego_state("IDLE")
        tom.update_alter_state(audio_rms=0.015, speech_text_or_intent="")
        assert tom.compute_cognitive_dissonance() < 0.50

        # Conflict: Jane is Navigating while Akhil is in Alarm -> Extreme dissonance
        tom.update_ego_state("NAVIGATE", status="EXECUTING")
        tom.update_alter_state(audio_rms=0.095, speech_text_or_intent="STOP!")
        dissonance = tom.compute_cognitive_dissonance()
        assert dissonance >= 0.85

    def test_causal_agency_triad_and_forecasting(self, tmp_tom_path):
        tom = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)

        # Establish causal triad: Jane_Vents_Gas -> Room_Pressure_Drops -> User_Affirming
        triad = tom.register_causal_triad(
            ego_action="VENT_GAS",
            env_delta="Room_Pressure_Drops",
            alter_reaction="good job"
        )

        assert triad["ego_action"] == "Self_Venting"
        assert triad["env_delta"] == "Env_Pressure_Drops"
        assert triad["alter_reaction"] == "User_Affirming"
        assert triad["superhighway_weight"] > 0.05

        # Predict Akhil's reaction to venting gas
        pred_alter, confidence = tom.predict_alter_reaction("VENT_GAS")
        assert pred_alter == "User_Affirming"
        assert confidence > 0.10

    def test_save_and_load_tom_manifold(self, tmp_tom_path):
        tom1 = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)
        tom1.update_ego_state("VENT_GAS")
        tom1.register_causal_triad("VENT_GAS", "Room_Pressure_Drops", "good job")
        tom1.save_tom_manifold()

        assert os.path.exists(tmp_tom_path)

        tom2 = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)
        assert len(tom2.causal_history) >= 1
        assert torch.allclose(tom1.adjacency, tom2.adjacency)


class TestMetacognitiveObserverThread:

    def test_watchdog_interrupt_on_high_dissonance(self, tmp_tom_path):
        tom = TheoryOfMindManifold(latent_dim=256, save_path=tmp_tom_path)
        interrupt_event = threading.Event()
        stop_event = threading.Event()

        watchdog = MetacognitiveObserverThread(
            tom_manifold=tom,
            interrupt_event=interrupt_event,
            stop_event=stop_event,
            dissonance_threshold=0.70,
            poll_interval_sec=0.01,
            verbose=False
        )

        watchdog.start()
        try:
            # Baseline: Navigating + Calm -> Dissonance is low
            tom.update_ego_state("NAVIGATE")
            tom.update_alter_state(audio_rms=0.020, speech_text_or_intent="")
            time.sleep(0.03)
            assert not interrupt_event.is_set()

            # Trigger Conflict: Akhil screams STOP!
            tom.update_alter_state(audio_rms=0.095, speech_text_or_intent="STOP!")
            time.sleep(0.03)

            # Watchdog MUST set interrupt
            assert interrupt_event.is_set()
            assert watchdog.interrupt_triggered
            assert len(watchdog.audit_log) > 0
            assert watchdog.audit_log[-1]["interrupted"] is True
        finally:
            stop_event.set()
            watchdog.join(timeout=1.0)
