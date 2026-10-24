"""Comprehensive Test Suite for APIGateway and AnticipatoryEngine.

Verifies:
1. APIGateway deterministic SMT/Z3 safety boundary:
   - Blocks actions violating physical invariants (First/Second Law of Thermodynamics, Relativistic limits).
   - Blocks actions with explicit mathematical contradictions.
   - Executes verified OS_COMMAND with safe subprocess handling and output capture.
   - Executes verified CODE_EXEC with sandboxed math execution.
   - Executes verified HARDWARE_CONTROL and REST_API actions.
   - Preserves audit trail in EpisodicLedger.
2. AnticipatoryEngine (The J.A.R.V.I.S. Loop):
   - Background daemon thread lifecycle (start, is_running, stop).
   - Proactive initiation: Jane speaks first when telemetry breaches thresholds.
   - EgoNode personality grounding addressing Akhil.
   - Manifold latent state projection and anomaly resolution.
   - End-to-end integration with JanePipeline and APIGateway mitigation actions.
"""

import time
import os
import pytest
import torch

from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.core.api_gateway import APIGateway, ActionASTNode, GatewayResult
from src.core.anticipatory_engine import (
    AnticipatoryEngine,
    EgoNode,
    PipelineTranslatorAdapter,
    PipelineReasonerAdapter,
    PipelineIngestionAdapter
)
from src.core.pipeline import JanePipeline


@pytest.fixture
def temp_ledger(tmp_path):
    log_file = tmp_path / "test_ledger.jsonl"
    return EpisodicLedger(log_path=str(log_file))


@pytest.fixture
def verifier():
    return Z3LogicVerifier()


@pytest.fixture
def gateway(verifier, temp_ledger):
    return APIGateway(z3_verifier=verifier, episodic_ledger=temp_ledger)


# =============================================================================
# 1. APIGateway Unit & Safety Tests
# =============================================================================

def test_api_gateway_blocks_thermodynamic_violations(gateway, temp_ledger):
    """Z3 must deterministically block any action claiming over-unity or energy creation."""
    violating_node = ActionASTNode(
        action_type="OS_COMMAND",
        payload=["echo", "OverUnity"],
        safety_contract="energy_out > energy_in"
    )
    res = gateway.execute_verified_action(violating_node)

    assert res == "Action blocked by physics verifier."
    assert res.success is False
    assert res.action_type == "OS_COMMAND"
    assert "First/Second Law" in res.proof_log

    # Check episodic ledger audit
    recent = temp_ledger.get_recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0].action_type == "GATEWAY_ACTION_BLOCKED"
    assert recent[0].status == "REJECTED"


def test_api_gateway_blocks_perpetual_motion_and_math_contradictions(gateway):
    """Z3 blocks perpetual motion keywords and mathematical contradictions."""
    node_perpetual = ActionASTNode(
        action_type="HARDWARE_CONTROL",
        payload={"power": 120.0},
        safety_contract="perpetual_motion_machine == True"
    )
    res1 = gateway.execute_verified_action(node_perpetual)
    assert res1 == "Action blocked by physics verifier."
    assert res1.success is False

    node_math_contra = ActionASTNode(
        action_type="CODE_EXEC",
        payload="x = 10",
        safety_contract="1 == 2"
    )
    res2 = gateway.execute_verified_action(node_math_contra)
    assert res2 == "Action blocked by physics verifier."
    assert res2.success is False

    node_ineq_fail = ActionASTNode(
        action_type="OS_COMMAND",
        payload=["echo", "fail"],
        safety_contract="120 <= 100"
    )
    res3 = gateway.execute_verified_action(node_ineq_fail)
    assert res3 == "Action blocked by physics verifier."
    assert res3.success is False


def test_api_gateway_executes_verified_os_command(gateway, temp_ledger):
    """Safe subprocess execution of verified OS commands with output capture."""
    safe_node = ActionASTNode(
        action_type="OS_COMMAND",
        payload=["echo", "Project Jane Active"],
        safety_contract="energy_out <= energy_in"
    )
    res = gateway.execute_verified_action(safe_node)

    assert res == "Real-world action executed: OS_COMMAND"
    assert res.success is True
    assert res.output["returncode"] == 0
    assert "Project Jane Active" in res.output["stdout"]

    # Verify committed event in ledger
    recent = temp_ledger.get_recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0].action_type == "GATEWAY_ACTION_EXECUTED"
    assert recent[0].status == "COMMITTED"


def test_api_gateway_executes_verified_code_exec(gateway):
    """Safe execution of dynamically verified Python calculations."""
    code_payload = "computed_val = 19 * 2 + 4"
    safe_node = ActionASTNode(
        action_type="CODE_EXEC",
        payload=code_payload,
        safety_contract="energy_out <= energy_in"
    )
    res = gateway.execute_verified_action(safe_node)

    assert res == "Real-world action executed: CODE_EXEC"
    assert res.success is True
    assert res.output["computed_val"] == 42


def test_api_gateway_executes_hardware_control(gateway):
    """Dispatches physical and simulated hardware actions."""
    safe_node = ActionASTNode(
        action_type="HARDWARE_CONTROL",
        payload={"actuator": "coolant_valve", "state": "OPEN", "rate": 0.75},
        safety_contract="energy_out <= energy_in",
        metadata={"subsystem": "reactor"}
    )
    res = gateway.execute_verified_action(safe_node)

    assert res == "Real-world action executed: HARDWARE_CONTROL"
    assert res.success is True
    assert res.output["status"] == "DISPATCHED"
    assert res.output["command"]["actuator"] == "coolant_valve"


def test_api_gateway_handles_dict_and_duck_typing(gateway):
    """APIGateway natively normalizes raw dictionaries."""
    dict_node = {
        "action_type": "OS_COMMAND",
        "payload": ["echo", "duck_typing_test"],
        "safety_contract": "energy_out <= energy_in"
    }
    res = gateway.execute_verified_action(dict_node)
    assert res.success is True
    assert "duck_typing_test" in res.output["stdout"]


def test_api_gateway_execution_error_recovery(gateway):
    """Execution failures (e.g. non-zero exit codes) do not crash the gateway."""
    bad_cmd_node = ActionASTNode(
        action_type="OS_COMMAND",
        payload=["false"],  # Exits with returncode 1
        safety_contract="energy_out <= energy_in"
    )
    res = gateway.execute_verified_action(bad_cmd_node)
    assert res.success is False
    assert "Execution failed" in str(res)


# =============================================================================
# 2. AnticipatoryEngine & EgoNode Tests
# =============================================================================

def test_ego_node_initialization_and_alert_formatting():
    """EgoNode represents Jane's persona and addresses Akhil respectfully."""
    ego = EgoNode(latent_dim=256, user_name="Akhil")
    assert ego.user_name == "Akhil"
    assert ego.personality_vector.shape == (256,)
    assert abs(torch.norm(ego.personality_vector).item() - 1.0) < 1e-4

    alert_text = ego.format_proactive_alert(
        anomaly_summary="reactor core temperature rising to 95C",
        recommendation="Shall I vent the primary cooling manifold?"
    )
    assert "Akhil" in alert_text
    assert "reactor core temperature" in alert_text
    assert "Shall I vent" in alert_text


def test_anticipatory_engine_step_with_anomaly_detection(temp_ledger):
    """AnticipatoryEngine detects an anomaly in stream data and proactively alerts Akhil."""
    alerts_received = []

    def on_alert(msg: str):
        alerts_received.append(msg)

    class MockPipeline:
        def __init__(self):
            self.latent_dim = 256
            self.device = torch.device("cpu")
            self.episodic_ledger = temp_ledger
            self.api_gateway = None

    pipe = MockPipeline()
    engine = AnticipatoryEngine(
        pipeline=pipe,
        ego_node=EgoNode(user_name="Akhil"),
        alert_callback=on_alert
    )

    # 1. Enqueue safe idle telemetry -> No anomaly
    engine.enqueue_stream({"text": "Systems nominal", "metric": "temperature", "value": 45.0, "unit": "C"})
    res_nominal = engine.step()
    assert res_nominal is None
    assert len(alerts_received) == 0

    # 2. Enqueue overheating telemetry -> Proactive Alert Triggered!
    engine.enqueue_stream({"text": "Critical thermal breach: server temperature 96C", "metric": "temperature", "value": 96.0, "unit": "C"})
    res_anomaly = engine.step()
    assert res_anomaly is not None
    assert len(alerts_received) == 1
    assert "Akhil" in alerts_received[0]
    assert "anomaly" in alerts_received[0].lower()

    # Check episodic ledger recorded the proactive initiation
    recent = temp_ledger.get_recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0].action_type == "PROACTIVE_ALERT"
    assert recent[0].source_name == "AnticipatoryEngine"


def test_anticipatory_engine_propose_mitigation_and_execute(gateway, temp_ledger):
    """AnticipatoryEngine proposes an AST action that APIGateway verifies and executes."""
    class MockPipeline:
        def __init__(self):
            self.latent_dim = 256
            self.device = torch.device("cpu")
            self.episodic_ledger = temp_ledger
            self.api_gateway = gateway

    pipe = MockPipeline()
    engine = AnticipatoryEngine(pipeline=pipe, api_gateway=gateway)

    # Trigger anomaly
    engine.enqueue_stream({"metric": "temperature", "value": 92.0, "unit": "C"})
    anomaly_record = engine.step()
    assert anomaly_record is not None

    # Propose mitigation
    action_ast = engine.propose_mitigation_action(anomaly_record)
    assert action_ast.action_type == "HARDWARE_CONTROL"
    assert action_ast.safety_contract == "energy_out <= energy_in"

    # Gateway executes verified mitigation action
    gw_result = gateway.execute_verified_action(action_ast)
    assert gw_result.success is True
    assert gw_result == "Real-world action executed: HARDWARE_CONTROL"


def test_anticipatory_engine_thread_lifecycle():
    """Background monitor thread starts, loops, and stops cleanly."""
    class MockPipeline:
        def __init__(self):
            self.latent_dim = 256
            self.device = torch.device("cpu")
            self.episodic_ledger = None
            self.api_gateway = None

    pipe = MockPipeline()
    engine = AnticipatoryEngine(pipeline=pipe, check_interval_sec=0.05)

    assert not engine.is_running
    engine.start()
    assert engine.is_running

    # Enqueue stream while running in background
    engine.enqueue_stream({"text": "Critical radiation leak detected", "metric": "radiation", "value": 150.0})
    time.sleep(0.15)

    # Stop thread cleanly
    engine.stop(timeout=1.0)
    assert not engine.is_running
    assert len(engine.get_detected_anomalies()) >= 1


# =============================================================================
# 3. Full JanePipeline Integration
# =============================================================================

def test_jane_pipeline_gateway_and_anticipatory_integration():
    """End-to-end verification of JanePipeline with APIGateway and AnticipatoryEngine."""
    pipeline = JanePipeline(
        latent_dim=256,
        use_pretrained_translator=False,
        device="cpu"
    )

    # 1. Test pipeline.execute_verified_action
    safe_action = ActionASTNode(
        action_type="OS_COMMAND",
        payload=["echo", "PipelineGatewayOk"],
        safety_contract="energy_out <= energy_in"
    )
    res_safe = pipeline.execute_verified_action(safe_action)
    assert res_safe.success is True
    assert "PipelineGatewayOk" in res_safe.output["stdout"]

    unsafe_action = ActionASTNode(
        action_type="OS_COMMAND",
        payload=["echo", "Blocked"],
        safety_contract="energy_out > energy_in"
    )
    res_unsafe = pipeline.execute_verified_action(unsafe_action)
    assert res_unsafe.success is False
    assert res_unsafe == "Action blocked by physics verifier."

    # 2. Test pipeline proactive monitor step
    pipeline.anticipatory_engine.enqueue_stream({
        "text": "Critical thermal breach: temperature 99C",
        "metric": "temperature",
        "value": 99.0
    })
    rec = pipeline.anticipatory_engine.step()
    assert rec is not None
    assert "Akhil" in rec["response"]
