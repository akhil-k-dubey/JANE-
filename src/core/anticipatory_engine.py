"""Proactive Anticipatory Intelligence Engine (The J.A.R.V.I.S. Loop) for Project Jane.

Transforms Jane from a purely reactive assistant into a proactive intelligence:
1. Continuous Ingestion & Telemetry Monitoring:
   Monitors streaming background data (system health, thermal metrics, sensor feeds, news, preprints).
2. Manifold Projection:
   Projects incoming streams into the 256D continuous concept space.
3. Latent Anomaly Detection:
   Identifies critical threshold violations, high energy states, and epistemic discrepancies.
4. Proactive Initiation (Speaking First):
   Without waiting to be addressed, Jane initiates conversation using her Ego Node (personality),
   alerting Akhil to anomalies and proposing concrete SMT-verified mitigation actions.
"""

import time
import threading
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Callable, Union

from src.utils.logger import get_logger
from src.core.api_gateway import ActionASTNode, APIGateway

logger = get_logger("anticipatory_engine")


class EgoNode:
    """
    Personality-grounded Ego Node for Jane's autonomous proactivity.
    Embodies Jane's self-awareness, identity, and relational stance toward Akhil.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        user_name: str = "Akhil",
        personality_vector: Optional[torch.Tensor] = None,
        traits: Optional[Dict[str, float]] = None,
        tom_manifold: Optional[Any] = None
    ):
        self.latent_dim = latent_dim
        self.user_name = user_name
        self.tom_manifold = tom_manifold
        self.traits = traits or {
            "proactivity": 0.95,
            "scientific_rigor": 0.98,
            "protectiveness": 0.92,
            "loyalty": 1.0,
            "calmness": 0.88
        }
        if personality_vector is not None:
            self.personality_vector = personality_vector
        else:
            torch.manual_seed(1337)
            vec = torch.randn(latent_dim)
            self.personality_vector = vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
        self.state_label: str = "Self_Observing"

    def format_proactive_alert(self, anomaly_summary: str, recommendation: Optional[str] = None) -> str:
        """Phrases a proactive alert addressed directly to Akhil."""
        rec = f" {recommendation}" if recommendation else " Shall I take corrective measures?"
        return f"{self.user_name}, I am detecting an anomaly: {anomaly_summary}.{rec}"

    def get_personality_summary(self) -> str:
        """Returns brief trait description."""
        traits_str = ", ".join(f"{k}={v:.2f}" for k, v in self.traits.items())
        return f"EgoNode(User='{self.user_name}', State='{self.state_label}', Traits=[{traits_str}])"


class PipelineTranslatorAdapter:
    """Adapts JanePipeline translator (Ear / Mouth / Encoder) for AnticipatoryEngine."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def encode(self, data_stream: Any) -> torch.Tensor:
        """Projects incoming stream text or dict into a 256D latent state."""
        if isinstance(data_stream, dict):
            text_rep = data_stream.get("text") or data_stream.get("summary") or str(data_stream)
        else:
            text_rep = str(data_stream)

        if hasattr(self.pipeline, "ear") and self.pipeline.ear is not None:
            try:
                return self.pipeline.ear(text_rep)
            except Exception:
                pass

        if hasattr(self.pipeline, "encoder") and self.pipeline.encoder is not None:
            try:
                tokens = self.pipeline.tokenizer.encode(text_rep)
                token_ids = [tok.token_id for tok in tokens]
                t = torch.tensor([token_ids], dtype=torch.long, device=self.pipeline.device)
                seq = self.pipeline.encoder(t)
                return self.pipeline.encoder.pool_state(seq)
            except Exception:
                pass

        # Fallback orthonormal projection
        torch.manual_seed(abs(hash(text_rep)) % (2**31))
        vec = torch.randn(self.pipeline.latent_dim, device=self.pipeline.device)
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

    def generate(self, alert_vector: torch.Tensor, personality_vector: Optional[torch.Tensor] = None) -> str:
        """Generates proactive verbalization addressing Akhil."""
        user_name = "Akhil"
        if hasattr(self.pipeline, "self_model") and self.pipeline.self_model:
            user_name = self.pipeline.self_model.user_profile.get("name", "Akhil")

        return f"{user_name}, I am detecting an anomaly in the monitored systems. Shall I vent the cooling systems?"


class PipelineReasonerAdapter:
    """Adapts JanePipeline reasoning and energy landscape for AnticipatoryEngine."""

    def __init__(self, pipeline: Any, energy_threshold: float = 0.85):
        self.pipeline = pipeline
        self.energy_threshold = energy_threshold
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        self.intent_manifold = TrueIntentManifold.get_instance()

    def detect_anomaly(self, latent_state: torch.Tensor, stream_data: Optional[Any] = None) -> bool:
        """Evaluates whether the latent state or stream data indicates an anomaly."""
        # 1. Explicit telemetry anomaly check
        if isinstance(stream_data, dict):
            if stream_data.get("is_anomaly") or stream_data.get("alert"):
                return True

            # Schema with {"metric": "...", "value": ...}
            metric = str(stream_data.get("metric", "")).lower()
            val = stream_data.get("value")
            if isinstance(val, (int, float)):
                fval = float(val)
                if ("temp" in metric) and fval > 85.0:
                    return True
                if ("press" in metric) and fval > 10.0:
                    return True
                if ("cpu" in metric or "mem" in metric) and fval > 90.0:
                    return True
                if ("power" in metric) and fval > 100.0:
                    return True
                if fval > 100.0:
                    return True

            # Direct key schema
            temp = stream_data.get("temperature") or stream_data.get("temp")
            if isinstance(temp, (int, float)) and float(temp) > 85.0:
                return True
            mem = stream_data.get("memory_percent") or stream_data.get("memory")
            if isinstance(mem, (int, float)) and float(mem) > 90.0:
                return True
            pressure = stream_data.get("pressure")
            if isinstance(pressure, (int, float)) and float(pressure) > 10.0:
                return True
            power = stream_data.get("power")
            if isinstance(power, (int, float)) and float(power) > 100.0:
                return True

        # 2. Continuous vector anomaly check
        text_str = str(stream_data).lower()
        if self.intent_manifold.similarity(text_str, "emergency") >= 0.45:
            return True

        # 3. Energy landscape check in 256D manifold
        if hasattr(self.pipeline, "energy_landscape") and self.pipeline.energy_landscape is not None:
            try:
                energy = self.pipeline.energy_landscape.compute_energy(latent_state)
                if float(energy.mean().item()) > self.energy_threshold:
                    return True
            except Exception:
                pass

        return False

    def resolve_anomaly(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Relaxes the anomalous latent state towards an equilibrium recovery vector."""
        if hasattr(self.pipeline, "energy_landscape") and self.pipeline.energy_landscape is not None:
            try:
                settled, _ = self.pipeline.energy_landscape.settle_equilibrium(latent_state, steps=5)
                return settled
            except Exception:
                pass
        return F.normalize(latent_state * 0.5, p=2, dim=-1)


class PipelineIngestionAdapter:
    """Adapts ingestion queues and telemetry feeds for AnticipatoryEngine."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.queue: List[Any] = []

    def enqueue(self, item: Any):
        self.queue.append(item)

    def get_latest_stream(self) -> Optional[Any]:
        if self.queue:
            return self.queue.pop(0)
        return None


class AnticipatoryEngine:
    """
    Proactive Anticipatory Intelligence Loop for Project Jane.
    Runs constantly in the background, monitoring data streams, projecting them
    into the 256D manifold, detecting anomalies, and initiating conversation with Akhil.
    """

    def __init__(
        self,
        pipeline: Any,
        ego_node: Optional[Any] = None,
        check_interval_sec: float = 10.0,
        alert_callback: Optional[Callable[[str], None]] = None,
        api_gateway: Optional[Any] = None
    ):
        self.pipeline = pipeline
        self.check_interval_sec = check_interval_sec
        self.alert_callback = alert_callback
        self.api_gateway = api_gateway or getattr(pipeline, "api_gateway", None)

        # 0. Ego Node Personality
        if ego_node is not None:
            self.ego_node = ego_node
        elif hasattr(pipeline, "ego_node") and pipeline.ego_node is not None:
            self.ego_node = pipeline.ego_node
        else:
            user_name = "Akhil"
            if hasattr(pipeline, "self_model") and pipeline.self_model:
                user_name = pipeline.self_model.user_profile.get("name", "Akhil")
            self.ego_node = EgoNode(
                latent_dim=getattr(pipeline, "latent_dim", 256),
                user_name=user_name
            )

        # 1. Adapters for pipeline components if not natively provided
        if not hasattr(self.pipeline, "translator"):
            self.pipeline.translator = PipelineTranslatorAdapter(self.pipeline)
        if not hasattr(self.pipeline, "reasoner"):
            self.pipeline.reasoner = PipelineReasonerAdapter(self.pipeline)
        if not hasattr(self.pipeline, "ingestion_daemon"):
            self.pipeline.ingestion_daemon = PipelineIngestionAdapter(self.pipeline)

        # Thread management
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._pending_streams: List[Any] = []
        self._detected_anomalies: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def enqueue_stream(self, data: Any):
        """Allows external systems, background tasks, or tests to inject real-time data."""
        with self._lock:
            self._pending_streams.append(data)

    def get_latest_stream(self) -> Optional[Any]:
        """Retrieves the latest available stream event."""
        with self._lock:
            if self._pending_streams:
                return self._pending_streams.pop(0)

        if hasattr(self.pipeline, "ingestion_daemon") and self.pipeline.ingestion_daemon is not None:
            if hasattr(self.pipeline.ingestion_daemon, "get_latest_stream"):
                return self.pipeline.ingestion_daemon.get_latest_stream()

        return None

    def _encode_stream(self, data_stream: Any) -> torch.Tensor:
        """Encodes stream data via pipeline translator into 256D continuous space."""
        if hasattr(self.pipeline, "translator") and hasattr(self.pipeline.translator, "encode"):
            return self.pipeline.translator.encode(data_stream)
        elif hasattr(self.pipeline, "ear"):
            return self.pipeline.ear(str(data_stream))
        torch.manual_seed(42)
        return torch.randn(getattr(self.pipeline, "latent_dim", 256))

    def _detect_anomaly(self, latent_state: torch.Tensor, stream_data: Optional[Any] = None) -> bool:
        """Evaluates whether the latent state or stream data indicates an anomaly."""
        if hasattr(self.pipeline, "reasoner") and hasattr(self.pipeline.reasoner, "detect_anomaly"):
            try:
                # Support both detect_anomaly(latent_state) and detect_anomaly(latent_state, stream_data)
                return self.pipeline.reasoner.detect_anomaly(latent_state, stream_data)
            except TypeError:
                return self.pipeline.reasoner.detect_anomaly(latent_state)
        return False

    def _resolve_anomaly(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Resolves the anomaly into a target alert/recovery vector."""
        if hasattr(self.pipeline, "reasoner") and hasattr(self.pipeline.reasoner, "resolve_anomaly"):
            return self.pipeline.reasoner.resolve_anomaly(latent_state)
        return F.normalize(latent_state, p=2, dim=-1)

    def _generate_alert(
        self,
        alert_vector: torch.Tensor,
        personality_vector: Optional[torch.Tensor] = None,
        data_stream: Optional[Any] = None
    ) -> str:
        """Generates proactive verbalization addressing Akhil."""
        # 1. Custom personality phrasing via EgoNode
        summary = ""
        if isinstance(data_stream, dict):
            metric = data_stream.get("metric") or "system telemetry"
            val = data_stream.get("value", "")
            unit = data_stream.get("unit", "")
            summary = f"{metric} reaching {val} {unit}".strip()
            if not val:
                summary = data_stream.get("summary") or data_stream.get("text") or "unexpected threshold breach"
        elif isinstance(data_stream, str):
            summary = data_stream
        else:
            summary = "monitored system parameters"

        # Check if pipeline translator generate is provided
        if hasattr(self.pipeline, "translator") and hasattr(self.pipeline.translator, "generate"):
            try:
                gen_text = self.pipeline.translator.generate(alert_vector, personality_vector)
                if gen_text and len(gen_text) > 5:
                    return gen_text
            except Exception as e:
                logger.debug("Translator generate fallback: %s", e)

        # Standard EgoNode verbalization
        user_name = getattr(self.ego_node, "user_name", "Akhil")
        if hasattr(self.ego_node, "format_proactive_alert"):
            return self.ego_node.format_proactive_alert(
                anomaly_summary=summary,
                recommendation="Shall I vent the cooling systems or execute containment?"
            )

        return f"{user_name}, I am detecting an anomaly in {summary}. Shall I vent the cooling systems?"

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Executes a single anticipatory monitoring step:
        1. Monitor data streams (e.g. system logs, news, sensors).
        2. Map the data into the 256D manifold.
        3. Check if the data violates any thresholds in the manifold.
        4. PROACTIVE INITIATION: Jane speaks first, without being asked.
           She uses her Ego_Node (personality) to phrase it correctly.
        """
        data_stream = self.get_latest_stream()
        if data_stream is None:
            return None

        # 2. Map data into 256D manifold
        latent_state = self._encode_stream(data_stream)

        # 3. Check if the data violates any thresholds in the manifold
        if self._detect_anomaly(latent_state, data_stream):
            # 4. Proactive initiation
            alert_vector = self._resolve_anomaly(latent_state)
            personality_vec = getattr(self.ego_node, "personality_vector", None)
            response = self._generate_alert(alert_vector, personality_vec, data_stream)

            # Announce proactively
            print(f"[JANE PROACTIVE]: {response}")
            if self.alert_callback:
                try:
                    self.alert_callback(response)
                except Exception as e:
                    logger.error("Error invoking alert_callback: %s", e)

            # Record in episodic memory ledger if present
            if hasattr(self.pipeline, "episodic_ledger") and self.pipeline.episodic_ledger:
                self.pipeline.episodic_ledger.record_event(
                    action_type="PROACTIVE_ALERT",
                    concept_name="AnticipatoryAnomaly",
                    source_name="AnticipatoryEngine",
                    formula_or_axiom="manifold_energy > threshold",
                    status="COMMITTED",
                    details={"alert": response, "data_stream": str(data_stream)[:200]}
                )

            record = {
                "timestamp": time.time(),
                "data_stream": data_stream,
                "response": response,
                "alert_vector_norm": float(torch.norm(alert_vector).item())
            }
            with self._lock:
                self._detected_anomalies.append(record)
            return record

        return None

    def propose_mitigation_action(self, anomaly_record: Dict[str, Any]) -> ActionASTNode:
        """
        Formulates a formal, Z3-verifiable action to mitigate the detected anomaly.
        Ready to be passed directly to APIGateway.execute_verified_action.
        """
        stream = anomaly_record.get("data_stream", {})
        if isinstance(stream, dict) and "temp" in str(stream).lower():
            return ActionASTNode(
                action_type="HARDWARE_CONTROL",
                payload={"subsystem": "cooling", "valve": "vent_1", "state": "OPEN", "power": 40.0},
                safety_contract="energy_out <= energy_in",
                metadata={"reason": "thermal_mitigation"}
            )
        elif isinstance(stream, dict) and "memory" in str(stream).lower():
            return ActionASTNode(
                action_type="CODE_EXEC",
                payload="import gc; gc.collect(); cleaned = True",
                safety_contract="energy_out <= energy_in",
                metadata={"reason": "memory_cleanup"}
            )
        else:
            return ActionASTNode(
                action_type="HARDWARE_CONTROL",
                payload={"command": "EMERGENCY_VENT", "threshold": 0.5},
                safety_contract="energy_out <= energy_in",
                metadata={"reason": "generic_anomaly_mitigation"}
            )

    def run_background_monitor(self):
        """Runs constantly, looking for things to bring to Akhil's attention."""
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception as e:
                logger.error("Error in anticipatory background loop: %s", e, exc_info=True)
            self._stop_event.wait(self.check_interval_sec)

    def start(self):
        """Starts the background monitor daemon thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self.run_background_monitor,
            name="JaneAnticipatoryMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("AnticipatoryEngine background monitor started (Interval: %.1fs).", self.check_interval_sec)

    def stop(self, timeout: float = 2.0):
        """Stops the background monitor thread gracefully."""
        self._stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
        self._monitor_thread = None
        logger.info("AnticipatoryEngine background monitor stopped.")

    @property
    def is_running(self) -> bool:
        """Returns True if the background thread is currently active."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """Returns history of detected anomalies."""
        with self._lock:
            return list(self._detected_anomalies)

    def clear_detected_anomalies(self):
        """Clears anomaly history."""
        with self._lock:
            self._detected_anomalies.clear()
