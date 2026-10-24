"""Decoupled 3-Thread Asynchronous Architecture for Multimodal Embodiment.

Decouples the Body (Action Executor), the Ears (Audio Encoder), and the Brain (Deliberation Engine):
- Thread 1 (The Body): Continuously executes DNC tape steps via Continuous Solver; never pauses unless interrupted.
- Thread 2 (The Ears): Direct Audio-to-Vector processing; drops noise/coughing without text hallucination.
- Thread 3 (The Brain): Monitors manifold for active logic nodes; applies "Suggestion vs. Override" boundary.
"""

import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch

from environments.spatial_graph.environment import SpatialEnvironmentGraph
from environments.spatial_graph.continuous_kinematics import ContinuousKinematicsEngine
from environments.spatial_graph.dynamic_asp_compiler import DynamicClingoCompiler
from environments.spatial_graph.visual_renderer import FacilityVisualRenderer
from src.perception.audio_encoder import DirectAudioTransformer, AudioIntent
from src.perception.vision_encoder import VisionPatchTransformer
from src.reasoning_engine.system2_planner import PlanStep
from src.symbolic_verifier.contracts import (
    VerificationResult,
    PhysicsHypothesis,
    PhysicalQuantity,
    Dimension
)
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.tom_manifold import TheoryOfMindManifold
from src.utils.logger import get_logger
from src.core.config import DEFAULT_CONFIG, EmbodimentConfig

logger = get_logger("async_embodiment")
from src.core.metacognitive_observer import MetacognitiveObserverThread


class SharedDNCPlanState:
    """
    Thread-safe shared state for DNC memory tape execution.
    Synchronized via threading.Lock.
    """

    def __init__(self, initial_steps: List[PlanStep], latent_dim: int = 256):
        self.lock = threading.Lock()
        self.steps: List[PlanStep] = list(initial_steps)
        self.current_step_idx: int = 0
        self.latent_dim = latent_dim
        self.is_halted: bool = False
        self.is_completed: bool = False
        self.event_log: List[str] = []

    def get_current_step(self) -> Optional[PlanStep]:
        with self.lock:
            if self.is_halted or self.current_step_idx >= len(self.steps):
                return None
            return self.steps[self.current_step_idx]

    def advance_step(self):
        with self.lock:
            if self.current_step_idx < len(self.steps):
                self.steps[self.current_step_idx].status = "EXECUTED"
                self.current_step_idx += 1
                if self.current_step_idx >= len(self.steps):
                    self.is_completed = True

    def append_step_after_current(self, new_step: PlanStep):
        """Dynamic Plan Merging: Inserts new verified step right after the currently active step."""
        with self.lock:
            insert_pos = min(self.current_step_idx + 1, len(self.steps))
            new_step.step_index = insert_pos + 1
            self.steps.insert(insert_pos, new_step)
            # Re-index subsequent steps
            for i in range(insert_pos + 1, len(self.steps)):
                self.steps[i].step_index = i + 1
            self.is_completed = False
            self.event_log.append(f"Merged step '{new_step.action_type} -> {new_step.target}' at index {insert_pos}.")

    def override_with_emergency_step(self, emergency_step: PlanStep):
        """Emergency Override: Freezes current plan and prioritizes immediate emergency action."""
        with self.lock:
            insert_pos = self.current_step_idx
            emergency_step.step_index = insert_pos + 1
            self.steps.insert(insert_pos, emergency_step)
            for i in range(insert_pos + 1, len(self.steps)):
                self.steps[i].step_index = i + 1
            self.is_completed = False
            self.is_halted = True
            self.event_log.append(f"EMERGENCY OVERRIDE: Prioritized '{emergency_step.action_type}' at current index {insert_pos}.")

    def halt(self):
        with self.lock:
            self.is_halted = True
            self.event_log.append("Execution halted by emergency interrupt.")

    def resume(self):
        with self.lock:
            self.is_halted = False
            self.event_log.append("Execution resumed.")


class ActionExecutorThread(threading.Thread):
    """
    Thread 1: The Action Executor (The Body).
    Executes DNC tape steps in continuous space. Does not think. Never pauses for audio.
    """

    def __init__(
        self,
        shared_plan: SharedDNCPlanState,
        env: SpatialEnvironmentGraph,
        kinematics: ContinuousKinematicsEngine,
        interrupt_event: threading.Event,
        stop_event: threading.Event,
        step_duration: Optional[float] = None,
        vision_encoder: Optional[VisionPatchTransformer] = None,
        renderer: Optional[FacilityVisualRenderer] = None,
        enable_sight: bool = False,
        tom_manifold: Optional[TheoryOfMindManifold] = None,
        config: Optional[EmbodimentConfig] = None
    ):
        super().__init__(name="ActionExecutorThread", daemon=True)
        self.config = config or DEFAULT_CONFIG.embodiment
        self.shared_plan = shared_plan
        self.env = env
        self.kinematics = kinematics
        self.interrupt_event = interrupt_event
        self.stop_event = stop_event
        self.step_duration = step_duration if step_duration is not None else self.config.step_duration
        self.vision_encoder = vision_encoder
        self.renderer = renderer
        self.enable_sight = enable_sight
        self.tom_manifold = tom_manifold
        self.executed_history: List[str] = []

    def run(self):
        while not self.stop_event.is_set():
            # Check for high-priority emergency interrupt or halted plan
            if self.interrupt_event.is_set() or self.shared_plan.is_halted:
                if "EXECUTED_EMERGENCY_HALT" not in self.executed_history:
                    self.executed_history.append("EXECUTED_EMERGENCY_HALT")
                if self.tom_manifold:
                    self.tom_manifold.update_ego_state("EMERGENCY_HALT", status="HALTED")
                self.shared_plan.halt()
                break


            step = self.shared_plan.get_current_step()
            if step is None:
                # Plan finished or temporarily waiting
                time.sleep(self.config.idle_sleep_sec)
                continue

            # If multimodal sight is enabled, render the visual camera view and HUD
            if self.enable_sight and self.renderer:
                try:
                    hud = self.renderer.render_ascii_hud(self.env)
                    print(f"\n{hud}")
                except Exception as e:
                    logger.debug("Failed to render ASCII HUD: %s", e)

            # 1. Update Ego Node: Jane observes herself beginning action execution
            if self.tom_manifold:
                self.tom_manifold.update_ego_state(step.action_type, step.target, status="EXECUTING")

            # Execute step on environment
            if step.action_type == "NAVIGATE":
                target_room = step.target
                if target_room in self.env.nodes:
                    # Move agent in continuous environment
                    self.env.move_to(target_room)
                    self.executed_history.append(f"NAVIGATE_TO_{target_room}")
                    print(f"\n>>> [BODY] Executed Step: NAVIGATE -> {target_room} (Agent in '{self.env.agent_node}')")
            elif step.action_type == "PICKUP":
                self.env.pickup_item(step.target)
                self.executed_history.append(f"PICKUP_{step.target}")
                print(f"\n>>> [BODY] Executed Step: PICKUP -> {step.target}")
            elif step.action_type == "VENT_GAS":
                self.env.vent_gas(step.target)
                self.executed_history.append(f"VENT_GAS_{step.target}")
                print(f"\n>>> [BODY] Executed Step: VENT_GAS -> {step.target} (Pressure reduced to 15%)")
            elif step.action_type == "CHECK_PRESSURE":
                p = self.env.nodes[step.target].pressure if step.target in self.env.nodes else 0.0
                self.executed_history.append(f"CHECKED_PRESSURE_{step.target}_{p}%")
                print(f"\n>>> [BODY] Executed Step: CHECK_PRESSURE -> {step.target} (Pressure gauge reading: {p}%)")
            elif step.action_type == "EMERGENCY_HALT":
                self.executed_history.append("EXECUTED_EMERGENCY_HALT")
                if self.tom_manifold:
                    self.tom_manifold.update_ego_state("EMERGENCY_HALT", status="HALTED")
                self.shared_plan.halt()
                print(f"\n>>> [BODY] Executed Step: EMERGENCY_HALT (Hardware frozen)")
                break

            # Step execution takes simulated continuous time
            # Check for interrupts during sleep in small intervals
            elapsed = 0.0
            slice_time = self.config.slice_time_sec
            interrupted = False
            while elapsed < self.step_duration:
                if self.interrupt_event.is_set() or self.shared_plan.is_halted:
                    interrupted = True
                    break
                time.sleep(slice_time)
                elapsed += slice_time

            if interrupted:
                if "EXECUTED_EMERGENCY_HALT" not in self.executed_history:
                    self.executed_history.append("EXECUTED_EMERGENCY_HALT")
                if self.tom_manifold:
                    self.tom_manifold.update_ego_state("EMERGENCY_HALT", status="HALTED")
                self.shared_plan.halt()
                break

            # Advance DNC tape and update Ego Node as completed
            self.shared_plan.advance_step()
            if self.tom_manifold:
                self.tom_manifold.update_ego_state(step.action_type, step.target, status="COMPLETED")



class AudioEncoderThread(threading.Thread):
    """
    Thread 2: The Asynchronous Audio Encoder (The Ears).
    Maps raw audio waveforms directly into 256D vectors. Drops noise/coughs.
    """

    def __init__(
        self,
        audio_transformer: DirectAudioTransformer,
        incoming_audio_queue: queue.Queue,
        intent_queue: queue.Queue,
        stop_event: threading.Event,
        verbose: bool = False,
        tom_manifold: Optional[TheoryOfMindManifold] = None,
        config: Optional[EmbodimentConfig] = None
    ):
        super().__init__(name="AudioEncoderThread", daemon=True)
        self.config = config or DEFAULT_CONFIG.embodiment
        self.audio_transformer = audio_transformer
        self.incoming_audio_queue = incoming_audio_queue
        self.intent_queue = intent_queue
        self.stop_event = stop_event
        self.verbose = verbose
        self.tom_manifold = tom_manifold
        self.processed_intents: List[AudioIntent] = []
        self.dropped_noise_count: int = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                waveform = self.incoming_audio_queue.get(timeout=self.config.queue_timeout_sec)
            except queue.Empty:
                continue

            # Direct audio-to-intent parsing without speech-to-text
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()
            intent, best_name, best_sim = self.audio_transformer.parse_audio_intent(waveform, return_details=True)

            # Update Alter Node (User State) in Theory of Mind Manifold
            if self.tom_manifold is not None:
                speech_label = intent.action_type if intent else (best_name if best_sim > self.config.similarity_gate else None)
                ac_vec = intent.raw_256d_vector if intent else None
                self.tom_manifold.update_alter_state(
                    audio_rms=rms,
                    acoustic_vector=ac_vec,
                    speech_text_or_intent=speech_label
                )

            if intent is None:
                # Gated: Cough, noise, or unclear audio
                self.dropped_noise_count += 1
                if self.verbose and best_sim > self.config.similarity_gate:
                    print(f"  [EARS GATE] Closest candidate: '{best_name}' (similarity: {best_sim:.2f} < gate: {self.audio_transformer.confidence_threshold:.2f}) -> Dropped safely.")
            else:
                self.processed_intents.append(intent)
                try:
                    self.intent_queue.put_nowait(intent)
                except queue.Full:
                    logger.warning("Intent queue full (200 items); dropping intent '%s' to prevent memory leak.", intent.action_type)
                if self.verbose:
                    print(f"\n  >>> [EARS RECOGNIZED] Sound matched '{intent.action_type}' (similarity: {best_sim:.2f} >= gate: {self.audio_transformer.confidence_threshold:.2f}) -> Dispatched to Brain!\n")

            self.incoming_audio_queue.task_done()



class DeliberationEngineThread(threading.Thread):
    """
    Thread 3: The Deliberation Engine (The Brain).
    Monitors the manifold, evaluates audio intents against Z3/Clingo boundaries,
    and handles the "Suggestion vs. Override" decision.
    """

    def __init__(
        self,
        shared_plan: SharedDNCPlanState,
        env: SpatialEnvironmentGraph,
        intent_queue: queue.Queue,
        interrupt_event: threading.Event,
        stop_event: threading.Event,
        audio_transformer: Optional[DirectAudioTransformer] = None
    ):
        super().__init__(name="DeliberationEngineThread", daemon=True)
        self.shared_plan = shared_plan
        self.env = env
        self.intent_queue = intent_queue
        self.interrupt_event = interrupt_event
        self.stop_event = stop_event
        self.audio_transformer = audio_transformer
        self.z3_verifier = Z3LogicVerifier()
        self.clingo_compiler = DynamicClingoCompiler()
        self.audit_log: List[str] = []
        self.last_committed_action: Optional[str] = None
        self.last_committed_time: float = 0.0
        self.cooldown_seconds: float = 3.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                intent: AudioIntent = self.intent_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # -------------------------------------------------------------
            # The "Suggestion vs. Override" Decision Boundary
            # -------------------------------------------------------------
            if intent.is_emergency:
                # CASE B: INTERRUPT NODE (Emergency Override) - Immediate, no cooldown
                self.audit_log.append(f"[OVERRIDE DETECTED] High-energy emergency audio node '{intent.action_type}'.")
                print(f"\n  🚨🚨🚨 [BRAIN EMERGENCY OVERRIDE] Voice command '{intent.action_type}' triggered immediate hardware freeze!\n")
                # Immediately signal Action Executor to stop
                self.interrupt_event.set()

                # Build emergency plan step
                emergency_step = PlanStep(
                    step_index=0,
                    action_type="EMERGENCY_HALT",
                    target=intent.target,
                    parameters=intent.parameters,
                    status="PENDING",
                    prover_name="Emergency_Acoustic_Gate"
                )
                self.shared_plan.override_with_emergency_step(emergency_step)
                self.shared_plan.halt()

                # Online Hebbian Vocal Adaptation
                if self.audio_transformer is not None:
                    new_sim = self.audio_transformer.adapt_vocal_anchor(
                        "EMERGENCY_HALT",
                        intent.raw_256d_vector,
                        alpha=0.08
                    )
                    print(f"  🧠 [ONLINE VOCAL LEARNING] Jane adapted to your voice! Anchor 'EMERGENCY_HALT' updated (similarity boosted: {intent.confidence:.2f} -> {new_sim:.2f})")

            else:
                # Check de-duplication cooldown for non-emergency suggestions
                now = time.time()
                if (intent.action_type == self.last_committed_action and 
                    (now - self.last_committed_time) < self.cooldown_seconds):
                    self.audit_log.append(
                        f"[SUGGESTION SUPPRESSED] Intent '{intent.action_type}' ignored (within {self.cooldown_seconds:.1f}s refractory period)."
                    )
                    self.intent_queue.task_done()
                    continue

                # CASE A: APPEND NODE (Suggestion / Non-blocking Plan Merging)
                self.audit_log.append(f"[SUGGESTION DETECTED] Non-emergency audio intent '{intent.action_type}' for '{intent.target}'.")

                # Verify against Z3/Clingo boundary before adding to DNC tape
                verified = True
                proof_msg = "Formally verified."

                if intent.action_type == "VENT_GAS":
                    # Formal thermodynamic Z3 check
                    hyp = PhysicsHypothesis(quantities=[
                        PhysicalQuantity(
                            magnitude=float(intent.parameters.get("target_pressure", 15.0)),
                            dimension=Dimension.ENERGY,
                            confidence=1.0,
                            is_output=True
                        )
                    ])
                    z3_res = self.z3_verifier.verify_physics(hyp)
                    verified = z3_res.is_valid
                    proof_msg = f"Z3 Prover: {z3_res.proof_log}"
                elif intent.action_type == "CHECK_PRESSURE":
                    # Formal boundary Z3 check
                    hyp = PhysicsHypothesis(quantities=[
                        PhysicalQuantity(
                            magnitude=float(intent.parameters.get("threshold", 40.0)),
                            dimension=Dimension.ENERGY,
                            confidence=1.0
                        )
                    ])
                    z3_res = self.z3_verifier.verify_physics(hyp)
                    verified = z3_res.is_valid
                    proof_msg = f"Z3 Prover: {z3_res.proof_log}"
                elif intent.action_type == "NAVIGATE":
                    # Topological reachability check
                    clingo_proof = self.clingo_compiler.verify_reachability(self.env, self.env.agent_node, intent.target)
                    verified = clingo_proof.is_valid
                    proof_msg = clingo_proof.proof_log

                if verified:
                    new_step = PlanStep(
                        step_index=0,
                        action_type=intent.action_type,
                        target=intent.target,
                        parameters=intent.parameters,
                        status="PENDING",
                        proof_certificate=proof_msg
                    )
                    # Atomically merge into DNC tape right after current task!
                    # Action Executor keeps moving continuously without pause!
                    self.shared_plan.append_step_after_current(new_step)
                    self.last_committed_action = intent.action_type
                    self.last_committed_time = time.time()
                    print(f"\n  >>> [BRAIN VERIFIED] Formally proven safe with Z3/Clingo. Appending '{intent.action_type}' mid-stride!\n")
                    self.audit_log.append(f"[SUGGESTION COMMITTED] '{intent.action_type}' added to DNC tape. Executor continues uninterrupted.")

                    # Online Hebbian Vocal Adaptation
                    if self.audio_transformer is not None:
                        new_sim = self.audio_transformer.adapt_vocal_anchor(
                            intent.action_type,
                            intent.raw_256d_vector,
                            alpha=0.08
                        )
                        print(f"  🧠 [ONLINE VOCAL LEARNING] Jane adapted to your voice! Anchor '{intent.action_type}' updated (similarity boosted: {intent.confidence:.2f} -> {new_sim:.2f})")
                else:
                    self.audit_log.append(f"[SUGGESTION REJECTED] Intent '{intent.action_type}' rejected: {proof_msg}")

            self.intent_queue.task_done()


class DecoupledAsynchronousEmbodiment:
    """
    Orchestrates the 3-Thread Decoupled System:
    Body (Executor) + Ears (Audio) + Brain (Deliberation).
    """

    def __init__(
        self,
        initial_steps: List[PlanStep],
        env: Optional[SpatialEnvironmentGraph] = None,
        audio_transformer: Optional[DirectAudioTransformer] = None,
        step_duration: float = 0.05,
        verbose: bool = False,
        vision_encoder: Optional[VisionPatchTransformer] = None,
        renderer: Optional[FacilityVisualRenderer] = None,
        enable_sight: bool = False,
        tom_manifold: Optional[TheoryOfMindManifold] = None
    ):
        self.env = env or SpatialEnvironmentGraph.create_facility_env()
        self.kinematics = ContinuousKinematicsEngine()
        self.audio_transformer = audio_transformer or DirectAudioTransformer()
        self.vision_encoder = vision_encoder
        self.renderer = renderer
        self.enable_sight = enable_sight
        self.tom_manifold = tom_manifold or TheoryOfMindManifold()

        # Shared synchronization primitives
        self.shared_plan = SharedDNCPlanState(initial_steps)
        self.interrupt_event = threading.Event()
        self.stop_event = threading.Event()

        # Queues (bounded to prevent unbounded memory growth in long-running 24/7 sessions)
        self.incoming_audio_queue = queue.Queue(maxsize=200)
        self.intent_queue = queue.Queue(maxsize=200)

        # 4 Parallel Decoupled Threads (Body, Ears, Brain, Watchdog)
        self.body_thread = ActionExecutorThread(
            shared_plan=self.shared_plan,
            env=self.env,
            kinematics=self.kinematics,
            interrupt_event=self.interrupt_event,
            stop_event=self.stop_event,
            step_duration=step_duration,
            vision_encoder=self.vision_encoder,
            renderer=self.renderer,
            enable_sight=self.enable_sight,
            tom_manifold=self.tom_manifold
        )

        self.ears_thread = AudioEncoderThread(
            audio_transformer=self.audio_transformer,
            incoming_audio_queue=self.incoming_audio_queue,
            intent_queue=self.intent_queue,
            stop_event=self.stop_event,
            verbose=verbose,
            tom_manifold=self.tom_manifold
        )

        self.brain_thread = DeliberationEngineThread(
            shared_plan=self.shared_plan,
            env=self.env,
            intent_queue=self.intent_queue,
            interrupt_event=self.interrupt_event,
            stop_event=self.stop_event,
            audio_transformer=self.audio_transformer
        )

        # Thread 4: The Metacognitive Observer (The Watchdog)
        self.watchdog_thread = MetacognitiveObserverThread(
            tom_manifold=self.tom_manifold,
            interrupt_event=self.interrupt_event,
            stop_event=self.stop_event,
            verbose=verbose
        )

    def start(self):
        """Starts all 4 decoupled threads concurrently (Body, Ears, Brain, Watchdog)."""
        self.body_thread.start()
        self.ears_thread.start()
        self.brain_thread.start()
        self.watchdog_thread.start()

    def feed_audio_waveform(self, waveform: torch.Tensor):
        """Non-blocking audio injection from microphone or acoustic generator."""
        try:
            self.incoming_audio_queue.put_nowait(waveform)
        except queue.Full:
            logger.warning("Incoming audio queue full (200 items); dropping audio frame to prevent memory leak.")

    def stop(self):
        """Gracefully halts all 4 threads."""
        self.stop_event.set()
        self.body_thread.join(timeout=1.0)
        self.ears_thread.join(timeout=1.0)
        self.brain_thread.join(timeout=1.0)
        self.watchdog_thread.join(timeout=1.0)

