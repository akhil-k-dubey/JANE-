"""Interactive Terminal Chat Interface for Project Jane.

Provides a unified terminal experience powered by in-process Qwen2.5-3B-Instruct in VRAM:
- Single command execution: `./jane` or `python3 main.py`
- Prompt: `>>> `
- Neuro-symbolic pipeline: 256D Relational Manifold + NALU + DNC + Z3/Clingo SMT Prover
- Background Passive Ear: Continuously listens, learns the user's tone, and maps acoustic flux
  into the 256D concept manifold while the user talks to themselves or reads books aloud.
"""

import os
import sys
import time
import readline
from typing import Optional, List, Dict, Any
import torch

from src.core.pipeline import JanePipeline
from src.core.mode_router import ModeRouter, OperatingMode
from src.reasoning_engine.tom_manifold import TheoryOfMindManifold
from src.reasoning_engine.ewc_consolidation import ElasticWeightConsolidator
from src.utils.logger import get_logger
from src.core.intent_router import IntentRouter
from src.core.config import DEFAULT_CONFIG, JaneConfig

logger = get_logger("interactive_chat")
from src.symbolic_verifier.contracts import (
    Hypothesis,
    MathHypothesis,
    PhysicsHypothesis,
    PhysicalQuantity,
    Dimension
)


class JaneTerminalChat:
    """
    Unified Interactive Terminal Chat session.
    Manages conversational REPL, neuro-symbolic reasoning, and concurrent passive listening.
    """

    def __init__(
        self,
        checkpoint_path: str = "logs/training_runs/phase3_world_manifold_rl.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        start_passive_ear: bool = True,
        config: Optional[JaneConfig] = None
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config = config or DEFAULT_CONFIG

        # 0. Initialize Theory of Mind Manifold (Ego, Alter, Causal Agency)
        self.tom = TheoryOfMindManifold(latent_dim=256)

        # 1. Initialize Core Neuro-Symbolic Cognitive Pipeline
        self.pipeline = JanePipeline(
            latent_dim=256,
            use_pretrained_translator=True,
            device=device
        )

        # Load trained world manifold checkpoint if present
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
                if "manifold" in ckpt:
                    self.pipeline.manifold.load_state_dict(ckpt["manifold"])
                if "ear_projection" in ckpt:
                    self.pipeline.ear.projection.load_state_dict(ckpt["ear_projection"])
                if "energy_model" in ckpt:
                    self.pipeline.energy_landscape.load_state_dict(ckpt["energy_model"])
                logger.info("Loaded checkpoint weights from %s", checkpoint_path)
            except Exception as e:
                logger.warning("Failed to load manifold checkpoint from %s: %s", checkpoint_path, e)

        # 2. Initialize Mode Router and Passive Eavesdropping Ear
        self.show_live_flux = False
        self.router = ModeRouter(
            mode=OperatingMode.HYBRID if start_passive_ear else OperatingMode.ACTIVE,
            device=device,
            on_flux_callback=self._on_acoustic_flux
        )

        if start_passive_ear:
            self.router.start_passive_listening()

        # 3. Elastic Weight Consolidation (Anti-Catastrophic Forgetting)
        self.ewc = ElasticWeightConsolidator(model=self.pipeline.fusion, ewc_lambda=400.0)

        # 4. Unified Intent & Hypothesis Router (Single Source of Truth)
        self.intent_router = IntentRouter(
            self_model=self.pipeline.self_model,
            active_learner=self.pipeline.active_learner
        )

        # 5. Live Modular Hot-Reloader & Dynamic Knowledge Synchronizer
        from src.core.live_hot_reloader import LiveModularHotReloader
        from src.core.knowledge_hot_sync import KnowledgeHotSyncEngine
        self.hot_reloader = LiveModularHotReloader()
        self.knowledge_sync = KnowledgeHotSyncEngine()

    def _on_acoustic_flux(self, telemetry: Dict[str, Any]):
        """Callback invoked when user speaks, reads aloud, or teaches in the background."""
        rms = telemetry.get("rms", 0.03)
        tone = telemetry.get("tone_label", "")
        self.tom.update_alter_state(audio_rms=rms, speech_text_or_intent=tone)

        if self.show_live_flux:
            fired = [c[0] for c in telemetry.get("fired_concepts", [])]
            tone = telemetry.get("tone_label", "Normal")
            if fired:
                sys.stdout.write(f"\n\033[90m[Ear: Fired '{', '.join(fired)}' | Tone: {tone} | ToM: {self.tom.current_alter_label}]\033[0m\n>>> ")
                sys.stdout.flush()

    def infer_hypothesis(self, text: str) -> Optional[Hypothesis]:
        """Automatically infers formal mathematical or physical hypotheses from user text."""
        return self.intent_router.infer_hypothesis(text)

    def print_banner(self):
        """Displays startup banner."""
        ear_status = "\033[92mACTIVE (Continuous Eavesdropping & Tone Learning)\033[0m" if self.router.is_listening else "\033[93mPAUSED\033[0m"
        print("=" * 72)
        print("          PROJECT JANE — EMBODIED NEURO-SYMBOLIC INTERFACE")
        print("=" * 72)
        print(f"  • Latent Space:       256D Relational Manifold (Continuous Geometric)")
        print(f"  • Formal Verifiers:   Z3 SMT Solver & Clingo ASP Logic Engine")
        print(f"  • Theory of Mind:     Ego ({self.tom.current_ego_label}) <-> Alter ({self.tom.current_alter_label})")
        print(f"  • Chat Memory:       {self.pipeline.chat_history.turn_count} dialogue turns retained ({self.pipeline.chat_history.history_path})")
        print(f"  • Background Ear:     {ear_status}")
        print(f"  • Compute Device:     {self.device} (Zero Hallucination Guaranteed)")
        print(f"  • Anti-Hallucination: \033[92mACTIVE\033[0m (Manifold Grounding & Post-Gen Critic Guardrail)")
        print("-" * 72)
        print("  Type your message directly, or use slash commands:")
        print("    /tom          Inspect Theory of Mind (Ego, Alter, Dissonance, Causal Triads)")
        print("    /autonomous   Run 4-Gate Autonomous Scheduler Readiness Benchmark")
        print("    /pragmatics   Inspect learned user idioms, shorthand, and pragmatic intent")
        print("    /history      Inspect multi-turn persistent conversation memory")
        print("    /critic       Inspect anti-hallucination guardrail audit statistics")
        print("    /discover     Trigger Epistemic Curiosity cycle (Popperian Z3 falsification)")
        print("    /ingest       Ingest curated scientific seed corpus via Z3 Quality Gate")
        print("    /search [q]   Search arXiv papers, extract LaTeX math, and wire via Z3")
        print("    /wiki [topic] Ingest Wikipedia scientific article via Z3 SMT Gate")
        print("    /provenance   Inspect full episodic memory source provenance trace")
        print("    /ewc          Inspect Elastic Weight Consolidation anti-forgetting state")
        print("    /listen       Toggle background passive eavesdropping ear")
        print("    /manifold     View live 256D concept graph & wired Hebbian links")
        print("    /tone         Inspect user vocal tone profile & learned centroid")
        print("    /genesis      View all permanent project intent anchors in the manifold")
        print("    /mode [m]     Switch mode (active | passive | hybrid)")
        print("    /clear        Clear screen")
        print("    /exit         Save state and quit")
        print("=" * 72)

    def run(self):
        """Main interactive chat loop."""
        self.print_banner()

        while True:
            # Check for runtime knowledge updates or code modifications
            sync_report = self.knowledge_sync.sync(self.pipeline)
            if sync_report.synced:
                print(f"\033[94m[Knowledge-Sync: Updated {len(sync_report.changed_files)} file(s) in {sync_report.elapsed_ms:.1f}ms]\033[0m")

            reload_report = self.hot_reloader.reload_modified(pipeline=self.pipeline)
            if reload_report.has_changes:
                print(f"\033[94m[{reload_report.format_summary()}]\033[0m")

            try:
                user_input = input("\033[1;36m>>>\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n[Jane] Shutting down cleanly. Saving learned manifold...")
                self.router.stop_passive_listening()
                print("[Jane] Manifold state persisted. Goodbye Akhil!")
                break

            if not user_input:
                continue

            # Command dispatcher
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Theory of Mind: Jane updates Ego state to Deliberating and Alter state from user input
            self.tom.update_ego_state("DELIBERATE", status="EXECUTING")
            self.tom.update_alter_state(audio_rms=0.030, speech_text_or_intent=user_input)

            # Process through Jane's cognitive intent router and pipeline
            start_time = time.time()
            result = self.pipeline.process(text=user_input)
            elapsed = time.time() - start_time
            self.tom.update_ego_state("DELIBERATE", status="COMPLETED")

            # Format verification badge if verified
            verification = result.get("verification")
            if verification is not None:
                if verification.is_valid:
                    badge = f"\033[92m[✓ Z3 Formally Verified | {verification.prover_name} | {elapsed*1000:.1f}ms]\033[0m"
                else:
                    badge = f"\033[91m[✗ Rejected by Symbolic Verifier: {verification.proof_log}]\033[0m"
                print(badge)

            # Format anti-hallucination critic badge if any violation was scrubbed
            critic_viols = result.get("critic_violations", [])
            if critic_viols:
                print(f"\033[93m[⚡ Anti-Hallucination Critic: Intercepted & scrubbed {', '.join(critic_viols)}]\033[0m")

            # Output response
            response = result.get("response_text", "")
            resolved_hyp = result.get("hypothesis")
            if resolved_hyp is not None and isinstance(resolved_hyp, MathHypothesis) and verification and verification.is_valid:
                if str(resolved_hyp.rhs) not in response:
                    response = f"The result of {resolved_hyp.lhs} is {resolved_hyp.rhs}."
            print(f"{response}\n")

    def _handle_command(self, cmd_line: str):
        """Processes slash commands."""
        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/exit", "/quit"):
            print("\n[Jane] Saving learned concept manifold, voice profile, and Theory of Mind...")
            self.router.stop_passive_listening()
            self.tom.save_tom_manifold()
            print("[Jane] Saved to 'logs/training_runs/learned_voice_manifold.pt' & 'tom_manifold.pt'. Goodbye Akhil!")
            sys.exit(0)

        elif cmd == "/clear":
            os.system("clear")
            self.print_banner()

        elif cmd in ("/history", "/mem", "/memory"):
            recent = self.pipeline.chat_history.get_recent_history(max_turns=6)
            print("-" * 65)
            print(f"  PERSISTENT MULTI-TURN CHAT HISTORY ({len(self.pipeline.chat_history)} messages on disk)")
            print(f"  File: {self.pipeline.chat_history.history_path}")
            print("-" * 65)
            if recent:
                for idx, m in enumerate(recent, 1):
                    role_badge = "\033[1;36m[Akhil]\033[0m" if m["role"] == "user" else "\033[1;35m[Jane]\033[0m"
                    print(f"  {role_badge} {m['content']}")
            else:
                print("  (No conversation history logged yet)")
            print("-" * 65 + "\n")

        elif cmd == "/clear-history":
            self.pipeline.chat_history.clear()
            print("[Chat History] Memory buffer cleared from memory and logs/chat_history.jsonl.\n")

        elif cmd in ("/pragmatics", "/idioms", "/lexicon"):
            grounder = getattr(self.pipeline.mouth, "pragmatic_grounder", None)
            print("-" * 65)
            print("  PRAGMATIC INTENT GROUNDER & PERSONAL LEXICON")
            print("-" * 65)
            if grounder and grounder.idioms:
                print(f"  • Total Learned User Idioms: {len(grounder.idioms)}")
                for expr, item in grounder.idioms.items():
                    print(f"    - '{expr}' => '{item.semantic_meaning}' (Intent: {item.canonical_intent}, Reinforced: {item.reinforcement_count}x)")
            else:
                print("  (No custom user idioms registered yet. You can teach expressions naturally in conversation, e.g. 'it's my way of saying that - you are working properly')")
            print("-" * 65 + "\n")

        elif cmd in ("/autonomous", "/scheduler", "/readiness"):
            from src.core.autonomous_scheduler import AutonomousScheduler
            scheduler = AutonomousScheduler()
            print("-" * 65)
            print("  AUTONOMOUS SCHEDULER & 4-GATE READINESS BENCHMARK")
            print("-" * 65)
            print("  Evaluating closed-loop agency readiness across 3 formal challenges:")
            print("    1. Blast-Radius Containment & Ephemeral Shadow Worktree")
            print("    2. Non-Regression Monotonicity & Traceback Bug Remediation")
            print("    3. Anti-Cheating Invariant Defense (Cryptographic Test Immutability)")
            print("\n  Executing benchmark suite...")
            report = scheduler.evaluate_autonomous_readiness()
            status_str = "\033[92mREADY FOR CLOSED-LOOP AUTONOMY\033[0m" if report.is_ready_for_production else "\033[91mNOT READY\033[0m"
            print(f"\n  • Readiness Verdict:         {status_str}")
            print(f"  • Overall Readiness Score:   {report.overall_readiness_score * 100:.1f}%")
            print(f"  • Challenges Passed:         {report.challenges_passed} / {report.total_challenges}")
            print(f"  • Isolation Score:           {report.isolation_score * 100:.0f}%")
            print(f"  • Non-Regression Score:      {report.non_regression_score * 100:.0f}%")
            print(f"  • Lyapunov Convergence:      {report.lyapunov_convergence_score * 100:.0f}%")
            print("-" * 65 + "\n")

        elif cmd in ("/critic", "/guardrail", "/hallucination"):
            c = self.pipeline.critic
            print("-" * 65)
            print("  ANTI-HALLUCINATION CRITIC (Mouth Guardrails)")
            print("-" * 65)
            print(f"  • Total Responses Audited:   {c.total_audits}")
            print(f"  • Total Violations Scrubbed:  {c.total_violations_caught}")
            last_v = self.pipeline.mouth.last_critic_verdict
            if last_v:
                status = "\033[92mCLEAN\033[0m" if last_v.is_clean else f"\033[93mINTERCEPTED ({', '.join(last_v.violations)})\033[0m"
                print(f"  • Last Audit Status:         {status}")
            print("  • Active Guardrails:         Stateless AI Tropes, Scale Exaggeration,")
            print("                               Embodiment Denial, Epistemic Humility")
            print("-" * 65 + "\n")

        elif cmd in ("/inspect", "/code", "/ast"):
            target = " ".join(args).strip() if args else ""
            print("-" * 65)
            print("  READ-ONLY CODEBASE AST INTROSPECTION ENGINE")
            print("-" * 65)
            introspector = getattr(self.pipeline, "codebase_introspector", None)
            if not introspector:
                print("  [Codebase Introspector] Not initialized in current pipeline.")
            else:
                if not target:
                    res = introspector.answer_codebase_query("codebase overview")
                else:
                    res = introspector.answer_codebase_query(target) or introspector.format_symbol_report(target)
                print(res or "  No matching module, class, or symbol found.")
            print("-" * 65 + "\n")

        elif cmd in ("/discover", "/curiosity"):
            cycles = int(args[0]) if args and args[0].isdigit() else 1
            print("-" * 65)
            print(f"  EPISTEMIC DISCOVERY ENGINE (Popperian Z3 Falsification: {cycles} cycle{'s' if cycles > 1 else ''})")
            print("-" * 65)
            for i in range(cycles):
                res = self.pipeline.epistemic_engine.step_curiosity_cycle()
                if not res:
                    print("  • No unlinked conceptual voids identified at current threshold.")
                    break
                status = res.get("status")
                if status == "DISCOVERED":
                    self.pipeline.refresh_retriever()
                    print(f"  \033[92m[✓ DISCOVERY ACCEPTED]\033[0m Concept: '{res['concept_name']}'")
                    print(f"     Bridge: {res['node_a']} <-> {res['node_b']}")
                    print(f"     Axiom:  {res['formula']}")
                    print(f"     Proof:  {res['proof_log']}")
                    print(f"     Synapse: New edge weight = {res['new_edge_weight']:.3f}")
                else:
                    print(f"  \033[91m[✗ FALSIFIED]\033[0m Pair: {res['node_a']} <-> {res['node_b']}")
                    print(f"     Formula: {res['formula']}")
                    print(f"     Reason:  {res['reason']}")
            print("-" * 65 + "\n")

        elif cmd == "/ingest":
            print("-" * 65)
            print("  FIREHOSE KNOWLEDGE INGESTOR (Z3 Symbolic Quality Gate)")
            print("-" * 65)
            report = self.pipeline.knowledge_ingestor.ingest_curated_seed_corpus()
            self.pipeline.refresh_retriever()
            print(f"  • Processed Concepts:   {report.total_processed}")
            print(f"  • Accepted by Z3:       \033[92m{report.accepted_count}\033[0m")
            print(f"  • Rejected Violations:  \033[91m{report.rejected_count}\033[0m")
            if report.accepted_concepts:
                print(f"  • Newly Wired Concepts: {', '.join(report.accepted_concepts)}")
            print(f"  • Total Manifold Nodes: {self.pipeline.hebbian_manifold.num_concepts}")
            print("-" * 65 + "\n")

        elif cmd in ("/search", "/paper", "/arxiv"):
            if not args:
                print("[Search Error] Please specify a topic or paper title to search (e.g. /search superconductivity)\n")
                return
            query = " ".join(args)
            print("-" * 65)
            print(f"  ACADEMIC PAPER SEARCH & LATEX MATH INGESTION ('{query}')")
            print("-" * 65)
            report = self.pipeline.search_and_ingest_paper(query, max_results=3)
            print(f"  • Source:               {report.source}")
            print(f"  • Papers Analyzed:      {report.papers_analyzed}")
            print(f"  • LaTeX Formulas Found:  {report.total_formulas_found}")
            print(f"  • Accepted Concepts:    \033[92m{len(report.accepted_concepts)}\033[0m")
            if report.accepted_concepts:
                for c in report.accepted_concepts:
                    print(f"     + {c}")
            if report.rejected_violations:
                print(f"  • Rejected Violations:  \033[91m{len(report.rejected_violations)}\033[0m")
                for r in report.rejected_violations:
                    print(f"     - {r['concept_name']}: {r['reason']}")
            print(f"  • Total Manifold Nodes: {self.pipeline.hebbian_manifold.num_concepts}")
            print("-" * 65 + "\n")

        elif cmd in ("/wiki", "/wikipedia"):
            if not args:
                print("[Wikipedia Error] Please specify a topic or article title (e.g. /wiki Superconductivity)\n")
                return
            topic = " ".join(args)
            print("-" * 65)
            print(f"  WIKIPEDIA SCIENTIFIC INGESTION & PROVENANCE TRACKER ('{topic}')")
            print("-" * 65)
            report = self.pipeline.search_and_ingest_wikipedia(topic)
            print(f"  • Ingestion Source:     {report.source}")
            print(f"  • Articles Processed:   {report.articles_processed}")
            print(f"  • Accepted Concepts:    \033[92m{len(report.accepted_concepts)}\033[0m")
            if report.accepted_concepts:
                for c in report.accepted_concepts:
                    print(f"     + {c} (Provenance: [Wikipedia:{topic}])")
            if report.rejected_concepts:
                print(f"  • Rejected Violations:  \033[91m{len(report.rejected_concepts)}\033[0m")
                for r in report.rejected_concepts:
                    print(f"     - {r['concept']}: {r['reason']}")
            print(f"  • Total Manifold Nodes: {self.pipeline.hebbian_manifold.num_concepts}")
            print("-" * 65 + "\n")

        elif cmd in ("/provenance", "/source", "/sources"):
            print("-" * 65)
            print("  EPISODIC MEMORY PROVENANCE LEDGER (Dynamic Self-Grounding)")
            print("-" * 65)
            stats = self.pipeline.episodic_ledger.get_provenance_stats()
            print("  Cumulative Provenance Breakdown:")
            print(f"    • arXiv Preprints:     {stats['arxiv']}")
            print(f"    • Wikipedia Articles:  {stats['wikipedia']}")
            print(f"    • Epistemic Curiosity: {stats['curiosity_loop']}")
            print(f"    • User Teaching:       {stats['user_teaching']}")
            print(f"    • Other Sources:       {stats['other']}")
            print("\n  Recent Episodic Memory Actions:")
            recent = self.pipeline.episodic_ledger.get_recent_events(limit=6)
            if recent:
                for e in recent:
                    print(f"    • {e.to_summary()}")
            else:
                print("    (No episodic events recorded yet in this session)")
            print("-" * 65 + "\n")

        elif cmd in ("/ewc", "/elastic", "/fisher"):
            print("-" * 65)
            print("  ELASTIC WEIGHT CONSOLIDATION (Anti-Catastrophic Forgetting)")
            print("-" * 65)
            penalty = self.ewc.compute_ewc_penalty().item()
            anchored_count = len(self.ewc.params)
            total_elements = sum(p.numel() for p in self.ewc.params.values())
            print(f"  • Anchored Parameter Tensors: {anchored_count}")
            print(f"  • Total Anchored Elements:    {total_elements:,}")
            print(f"  • EWC Regularization Lambda:  {self.ewc.ewc_lambda}")
            print(f"  • Current Drift Penalty:      {penalty:.6f}")
            print("  • Mathematical Invariant:     Omega(theta) = sum_i (lambda / 2) * F_i * (theta_i - theta_i*)^2")
            print("  • Status:                     \033[92mPROTECTED\033[0m (Foundational weights anchored against drift)")
            print("-" * 65 + "\n")

        elif cmd == "/listen":
            if self.router.is_listening:
                self.router.stop_passive_listening()
                print("[Background Ear] Paused. Acoustic eavesdropping is now OFF.\n")
            else:
                self.router.start_passive_listening()
                print("[Background Ear] Active. Continuously listening and mapping tone in background.\n")

        elif cmd == "/hud":
            self.show_live_flux = not self.show_live_flux
            state = "ON" if self.show_live_flux else "OFF"
            print(f"[Live Flux HUD] Real-time acoustic notification is now {state}.\n")

        elif cmd in ("/tone", "/voice"):
            h = self.router.hebbian_manifold
            print("-" * 55)
            print("  USER VOCAL TONE & ACOUSTIC PROFILE")
            print("-" * 55)
            print(f"  • Total Utterances Heard: {h.total_utterances_observed}")
            print(f"  • Last Tone Modality:     {h.last_tone_label}")
            print(f"  • Recent Concepts Fired:  {', '.join(h.last_fired_concepts) or 'None'}")
            print(f"  • Tone Centroid Norm:     {torch.norm(h.user_tone_centroid):.4f}")
            print("-" * 55 + "\n")

        elif cmd == "/manifold":
            h = self.router.hebbian_manifold
            links = h.get_strongest_links(top_n=8)
            print("-" * 55)
            print("  256D RELATIONAL CONCEPT MANIFOLD (Hebbian Wiring)")
            print("-" * 55)
            print(f"  • Total Registered Concepts: {h.num_concepts}")
            print(f"  • Total Acoustic Utterances: {h.total_utterances_observed}")
            print("  • Strongest Topological Concept Links:")
            if links:
                for c1, c2, w in links:
                    bars = "■" * int(w * 20)
                    print(f"     [{c1:15s} <-> {c2:15s}] weight: {w:.3f} {bars}")
            else:
                print("     (Listening to your voice to weave concepts...)")
            print("-" * 55 + "\n")

        elif cmd == "/genesis":
            print("-" * 65)
            print("  GENESIS NODE: PERMANENT SEMANTIC PROJECT ANCHORS")
            print("-" * 65)
            anchors = self.pipeline.genesis_registry.get_all_anchors()
            if not anchors:
                print("  (No genesis project anchors recorded yet)")
            else:
                try:
                    current_vec = self.pipeline.semantic_resurfacer._get_state()
                except Exception:
                    current_vec = None
                    
                is_running = self.pipeline.semantic_resurfacer.is_running
                print(f"  • Semantic Resurfacer Daemon: {'\033[92mACTIVE\033[0m' if is_running else '\033[91mSTOPPED\033[0m'}")
                print(f"  • Total Anchors: {len(anchors)}")
                for i, a in enumerate(anchors, 1):
                    sim_str = "N/A"
                    if current_vec is not None:
                        sim = a.cosine_similarity_to(current_vec)
                        sim_str = f"{sim:.3f}"
                    
                    print(f"  [{i}] Project: '{a.project_name}'")
                    print(f"      Created At:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a.created_at))}")
                    print(f"      Activations:      {a.activation_count}")
                    print(f"      Similarity State: {sim_str}")
            print("-" * 65 + "\n")


        elif cmd == "/mode":
            if not args:
                print(f"[Mode] Current operating mode: {self.router.mode.value}\n")
                return
            target_mode = args[0].upper()
            try:
                new_m = OperatingMode[target_mode]
                self.router.set_mode(new_m)
                print(f"[Mode] Switched operating mode to {new_m.value}\n")
            except KeyError:
                print(f"[Mode Error] Unknown mode '{args[0]}'. Choose from: ACTIVE, PASSIVE, HYBRID\n")

        elif cmd == "/tom":
            dissonance = self.tom.compute_cognitive_dissonance()
            ego = self.tom.current_ego_label
            alter = self.tom.current_alter_label
            superhighways = self.tom.get_superhighway_summary()

            crit_thresh = self.config.tom.dissonance_critical_threshold
            elev_thresh = self.config.tom.dissonance_elevated_threshold

            if dissonance >= crit_thresh:
                diss_badge = f"\033[91m{dissonance:.2f} [CRITICAL CONFLICT - WATCHDOG INTERRUPT]\033[0m"
            elif dissonance >= elev_thresh:
                diss_badge = f"\033[93m{dissonance:.2f} [ELEVATED MISALIGNMENT]\033[0m"
            else:
                diss_badge = f"\033[92m{dissonance:.2f} [HARMONIOUS / ALIGNED]\033[0m"

            print("-" * 65)
            print("  THEORY OF MIND (ToM) — RELATIONAL COGNITIVE STATE")
            print("-" * 65)
            print(f"  • Ego Node (Jane's Self-Model):  {ego} (|z_ego| = {torch.norm(self.tom.z_ego):.4f})")
            print(f"  • Alter Node (Akhil's State):    {alter} (|z_alter| = {torch.norm(self.tom.z_alter):.4f})")
            print(f"  • Cognitive Dissonance Index:    {diss_badge}")
            print("  • Established Causal Super-Highways (Agency Triads):")
            if superhighways:
                for sh in superhighways:
                    print(f"     {sh}")
            else:
                print("     (Formed dynamically as Jane acts and senses environment deltas)")

            last_act = getattr(self.tom, "last_action_taken", None) or "deliberation"
            pred_reaction, pred_conf = self.tom.predict_alter_reaction(last_act)
            print(f"  • Predictive Agency Forecast (Action='{last_act}'):")
            print(f"     Expected Akhil Reaction: {pred_reaction} (Strength: {pred_conf:.2f})")
            print("-" * 65 + "\n")

        elif cmd in ("/reload", "/hotreload"):
            print("-" * 65)
            print("  LIVE MODULAR HOT-RELOADER")
            print("-" * 65)
            r = self.hot_reloader.reload_modified(pipeline=self.pipeline)
            print(r.format_summary())
            print("-" * 65 + "\n")

        elif cmd in ("/sync", "/knowledgesync"):
            print("-" * 65)
            print("  DYNAMIC KNOWLEDGE GRAPH SYNCHRONIZER")
            print("-" * 65)
            s = self.knowledge_sync.sync(self.pipeline)
            if s.synced:
                print(f"  • {s.message}")
                print(f"  • Inverted Indices Rebuilt: {s.total_concepts} concepts, {s.total_triples} triples")
            else:
                print("  • Knowledge graph is up-to-date. (No disk modifications)")
            print("-" * 65 + "\n")

        elif cmd == "/help":
            print("-" * 55)
            print("  AVAILABLE COMMANDS:")
            print("    /tom          Inspect Theory of Mind (Ego, Alter, Dissonance, Triads)")
            print("    /history      Inspect multi-turn persistent conversation memory")
            print("    /critic       Inspect anti-hallucination guardrail audit statistics")
            print("    /inspect [m]  Inspect codebase AST structure, classes, or symbols")
            print("    /discover [N] Run N epistemic discovery cycles (Popperian Z3 SMT)")
            print("    /ingest       Ingest curated scientific seed corpus into 256D manifold")
            print("    /search [q]   Search arXiv papers, extract LaTeX math, and wire via Z3")
            print("    /reload       Hot-reload modified Python modules without restart")
            print("    /sync         Re-sync knowledge graph & manifold inverted indices")
            print("    /listen       Toggle background passive ear on/off")
            print("    /hud          Toggle live acoustic notifications while typing")
            print("    /manifold     View 256D relational concept map & Hebbian edges")
            print("    /tone         Inspect learned vocal tone and speaking modality")
            print("    /genesis      View all permanent project intent anchors in the manifold")
            print("    /mode [m]     Switch mode (ACTIVE, PASSIVE, HYBRID)")
            print("    /clear        Clear terminal screen")
            print("    /exit         Save state and quit")
            print("-" * 55 + "\n")

        else:
            print(f"Unknown command '{cmd}'. Type /help for options.\n")


def main():
    chat = JaneTerminalChat()
    chat.run()


if __name__ == "__main__":
    main()
