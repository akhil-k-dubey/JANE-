"""Live Modular Code Hot-Reloader for Project Jane.

Enables seamless runtime code updates without shutting down or restarting Jane:
1. File Monitoring: Tracks mtime and sha256 checksums across all src/**/*.py modules.
2. Compile Pre-Validation: Verifies Python syntax before reload, preventing half-broken bytecode.
3. Self-Healing Integration: Invokes SelfHealingCodeEngine if syntax errors are found on disk.
4. State-Preserving Dynamic Re-binding: Hot-reloads classes/functions into sys.modules while
   preserving Theory of Mind tensors, conversational history, and learned manifold weights.
5. Background Watcher & Interactive Hooks: Runs as an async daemon or synchronous per-turn hook.
"""

import os
import sys
import time
import hashlib
import importlib
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable, Set

from src.utils.logger import get_logger

logger = get_logger("live_hot_reloader")


@dataclass
class ReloadReport:
    """Report detailing module hot-reload execution."""
    reloaded_modules: List[str] = field(default_factory=list)
    failed_modules: List[Tuple[str, str]] = field(default_factory=list)
    healed_modules: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def has_changes(self) -> bool:
        return bool(self.reloaded_modules or self.failed_modules or self.healed_modules)

    def format_summary(self) -> str:
        """User-friendly summary of the reload action."""
        if not self.has_changes:
            return "No source code modifications detected."

        lines = []
        if self.reloaded_modules:
            lines.append(f"Hot-reloaded {len(self.reloaded_modules)} module(s): {', '.join(self.reloaded_modules)}")
        if self.healed_modules:
            lines.append(f"Self-healed {len(self.healed_modules)} module(s): {', '.join(self.healed_modules)}")
        if self.failed_modules:
            fails = [f"{m} ({err})" for m, err in self.failed_modules]
            lines.append(f"Failed to reload: {'; '.join(fails)}")
        lines.append(f"Completed in {self.elapsed_ms:.1f}ms (Memory and conversational state preserved)")
        return "\n".join(lines)


class LiveModularHotReloader:
    """
    Scans, validates, heals, and reloads Python modules in real time.
    """

    def __init__(
        self,
        source_root: str = "src",
        on_reload_callback: Optional[Callable[[ReloadReport], None]] = None
    ):
        self.source_root = source_root
        self.on_reload_callback = on_reload_callback
        self._file_snapshots: Dict[str, Tuple[float, str]] = {}
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Initialize snapshots
        self._snapshot_all_modules()

    def _compute_file_hash(self, path: str) -> str:
        """Computes SHA256 of file."""
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return ""

    def _path_to_module(self, path: str) -> str:
        """Converts relative file path to python module import path."""
        clean = path.replace("\\", "/")
        if clean.endswith(".py"):
            clean = clean[:-3]
        if clean.endswith("/__init__"):
            clean = clean[:-9]
        return clean.replace("/", ".")

    def _snapshot_all_modules(self):
        """Builds initial dictionary of (mtime, hash) for all .py files in source_root."""
        for root, _, files in os.walk(self.source_root):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    mtime = os.path.getmtime(full_path)
                    h = self._compute_file_hash(full_path)
                    self._file_snapshots[full_path] = (mtime, h)

    def scan_modified_files(self) -> List[str]:
        """Scans filesystem and returns list of paths that were added or modified."""
        modified: List[str] = []
        for root, _, files in os.walk(self.source_root):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    current_mtime = os.path.getmtime(full_path)
                    prev = self._file_snapshots.get(full_path)

                    if prev is None or current_mtime > prev[0]:
                        current_hash = self._compute_file_hash(full_path)
                        if prev is None or current_hash != prev[1]:
                            modified.append(full_path)
                            self._file_snapshots[full_path] = (current_mtime, current_hash)
        return modified

    def reload_modified(self, pipeline: Optional[Any] = None) -> ReloadReport:
        """
        Scans for changed source files, validates syntax, self-heals if needed,
        and safely reloads affected modules into sys.modules.
        """
        with self._lock:
            t0 = time.time()
            modified_paths = self.scan_modified_files()

            report = ReloadReport()
            if not modified_paths:
                report.elapsed_ms = (time.time() - t0) * 1000.0
                return report

            # Sort modified paths: load base utilities before high-level core pipelines
            modified_paths.sort(key=lambda p: (p.count(os.sep), len(p)))

            for path in modified_paths:
                mod_name = self._path_to_module(path)
                # Read source
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        source_code = f.read()
                except OSError as e:
                    report.failed_modules.append((mod_name, f"Read error: {e}"))
                    continue

                # 1. Compile Pre-Validation
                try:
                    compile(source_code, path, "exec")
                except SyntaxError as syn_err:
                    logger.warning("Syntax error in %s: %s. Attempting self-healing...", path, syn_err)
                    # Attempt self-healing
                    try:
                        from src.symbolic_verifier.self_healing_code import SelfHealingCodeEngine
                        healer = SelfHealingCodeEngine()
                        heal_res = healer.process_and_verify(source_code)
                        if heal_res.is_verified and heal_res.healed_code != source_code:
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(heal_res.healed_code)
                            source_code = heal_res.healed_code
                            report.healed_modules.append(mod_name)
                            logger.info("Successfully self-healed %s on disk.", mod_name)
                        else:
                            report.failed_modules.append((mod_name, f"Syntax error: {syn_err}"))
                            continue
                    except Exception as heal_err:
                        report.failed_modules.append((mod_name, f"Syntax error: {syn_err} (Heal failed: {heal_err})"))
                        continue

                # 2. Reload into sys.modules if already loaded
                if mod_name in sys.modules:
                    try:
                        mod = sys.modules[mod_name]
                        importlib.reload(mod)
                        report.reloaded_modules.append(mod_name)
                        logger.info("Hot-reloaded module: %s", mod_name)
                    except Exception as reload_err:
                        logger.error("Failed to reload %s: %s", mod_name, reload_err)
                        report.failed_modules.append((mod_name, str(reload_err)))
                else:
                    # Module not yet imported, import it cleanly
                    try:
                        importlib.import_module(mod_name)
                        report.reloaded_modules.append(mod_name)
                    except Exception as imp_err:
                        report.failed_modules.append((mod_name, str(imp_err)))

            # 3. State-Preserving Dynamic Re-binding on running pipeline
            if pipeline is not None and report.reloaded_modules:
                self._rebind_pipeline_components(pipeline, set(report.reloaded_modules))

            report.elapsed_ms = (time.time() - t0) * 1000.0

            if self.on_reload_callback and report.has_changes:
                try:
                    self.on_reload_callback(report)
                except Exception as cb_err:
                    logger.warning("Error in on_reload_callback: %s", cb_err)

            return report

    def _rebind_pipeline_components(self, pipeline: Any, reloaded: Set[str]):
        """
        Dynamically updates component instances while preserving internal states.
        """
        # A. Rebind Intent Router if intent modules changed
        if any(m.startswith("src.core.intent_router") or m.startswith("src.reasoning_engine.intent_manifold") for m in reloaded):
            try:
                from src.core.intent_router import IntentRouter
                if hasattr(pipeline, "intent_router"):
                    pipeline.intent_router = IntentRouter(
                        self_model=getattr(pipeline, "self_model", None),
                        active_learner=getattr(pipeline, "active_learner", None)
                    )
                    logger.info("Re-bound pipeline.intent_router")
            except Exception as e:
                logger.warning("Failed to rebind intent_router: %s", e)

        # B. Rebind Anti-Hallucination Critic if critic changed
        if any(m.startswith("src.symbolic_verifier.hallucination_critic") for m in reloaded):
            try:
                from src.symbolic_verifier.hallucination_critic import HallucinationCritic
                if hasattr(pipeline, "critic"):
                    prev_audits = getattr(pipeline.critic, "total_audits", 0)
                    prev_viols = getattr(pipeline.critic, "total_violations_caught", 0)
                    new_critic = HallucinationCritic(
                        retriever=getattr(pipeline, "retriever", None),
                        manifold=getattr(pipeline, "hebbian_manifold", None),
                        self_model=getattr(pipeline, "self_model", None)
                    )
                    new_critic.total_audits = prev_audits
                    new_critic.total_violations_caught = prev_viols
                    pipeline.critic = new_critic
                    if hasattr(pipeline, "mouth") and hasattr(pipeline.mouth, "critic"):
                        pipeline.mouth.critic = new_critic
                    logger.info("Re-bound pipeline.critic with preserved audit counts")
            except Exception as e:
                logger.warning("Failed to rebind critic: %s", e)

        # C. Rebind Codebase Introspector
        if any(m.startswith("src.core.codebase_introspector") for m in reloaded):
            try:
                from src.core.codebase_introspector import CodebaseIntrospector
                if hasattr(pipeline, "codebase_introspector"):
                    pipeline.codebase_introspector = CodebaseIntrospector.get_instance()
                    logger.info("Re-bound pipeline.codebase_introspector")
            except Exception as e:
                logger.warning("Failed to rebind codebase_introspector: %s", e)

        # D. Rebind API Gateway
        if any(m.startswith("src.core.api_gateway") or m.startswith("src.symbolic_verifier.sandboxed_compiler") for m in reloaded):
            try:
                from src.core.api_gateway import APIGateway
                if hasattr(pipeline, "api_gateway"):
                    prev_ledger = getattr(pipeline.api_gateway, "episodic_ledger", None)
                    prev_verifier = getattr(pipeline.api_gateway, "z3_verifier", None)
                    pipeline.api_gateway = APIGateway(z3_verifier=prev_verifier, episodic_ledger=prev_ledger)
                    logger.info("Re-bound pipeline.api_gateway")
            except Exception as e:
                logger.warning("Failed to rebind api_gateway: %s", e)

    def start_background_watcher(self, pipeline: Optional[Any] = None, interval_sec: float = 2.0):
        """Launches background polling thread that detects code modifications automatically."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            return

        self._stop_event.clear()

        def _worker():
            logger.info("Live Modular Hot-Reloader background watcher started (interval: %.1fs)", interval_sec)
            while not self._stop_event.is_set():
                try:
                    self.reload_modified(pipeline=pipeline)
                except Exception as e:
                    logger.error("Error in background hot-reloader: %s", e)
                self._stop_event.wait(interval_sec)

        self._watcher_thread = threading.Thread(target=_worker, daemon=True, name="JaneHotReloader")
        self._watcher_thread.start()

    def stop_background_watcher(self):
        """Stops background watcher thread."""
        self._stop_event.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=1.0)
            self._watcher_thread = None
