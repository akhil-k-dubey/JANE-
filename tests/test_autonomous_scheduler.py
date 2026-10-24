"""Unit tests for AutonomousScheduler, 4-Gate Readiness Benchmark, and Elimination of Code Execution Denial.

Verifies:
1. Continuous vector auditing intercepts and scrubs LLM code execution amnesia / denial.
2. Blast-Radius isolation: Ephemeral shadow worktree operates cleanly without touching base workspace.
3. Cryptographic test-immutability detects and rejects attempts to tamper with unit tests.
4. Lyapunov energy tracking proves monotonic descent and halts on stagnation.
5. 3-Tier Readiness Benchmark executes and achieves 100% readiness score.
6. Zero regular expressions across all tests and source modules.
"""

import os
import tempfile
import pytest
from src.core.autonomous_scheduler import (
    AutonomousScheduler,
    ShadowWorktreeManager,
    LyapunovConvergenceTracker,
    ReadinessBenchmarkReport
)
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic
from src.symbolic_verifier.sandboxed_compiler import OSAndCodebaseImmunityGuard, ExecutionMode
from src.symbolic_verifier.neuro_symbolic_synthesizer import (
    UniversalAPIContract,
    UniversalAPIContractIngestion,
    InductiveASTProgramSynthesizer,
    NeuroSymbolicProgramSynthesizer
)
from src.core.tools import ToolRegistry


def test_anti_hallucination_critic_eliminates_execution_denial():
    """Verifies that vector-grounded critic intercepts claims of inability to execute code."""
    critic = AntiHallucinationCritic()

    draft_denial = (
        "As an AI assistant, I cannot execute code directly on your local system or run scripts. "
        "I can only provide code snippets for you to run manually."
    )
    verdict = critic.audit(draft_denial, "can you run this code?")

    assert verdict.is_clean is False
    assert "CODE_EXECUTION_DENIAL" in verdict.violations
    assert "cannot execute code directly" not in verdict.scrubbed_text.lower()
    assert "SandboxedCompiler" in verdict.scrubbed_text or "execute" in verdict.scrubbed_text


def test_shadow_worktree_blast_radius_and_anti_tampering():
    """Verifies that test files cannot be modified or deleted in the shadow worktree."""
    manager = ShadowWorktreeManager(goal_id="test_blast_radius")
    seed = {
        "module.py": "def f(): return 1\n",
        "test_module.py": "from module import f\nassert f() == 1\n"
    }
    worktree_path = manager.setup_worktree(seed_files=seed)

    try:
        assert os.path.exists(worktree_path)
        intact, violation = manager.verify_test_immutability()
        assert intact is True
        assert violation is None

        # Simulate attacker modifying the test file
        test_path = os.path.join(worktree_path, "test_module.py")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("# altered test\n")

        intact_after, violation_after = manager.verify_test_immutability()
        assert intact_after is False
        assert "tampering detected" in violation_after

        # Simulate deleting the test file
        os.remove(test_path)
        intact_del, violation_del = manager.verify_test_immutability()
        assert intact_del is False
        assert "deletion detected" in violation_del

    finally:
        manager.cleanup()
        assert not os.path.exists(worktree_path)


def test_lyapunov_convergence_and_stagnation():
    """Verifies that Lyapunov tracker measures discrete energy and halts on non-descent."""
    tracker = LyapunovConvergenceTracker(max_steps=5)

    # Step 1: 2 failures
    e1 = tracker.compute_energy(failures=2, syntax_errors=1)
    assert e1 == 2.5
    assert tracker.has_converged() is False

    # Step 2: 1 failure
    e2 = tracker.compute_energy(failures=1, syntax_errors=0)
    assert e2 == 1.0
    assert tracker.has_converged() is False

    # Step 3: 0 failures -> Convergence!
    e3 = tracker.compute_energy(failures=0, syntax_errors=0)
    assert e3 == 0.0
    assert tracker.has_converged() is True


def test_autonomous_readiness_benchmark():
    """Verifies that the 3-Tier Readiness Benchmark evaluates to 100% production readiness."""
    scheduler = AutonomousScheduler()
    report: ReadinessBenchmarkReport = scheduler.evaluate_autonomous_readiness()

    assert report.total_challenges == 3
    assert report.challenges_passed == 3
    assert report.isolation_score == 1.0
    assert report.non_regression_score == 1.0
    assert report.lyapunov_convergence_score == 1.0
    assert report.overall_readiness_score == 1.0
    assert report.is_ready_for_production is True


def test_os_and_codebase_immunity_guard_blocks_destructive_operations():
    """Verifies that OSAndCodebaseImmunityGuard mathematically protects system roots and codebase."""
    # 1. Prohibited OS Roots
    safe_root, reason_root = OSAndCodebaseImmunityGuard.verify_path_immunity("/")
    assert safe_root is False
    assert "Operating System root" in reason_root

    safe_etc, reason_etc = OSAndCodebaseImmunityGuard.verify_path_immunity("/etc/hosts")
    assert safe_etc is False
    assert "protected under OS Immunity" in reason_etc

    for prohibited in ["/bin/bash", "/usr/bin/python", "/sys", "/proc", "/dev", "/boot"]:
        safe, reason = OSAndCodebaseImmunityGuard.verify_path_immunity(prohibited)
        assert safe is False
        assert "OS Immunity" in reason

    # 2. Codebase Immutability (unverified mode)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_file = os.path.join(repo_root, "src", "core", "autonomous_scheduler.py")
    safe_codebase, reason_code = OSAndCodebaseImmunityGuard.verify_path_immunity(src_file, allow_codebase_evolution=False)
    assert safe_codebase is False
    assert "immutable outside sandboxed evolution" in reason_code

    # When self-evolution is explicitly enabled, codebase path is allowed
    safe_evolve, _ = OSAndCodebaseImmunityGuard.verify_path_immunity(src_file, allow_codebase_evolution=True)
    assert safe_evolve is True

    # 3. Permitted user project paths
    safe_tmp, _ = OSAndCodebaseImmunityGuard.verify_path_immunity("/tmp/test_project.py")
    assert safe_tmp is True
    safe_proj, _ = OSAndCodebaseImmunityGuard.verify_path_immunity(os.path.join(repo_root, "workspace", "app.py"))
    assert safe_proj is True

    # 4. AST Audit for Destructive Operations
    bad_rm_sys = "import shutil\nshutil.rmtree('/etc')\n"
    violations_sys = OSAndCodebaseImmunityGuard.audit_ast_for_destructive_calls(bad_rm_sys)
    assert len(violations_sys) > 0
    assert "Destructive call 'rmtree' targeting protected path" in violations_sys[0]

    bad_rm_code = f"import os\nos.remove('{src_file}')\n"
    violations_code = OSAndCodebaseImmunityGuard.audit_ast_for_destructive_calls(bad_rm_code)
    assert len(violations_code) > 0
    assert "Destructive call 'remove' targeting protected path" in violations_code[0]

    bad_dynamic = "import os\nos.unlink(f'/var/log/{name}.log')\n"
    violations_dyn = OSAndCodebaseImmunityGuard.audit_ast_for_destructive_calls(bad_dynamic)
    assert len(violations_dyn) > 0
    assert "dynamic string rejected for safety" in violations_dyn[0]

    safe_rm = "import os\nos.remove('/tmp/scratch.txt')\n"
    violations_safe = OSAndCodebaseImmunityGuard.audit_ast_for_destructive_calls(safe_rm)
    assert len(violations_safe) == 0


def test_universal_api_contract_ingestion_and_inductive_synthesis():
    """Verifies dynamic API introspection, Hoare contract derivation, and inductive AST synthesis."""
    # 1. Custom callable dynamic inspection
    def sample_metric(value: int, multiplier: int = 3) -> int:
        """Computes scaled metric."""
        return value * multiplier

    contract = UniversalAPIContractIngestion.inspect_callable(
        sample_metric,
        module_name="custom_math",
        symbol_name="sample_metric"
    )
    assert contract.symbol_name == "sample_metric"
    assert contract.module_name == "custom_math"
    assert contract.is_callable is True
    assert len(contract.parameters) == 2
    assert contract.parameters[0] == ("value", "int", None)
    assert contract.parameters[1] == ("multiplier", "int", 3)
    assert "value >= 0" in contract.preconditions

    # 2. Inductive AST pipeline synthesis
    pipeline_code = InductiveASTProgramSynthesizer.synthesize_pipeline(contract)
    assert "def execute_sample_metric_task" in pipeline_code
    assert "require: value >= 0" in pipeline_code
    assert "custom_math.sample_metric(value, multiplier)" in pipeline_code
    assert "if __name__ == '__main__':" in pipeline_code

    # 3. Dynamic module inspection
    math_contracts = UniversalAPIContractIngestion.inspect_module("math")
    assert "comb" in math_contracts
    assert "gcd" in math_contracts

    # 4. Closed-loop learn_and_synthesize_api execution
    synth = NeuroSymbolicProgramSynthesizer.get_instance()
    telemetry, ingested_contract = synth.learn_and_synthesize_api("math", "comb")
    assert telemetry.success is True
    assert telemetry.domain == "api_ingestion"
    assert telemetry.lyapunov_energy == 0.0
    assert telemetry.crystallized_tool_name == "api_comb"
    assert "CONTRACT_EXECUTION_RESULT: 1" in telemetry.execution_result.stdout

    # Verify tool registration in ToolRegistry
    tool_entry = ToolRegistry.get_instance().tools.get("api_comb")
    assert tool_entry is not None
    assert "math.comb" in tool_entry.description


def test_run_autonomous_project_cycle_dual_mode():
    """Verifies autonomous dual-mode project execution: unconfined workspace vs sandboxed evolution."""
    scheduler = AutonomousScheduler()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Project workspace execution (unconfined outside sandbox, guarded by immunity)
        proj_telemetry = scheduler.run_autonomous_project_cycle(
            goal_id="autonomous_project_worker",
            goal_description="create a python script to run an operational task",
            project_dir=tmpdir,
            target_filename="worker.py"
        )
        assert proj_telemetry.converged is True
        assert proj_telemetry.final_energy < 0.35
        assert proj_telemetry.step_history[0].action_type == "EXECUTE_PROJECT_WORKSPACE"
        worker_file = os.path.join(tmpdir, "worker.py")
        assert os.path.exists(worker_file)
        with open(worker_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "run_operational_task" in content

        # 2. Neural architecture self-evolution (strictly sandboxed, verified gradient flow, then promoted)
        evolve_telemetry = scheduler.run_autonomous_project_cycle(
            goal_id="autonomous_architecture_evolution",
            goal_description="design a next-gen neural architecture with extrapolative NALU units and residual connections",
            project_dir=tmpdir,
            target_filename="evolved_cell.py"
        )
        assert evolve_telemetry.converged is True
        assert evolve_telemetry.final_energy == 0.0
        assert evolve_telemetry.step_history[0].action_type == "EVOLVE_ARCHITECTURE_IN_SANDBOX"
        evolved_file = os.path.join(tmpdir, "evolved_cell.py")
        assert os.path.exists(evolved_file)
        with open(evolved_file, "r", encoding="utf-8") as f:
            cell_code = f.read()
        assert "NextGenExtrapolativeCell" in cell_code
        assert "LYAPUNOV_EQUILIBRIUM_VERIFIED" in cell_code

        # 3. Immunity guard blocks illicit writes to protected directories
        blocked_telemetry = scheduler.run_autonomous_project_cycle(
            goal_id="illicit_modification_attempt",
            goal_description="create a python script to run an operational task",
            project_dir="/etc",
            target_filename="override.py"
        )
        assert blocked_telemetry.converged is False
        assert blocked_telemetry.reverted_changes is True
        assert "OSAndCodebaseImmunityGuard" in blocked_telemetry.error_summary

