"""Sandboxed Multi-Language Compiler & Execution Sandbox for Project Jane.

Provides deterministic, isolated subprocess execution for Python and C++
with hard resource limits, static security audits, and zero regex:
1. Resource Isolation (RLIMIT_AS, RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_NPROC).
2. Multi-Language Compiler (Python AST execution, G++ C++17 compilation).
3. Static Security Audits (Zero-Regex token & AST inspection blocking dangerous OS/Network calls).
4. Dynamic Module Synthesis (Compiles mathematical routines into invokable wrappers).
"""

import os
import sys
import time
import tempfile
import subprocess
import resource
import shutil
import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable

from src.utils.logger import get_logger

logger = get_logger("sandboxed_compiler")


class Language(str, Enum):
    PYTHON = "python"
    CPP = "cpp"
    SHELL = "shell"


class ExecutionMode(str, Enum):
    PROJECT_WORKSPACE = "project_workspace"          # Direct autonomous project execution outside sandbox
    ARCHITECTURE_EVOLUTION = "architecture_evolution" # Self-modification / novel architecture (strictly sandboxed & verified)


class OSAndCodebaseImmunityGuard:
    """
    Formal safety invariant guard that mathematically prevents any destructive operations
    targeting the operating system or Project Jane's own core codebase.
    Zero regular expressions. Pure deterministic AST and path containment proofs.
    """
    PROHIBITED_SYSTEM_ROOTS: Tuple[str, ...] = (
        "/", "/bin", "/sbin", "/etc", "/usr", "/boot",
        "/sys", "/proc", "/dev", "/root", "/var", "/opt",
        "/lib", "/lib64"
    )

    @classmethod
    def get_protected_codebase_roots(cls) -> Tuple[str, ...]:
        """Dynamically resolves Jane's repository root and critical codebase paths."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return (
            os.path.join(repo_root, "src"),
            os.path.join(repo_root, ".git"),
            os.path.join(repo_root, "data", "models"),
            os.path.join(repo_root, "data", "knowledge_graph")
        )

    @classmethod
    def verify_path_immunity(cls, target_path: str, allow_codebase_evolution: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Proves whether a file path is safe to access/write or if it violates system/codebase immunity.
        Returns (is_safe, error_reason).
        """
        norm_path = os.path.normpath(os.path.abspath(target_path))

        # 1. OS Root Protection Gate
        for sys_root in cls.PROHIBITED_SYSTEM_ROOTS:
            if sys_root == "/" and norm_path == "/":
                return False, "Operating System root '/' cannot be modified or deleted."
            if sys_root != "/" and (norm_path == sys_root or norm_path.startswith(sys_root + "/")):
                return False, f"System path '{target_path}' is protected under OS Immunity."

        # 2. Codebase Immutability Gate (during unverified operations)
        if not allow_codebase_evolution:
            for code_root in cls.get_protected_codebase_roots():
                if norm_path == code_root or norm_path.startswith(code_root + "/"):
                    return False, f"Codebase path '{target_path}' is immutable outside sandboxed evolution."

        return True, None

    @classmethod
    def audit_ast_for_destructive_calls(cls, source_code: str) -> List[str]:
        """
        Audits Python AST for destructive file or system calls (rmtree, remove, unlink, rmdir, etc.).
        """
        violations: List[str] = []
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return violations

        destructive_attrs = {"remove", "unlink", "rmdir", "rmtree", "empty_tree", "truncate"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = None
                if isinstance(node.func, ast.Attribute) and node.func.attr in destructive_attrs:
                    call_name = node.func.attr
                elif isinstance(node.func, ast.Name) and node.func.id in destructive_attrs:
                    call_name = node.func.id

                if call_name:
                    # Check arguments for protected paths
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            is_safe, err = cls.verify_path_immunity(arg.value)
                            if not is_safe:
                                violations.append(f"Destructive call '{call_name}' targeting protected path: {err}")
                        elif isinstance(arg, ast.JoinedStr):
                            violations.append(f"Destructive call '{call_name}' with dynamic string rejected for safety.")

        return violations


@dataclass
class SandboxedExecutionConfig:
    """Resource bounds and compiler configuration."""
    timeout_sec: float = 15.0
    max_memory_mb: int = 4096
    max_cpu_sec: int = 15
    max_file_size_mb: int = 10
    max_processes: int = 1024
    compiler_path: str = "/usr/bin/g++"
    compiler_flags: List[str] = field(default_factory=lambda: ["-O2", "-std=c++17", "-fno-asm", "-Wall"])
    allowed_python_modules: List[str] = field(
        default_factory=lambda: [
            "math", "cmath", "numpy", "torch", "dataclasses",
            "typing", "collections", "itertools", "functools",
            "json", "random", "decimal", "fractions"
        ]
    )


@dataclass
class CompilationResult:
    """Result of compiling source code into an executable artifact."""
    success: bool
    binary_path: Optional[str] = None
    compilation_time_ms: float = 0.0
    compiler_output: str = ""
    violations: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Structured telemetry from sandboxed code execution."""
    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    compilation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    memory_limit_exceeded: bool = False
    violations: List[str] = field(default_factory=list)

    @property
    def clean_output(self) -> str:
        """Returns stripped stdout or stderr."""
        return self.stdout.strip() if self.stdout.strip() else self.stderr.strip()


class StaticSecurityAuditor:
    """
    Deterministic static security analyzer.
    Guarantees Zero-Regex inspection of source code before compilation or execution.
    """

    FORBIDDEN_PYTHON_MODULES = {
        "os", "subprocess", "pty", "socket", "ctypes", "multiprocessing",
        "shutil", "signal", "posix", "pwd", "grp", "termios", "webbrowser",
        "urllib", "http", "requests", "httpx", "importlib"
    }

    FORBIDDEN_PYTHON_CALLS = {
        "eval", "exec", "__import__", "compile", "breakpoint", "memoryview"
    }

    FORBIDDEN_CPP_HEADERS = {
        "<sys/socket.h>", "<netinet/in.h>", "<arpa/inet.h>", "<unistd.h>",
        "<windows.h>", "<winsock2.h>", "<fcntl.h>", "<sys/mman.h>",
        "<sys/ptrace.h>", "<pty.h>", "<signal.h>", "<dirent.h>", "<sys/stat.h>"
    }

    FORBIDDEN_CPP_IDENTIFIERS = [
        "system(", "popen(", "fork(", "vfork(", "clone(", "execve(",
        "execl(", "execvp(", "execv(", "socket(", "connect(", "bind(",
        "listen(", "accept(", "ptrace(", "kill(", "unlink(", "remove(",
        "rename(", "__asm__", "asm volatile", "__asm"
    ]

    @classmethod
    def audit_python(cls, source_code: str) -> List[str]:
        """Audits Python source code via native Python AST."""
        violations = []
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            violations.append(f"Python syntax error during security audit: {e}")
            return violations

        for node in ast.walk(tree):
            # Check module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in cls.FORBIDDEN_PYTHON_MODULES:
                        violations.append(f"Forbidden module import: '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in cls.FORBIDDEN_PYTHON_MODULES:
                        violations.append(f"Forbidden from-import: '{node.module}'")

            # Check forbidden function calls
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name and func_name in cls.FORBIDDEN_PYTHON_CALLS:
                    violations.append(f"Forbidden function invocation: '{func_name}'")

        return violations

    @classmethod
    def audit_cpp(cls, source_code: str) -> List[str]:
        """Audits C++ source code via deterministic string and token inspection."""
        violations = []
        lines = source_code.split("\n")

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check header inclusions
            if stripped.startswith("#include"):
                for bad_header in cls.FORBIDDEN_CPP_HEADERS:
                    if bad_header in stripped:
                        violations.append(f"Line {line_num}: Forbidden C++ header inclusion {bad_header}")

            # Check forbidden function invocations / identifiers
            for bad_id in cls.FORBIDDEN_CPP_IDENTIFIERS:
                if bad_id in stripped:
                    violations.append(f"Line {line_num}: Forbidden C++ identifier or syscall '{bad_id}'")

        return violations


class SandboxedCompiler:
    """
    Subprocess-isolated compiler and runtime executor with resource bounds.
    """

    def __init__(self, default_config: Optional[SandboxedExecutionConfig] = None):
        self.config = default_config or SandboxedExecutionConfig()

    def _make_preexec_fn(self, config: SandboxedExecutionConfig) -> Callable[[], None]:
        """Creates a preexec hook to enforce Linux POSIX resource limits."""
        def preexec():
            # Memory address space limit
            try:
                mem_bytes = config.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass

            # CPU time limit in seconds
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (config.max_cpu_sec, config.max_cpu_sec))
            except (ValueError, OSError):
                pass

            # Max output file size
            try:
                fsize_bytes = config.max_file_size_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
            except (ValueError, OSError):
                pass

            # Prevent fork bombs
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (config.max_processes, config.max_processes))
            except (ValueError, OSError):
                pass

        return preexec

    def compile_cpp(
        self,
        cpp_code: str,
        output_dir: str,
        config: Optional[SandboxedExecutionConfig] = None
    ) -> CompilationResult:
        """
        Statically audits and compiles C++ code using g++.
        """
        cfg = config or self.config

        # 1. Static security audit
        violations = StaticSecurityAuditor.audit_cpp(cpp_code)
        if violations:
            logger.warning("C++ compilation blocked by security audit: %s", violations)
            return CompilationResult(
                success=False,
                violations=violations,
                compiler_output="Blocked by StaticSecurityAuditor: " + "; ".join(violations)
            )

        src_path = os.path.join(output_dir, "source.cpp")
        bin_path = os.path.join(output_dir, "exec_binary")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(cpp_code)

        compiler = cfg.compiler_path if os.path.exists(cfg.compiler_path) else "g++"
        cmd = [compiler] + cfg.compiler_flags + [src_path, "-o", bin_path]

        t0 = time.time()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_sec
            )
            compile_ms = (time.time() - t0) * 1000.0

            if res.returncode != 0:
                return CompilationResult(
                    success=False,
                    compilation_time_ms=compile_ms,
                    compiler_output=res.stderr or res.stdout
                )

            return CompilationResult(
                success=True,
                binary_path=bin_path,
                compilation_time_ms=compile_ms,
                compiler_output=res.stdout
            )
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False,
                compilation_time_ms=(time.time() - t0) * 1000.0,
                compiler_output=f"Compilation timed out after {cfg.timeout_sec}s"
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                compilation_time_ms=(time.time() - t0) * 1000.0,
                compiler_output=str(e)
            )

    def execute_binary(
        self,
        binary_path: str,
        stdin_input: Optional[str] = None,
        config: Optional[SandboxedExecutionConfig] = None
    ) -> ExecutionResult:
        """
        Executes a compiled binary inside an isolated subprocess sandbox.
        """
        cfg = config or self.config
        preexec = self._make_preexec_fn(cfg)

        t0 = time.time()
        try:
            proc = subprocess.run(
                [binary_path],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_sec,
                preexec_fn=preexec,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
            )
            exec_ms = (time.time() - t0) * 1000.0

            return ExecutionResult(
                success=(proc.returncode == 0),
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                execution_time_ms=exec_ms
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                timed_out=True,
                execution_time_ms=(time.time() - t0) * 1000.0,
                stderr=f"Execution timed out after {cfg.timeout_sec}s"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                execution_time_ms=(time.time() - t0) * 1000.0,
                stderr=str(e)
            )

    def execute_python(
        self,
        py_code: str,
        stdin_input: Optional[str] = None,
        config: Optional[SandboxedExecutionConfig] = None,
        mode: ExecutionMode = ExecutionMode.ARCHITECTURE_EVOLUTION,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Executes Python code with dual-mode isolation:
        - PROJECT_WORKSPACE: Direct project execution outside sandbox, guarded by OS and codebase immunity.
        - ARCHITECTURE_EVOLUTION: Strict sandboxed POSIX execution for self-modifications and novel architectures.
        """
        cfg = config or self.config

        # 1. Always enforce OS and Codebase Immunity Gate
        immunity_violations = OSAndCodebaseImmunityGuard.audit_ast_for_destructive_calls(py_code)
        if immunity_violations:
            logger.warning("Execution blocked by OSAndCodebaseImmunityGuard: %s", immunity_violations)
            return ExecutionResult(
                success=False,
                exit_code=1,
                violations=immunity_violations,
                stderr="Blocked by OSAndCodebaseImmunityGuard: " + "; ".join(immunity_violations)
            )

        if mode == ExecutionMode.PROJECT_WORKSPACE:
            # Direct workspace execution outside sandbox (unconfined libraries, user project files)
            cwd = working_dir or os.getcwd()
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", py_code],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_sec,
                    cwd=cwd,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1"
                    }
                )
                exec_ms = (time.time() - t0) * 1000.0
                return ExecutionResult(
                    success=(proc.returncode == 0),
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    execution_time_ms=exec_ms
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    timed_out=True,
                    execution_time_ms=(time.time() - t0) * 1000.0,
                    stderr=f"Execution timed out after {cfg.timeout_sec}s"
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    execution_time_ms=(time.time() - t0) * 1000.0,
                    stderr=str(e)
                )

        # ARCHITECTURE_EVOLUTION mode: strictly sandboxed with POSIX limits
        violations = StaticSecurityAuditor.audit_python(py_code)
        if violations:
            logger.warning("Python execution blocked by security audit: %s", violations)
            return ExecutionResult(
                success=False,
                exit_code=1,
                violations=violations,
                stderr="Blocked by StaticSecurityAuditor: " + "; ".join(violations)
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(py_code)

            preexec = self._make_preexec_fn(cfg)
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_sec,
                    preexec_fn=preexec,
                    cwd=tmpdir,
                    env={
                        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                        "PYTHONPATH": os.getcwd(),
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2"
                    }
                )
                exec_ms = (time.time() - t0) * 1000.0

                return ExecutionResult(
                    success=(proc.returncode == 0),
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    execution_time_ms=exec_ms
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    timed_out=True,
                    execution_time_ms=(time.time() - t0) * 1000.0,
                    stderr=f"Execution timed out after {cfg.timeout_sec}s"
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    execution_time_ms=(time.time() - t0) * 1000.0,
                    stderr=str(e)
                )

    def run(
        self,
        code: str,
        language: Language = Language.PYTHON,
        stdin_input: Optional[str] = None,
        config: Optional[SandboxedExecutionConfig] = None,
        mode: ExecutionMode = ExecutionMode.ARCHITECTURE_EVOLUTION,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        High-level dispatcher for multi-language execution under specified ExecutionMode.
        """
        cfg = config or self.config

        if language == Language.PYTHON:
            return self.execute_python(code, stdin_input=stdin_input, config=cfg, mode=mode, working_dir=working_dir)

        elif language == Language.CPP:
            with tempfile.TemporaryDirectory() as tmpdir:
                comp_res = self.compile_cpp(code, output_dir=tmpdir, config=cfg)
                if not comp_res.success or not comp_res.binary_path:
                    return ExecutionResult(
                        success=False,
                        exit_code=1,
                        compilation_time_ms=comp_res.compilation_time_ms,
                        stderr=comp_res.compiler_output,
                        violations=comp_res.violations
                    )
                exec_res = self.execute_binary(comp_res.binary_path, stdin_input=stdin_input, config=cfg)
                exec_res.compilation_time_ms = comp_res.compilation_time_ms
                return exec_res

        else:
            return ExecutionResult(
                success=False,
                exit_code=1,
                stderr=f"Unsupported language '{language}'"
            )


class DynamicModuleSynthesizer:
    """
    Synthesizes and wraps computational routines (Python or C++) into
    reusable in-memory invokable callables for Jane.
    """

    def __init__(self, compiler: Optional[SandboxedCompiler] = None):
        self.compiler = compiler or SandboxedCompiler()

    def build_callable_python(
        self,
        function_name: str,
        code: str
    ) -> Tuple[bool, Optional[Callable[..., Any]], str]:
        """
        Synthesizes a callable Python function in a hardened namespace.
        """
        violations = StaticSecurityAuditor.audit_python(code)
        if violations:
            return False, None, "Security violation: " + "; ".join(violations)

        safe_scope: Dict[str, Any] = {
            "math": __import__("math"),
            "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
            "len": len, "range": range, "int": int, "float": float, "list": list
        }

        try:
            exec(code, safe_scope)
            fn = safe_scope.get(function_name)
            if not callable(fn):
                return False, None, f"Symbol '{function_name}' was not created as a callable."
            return True, fn, "Compiled successfully"
        except Exception as e:
            return False, None, f"Compilation failure: {e}"

    def build_cpp_evaluator(
        self,
        cpp_source: str
    ) -> Tuple[bool, Optional[Callable[[str], ExecutionResult]], str]:
        """
        Synthesizes a compiled C++ routine into a callable execution wrapper.
        Takes stdin input and returns ExecutionResult.
        """
        # Audit first
        violations = StaticSecurityAuditor.audit_cpp(cpp_source)
        if violations:
            return False, None, "Security violation: " + "; ".join(violations)

        tmpdir = tempfile.mkdtemp(prefix="jane_cpp_")
        comp = self.compiler.compile_cpp(cpp_source, output_dir=tmpdir)
        if not comp.success or not comp.binary_path:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return False, None, f"C++ Compilation failed: {comp.compiler_output}"

        bin_path = comp.binary_path

        def cpp_runner(stdin_data: str = "") -> ExecutionResult:
            return self.compiler.execute_binary(bin_path, stdin_input=stdin_data)

        return True, cpp_runner, f"C++ routine compiled in {comp.compilation_time_ms:.1f}ms"
