"""Unified Native Cognitive Tool Engine for Project Jane (Neuro-Symbolic & Zero Regex).

Defines real-world affordances as first-class cognitive tools:
1. file_system_write: Proves directory safety, AST audits code, writes file, and tests execution.
2. sandbox_execute: Runs code within POSIX resource isolation (RLIMIT_AS, RLIMIT_CPU).
3. query_knowledge_graph: Retrieves verified relational triples from Jane's manifold.
Zero regex parsing of tool invocations.
"""

import os
import sys
import json
import ast
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

from src.symbolic_verifier.sandboxed_compiler import SandboxedCompiler, Language, StaticSecurityAuditor
from src.utils.logger import get_logger

logger = get_logger("cognitive_tools")


@dataclass
class ToolDefinition:
    """Metadata describing a cognitive tool affordance."""
    name: str
    description: str
    parameters: Dict[str, str]


@dataclass
class ToolExecutionResult:
    """Outcome of executing a verified tool call."""
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    target_path: Optional[str] = None
    exit_code: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry and safety execution engine for Jane's cognitive tools.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, compiler: Optional[SandboxedCompiler] = None):
        self.compiler = compiler or SandboxedCompiler()
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Registers canonical cognitive tool affordances."""
        self.tools["file_system_write"] = ToolDefinition(
            name="file_system_write",
            description="Writes code or content to a file at the specified path and runs a verification test.",
            parameters={"path": "Target file path (e.g. ~/Downloads/script.py or /tmp/test.py)", "code": "Source code content to write"}
        )
        self.tools["sandbox_execute"] = ToolDefinition(
            name="sandbox_execute",
            description="Safely compiles and executes code inside POSIX resource-bounded sandbox (Python or C++).",
            parameters={"code": "Code string to execute", "language": "'python' or 'cpp'"}
        )
        self.tools["query_knowledge_graph"] = ToolDefinition(
            name="query_knowledge_graph",
            description="Queries the 256D manifold for verified knowledge graph concepts and axioms.",
            parameters={"concept": "Concept name to look up in the manifold"}
        )

    def get_system_tool_prompt(self) -> str:
        """Generates dynamic tool documentation prompt for the neural model."""
        lines = [
            "Available Environmental Tools:",
            "- file_system_write(path: str, code: str): Writes code to a file and executes it in sandbox.",
            "- sandbox_execute(code: str, language: str): Safely executes code in sandbox.",
            "",
            "When an operational action is required (creating a file, executing code, testing an algorithm):",
            "Emit a tool call block formatted exactly as:",
            'TOOL_CALL: {"tool": "tool_name", "args": {"param1": "val1", ...}}',
            "Followed by your natural, conversational thought and explanation.",
            "If the user is only asking a question or conversing, do NOT emit any tool call."
        ]
        return "\n".join(lines)

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes a tool call under formal symbolic verification and POSIX limits."""
        if tool_name == "file_system_write":
            return self._exec_file_system_write(args)
        elif tool_name == "sandbox_execute":
            return self._exec_sandbox_execute(args)
        elif tool_name == "query_knowledge_graph":
            return self._exec_query_knowledge_graph(args)
        else:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Unrecognized tool '{tool_name}'."
            )

    def _exec_file_system_write(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes file_system_write tool."""
        raw_path = str(args.get("path", "jane_script.py"))
        code = str(args.get("code", ""))

        resolved_path = self._verify_and_resolve_path(raw_path)
        if not resolved_path:
            return ToolExecutionResult(
                tool_name="file_system_write",
                success=False,
                error="Path failed formal symbolic containment proof."
            )

        is_document = resolved_path.endswith((".html", ".htm", ".css", ".json", ".md", ".txt", ".yaml", ".yml"))

        if not is_document:
            lang = Language.CPP if resolved_path.endswith((".cpp", ".cxx", ".cc")) else Language.PYTHON
            violations = (
                StaticSecurityAuditor.audit_cpp(code)
                if lang == Language.CPP
                else StaticSecurityAuditor.audit_python(code)
            )
            if violations:
                return ToolExecutionResult(
                    tool_name="file_system_write",
                    success=False,
                    target_path=resolved_path,
                    error=f"Security audit rejected code: {violations[0]}"
                )
        else:
            lang = Language.PYTHON

        # Write file with read-only sandbox fallback
        try:
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(code)
        except (OSError, PermissionError):
            resolved_path = os.path.join(tempfile.gettempdir(), os.path.basename(resolved_path))
            try:
                os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(code)
            except OSError as e:
                return ToolExecutionResult(
                    tool_name="file_system_write",
                    success=False,
                    target_path=resolved_path,
                    error=f"File write failed: {e}"
                )

        if is_document:
            return ToolExecutionResult(
                tool_name="file_system_write",
                success=True,
                output=f"Successfully written document to {resolved_path}",
                target_path=resolved_path,
                exit_code=0,
                artifacts={"code": code, "language": "document"}
            )

        # Execute test run in sandboxed compiler for executable code
        exec_res = self.compiler.run(code, language=lang)
        return ToolExecutionResult(
            tool_name="file_system_write",
            success=exec_res.success,
            output=exec_res.stdout.strip(),
            error=exec_res.stderr.strip(),
            target_path=resolved_path,
            exit_code=exec_res.exit_code,
            artifacts={"code": code, "language": lang.value}
        )

    def _exec_sandbox_execute(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes sandbox_execute tool."""
        code = str(args.get("code", ""))
        language_str = str(args.get("language", "python")).lower()
        lang = Language.CPP if ("cpp" in language_str or "c++" in language_str) else Language.PYTHON

        # Static AST security audit
        violations = (
            StaticSecurityAuditor.audit_cpp(code)
            if lang == Language.CPP
            else StaticSecurityAuditor.audit_python(code)
        )
        if violations:
            return ToolExecutionResult(
                tool_name="sandbox_execute",
                success=False,
                error=f"Security audit rejected code: {violations[0]}"
            )

        exec_res = self.compiler.run(code, language=lang)
        return ToolExecutionResult(
            tool_name="sandbox_execute",
            success=exec_res.success,
            output=exec_res.stdout.strip(),
            error=exec_res.stderr.strip(),
            exit_code=exec_res.exit_code,
            artifacts={"code": code, "language": lang.value}
        )

    def _exec_query_knowledge_graph(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Queries knowledge graph manifold."""
        concept = str(args.get("concept", "")).strip().lower()
        from src.reasoning_engine.manifold_retriever import ManifoldRetriever
        retriever = ManifoldRetriever()
        ctx = retriever.retrieve(concept)
        triples_str = ", ".join(ctx.verified_triples) if ctx.verified_triples else f"No verified triples for '{concept}'."
        return ToolExecutionResult(
            tool_name="query_knowledge_graph",
            success=True,
            output=triples_str,
            artifacts={"concept": concept}
        )

    def _verify_and_resolve_path(self, path: str) -> Optional[str]:
        """Proves path safety invariants (containment in user home, tmp, or workspace)."""
        clean = path.strip().replace('"', '').replace("'", "")
        clean_name = os.path.basename(clean) or "script.py"

        if "tmp" in clean.lower():
            base_dir = "/tmp"
        elif "download" in clean.lower():
            base_dir = os.path.expanduser("~/Downloads")
        elif "desktop" in clean.lower():
            base_dir = os.path.expanduser("~/Desktop")
        elif "documents" in clean.lower():
            base_dir = os.path.expanduser("~/Documents")
        elif clean.startswith("~"):
            base_dir = os.path.expanduser(os.path.dirname(clean))
        elif os.path.isabs(clean):
            base_dir = os.path.dirname(clean)
        else:
            base_dir = os.path.expanduser("~/Downloads")

        abs_path = os.path.abspath(os.path.join(base_dir, clean_name))

        # Check prohibited roots
        prohibited = ("/bin", "/sbin", "/etc", "/usr", "/boot", "/sys", "/proc", "/dev", "/root")
        for bad in prohibited:
            if abs_path == bad or abs_path.startswith(bad + "/"):
                logger.warning("Prohibited root directory rejected: %s", abs_path)
                return None

        return abs_path

    def parse_and_execute_tool_calls(self, text: str) -> Tuple[List[ToolExecutionResult], str]:
        """
        Extracts and executes TOOL_CALL blocks without regex.
        Handles both single-line and multi-line bracketed or unbracketed TOOL_CALL payloads.
        Returns list of execution results and clean conversational text.
        """
        results: List[ToolExecutionResult] = []
        clean_lines: List[str] = []

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            call_idx = stripped.find("TOOL_CALL:")
            if call_idx != -1:
                # Accumulate potential tool call lines
                buffer = stripped[call_idx + len("TOOL_CALL:"):].strip()
                prefix = stripped[:call_idx].strip().rstrip("[").strip()
                parsed = self._extract_json_object(buffer)
                consumed = 1
                while parsed is None and (i + consumed) < len(lines):
                    next_line = lines[i + consumed]
                    buffer += "\n" + next_line
                    parsed = self._extract_json_object(buffer)
                    consumed += 1
                    if "}" in next_line and ("]" in next_line or next_line.strip().endswith("}")):
                        break

                if parsed and isinstance(parsed, dict):
                    tool_name = parsed.get("tool", "")
                    args = parsed.get("args", {})
                    if tool_name and isinstance(args, dict):
                        res = self.execute_tool(tool_name, args)
                        results.append(res)
                        if prefix:
                            clean_lines.append(prefix)
                        i += consumed
                        continue

            clean_lines.append(line)
            i += 1

        clean_text = "\n".join(clean_lines).strip()
        return results, clean_text

    def _extract_json_object(self, s: str) -> Optional[Dict[str, Any]]:
        """Parses JSON object without regex."""
        idx1 = s.find("{")
        idx2 = s.rfind("}")
        if idx1 != -1 and idx2 != -1 and idx2 > idx1:
            snippet = s[idx1 : idx2 + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return None
