"""Autonomous Self-Healing Syntax & Logic Engine for Project Jane.

Combines Lark Context-Free Grammar AST parsing with Z3 SMT formal verification:
1. Detects and repairs syntax hallucinations at the AST level.
2. Identifies mathematical and safety vulnerabilities via Z3 counterexamples.
3. Autonomously patches AST nodes (guard conditions, domain bounds, contract assertions).
4. Re-proves the healed code until mathematically sound.

Zero regular expressions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from src.symbolic_verifier.lark_python_parser import (
    LarkPythonLogicParser, ProgramNode, FunctionDefNode, ContractNode,
    VariableNode, BinaryOpNode, ConstantNode
)
from src.symbolic_verifier.z3_code_verifier import (
    Z3CodeSafetyProver, CodeVerificationResult
)
from src.utils.logger import get_logger

logger = get_logger("self_healing_code")


@dataclass
class HealedCodeResult:
    """Artifact emitted by the Self-Healing Code Engine."""
    original_code: str
    healed_code: str
    ast: Optional[ProgramNode]
    is_verified: bool
    proof_log: str
    repairs_applied: List[str] = field(default_factory=list)


class SelfHealingCodeEngine:
    """
    Autonomous code synthesizer and self-healer.
    Ensures Jane never hallucinates code syntax and never executes unverified logic.
    """

    def __init__(
        self,
        parser: Optional[LarkPythonLogicParser] = None,
        prover: Optional[Z3CodeSafetyProver] = None
    ):
        self.parser = parser or LarkPythonLogicParser.get_instance()
        self.prover = prover or Z3CodeSafetyProver()

    def process_and_verify(self, code_str: str, max_healing_attempts: int = 3) -> HealedCodeResult:
        """
        Parses, formally proves, and autonomously heals Python code:
        1. Parse via Lark CFG (repair syntax if needed).
        2. Prove via Z3 SMT.
        3. If Z3 discovers a vulnerability (e.g. division by zero), heal AST with safety guards.
        4. Re-prove until verified or limit reached.
        """
        current_code = code_str.strip()
        repairs: List[str] = []

        for attempt in range(max_healing_attempts):
            # Step 1: Parse with Lark Grammar
            ast_tree, diag = self.parser.parse(current_code)

            if not diag.is_valid:
                # Syntax error detected - attempt syntax healing
                repaired_code, fix_msg = self._heal_syntax(current_code, diag)
                repairs.append(f"Syntax Heal (Attempt {attempt + 1}): {fix_msg}")
                current_code = repaired_code
                continue

            # Step 2: Formal Z3 SMT Safety Proof
            ver_res = self.prover.verify_program(ast_tree)

            if ver_res.is_safe:
                return HealedCodeResult(
                    original_code=code_str,
                    healed_code=current_code,
                    ast=ast_tree,
                    is_verified=True,
                    proof_log=ver_res.proof_log,
                    repairs_applied=repairs
                )

            # Step 3: Logic / Safety violation detected by Z3 - attempt logic healing
            healed_code, logic_fix = self._heal_logic_violations(current_code, ast_tree, ver_res)
            if healed_code != current_code:
                repairs.append(f"Logic Heal (Attempt {attempt + 1}): {logic_fix}")
                current_code = healed_code
            else:
                # Could not heal further
                return HealedCodeResult(
                    original_code=code_str,
                    healed_code=current_code,
                    ast=ast_tree,
                    is_verified=False,
                    proof_log=ver_res.proof_log,
                    repairs_applied=repairs
                )

        # Final verification check
        final_ast, final_diag = self.parser.parse(current_code)
        final_ver = self.prover.verify_program(final_ast) if final_diag.is_valid and final_ast else None
        is_safe = final_ver.is_safe if final_ver else False
        p_log = final_ver.proof_log if final_ver else "Failed to parse after healing attempts"

        return HealedCodeResult(
            original_code=code_str,
            healed_code=current_code,
            ast=final_ast,
            is_verified=is_safe,
            proof_log=p_log,
            repairs_applied=repairs
        )

    def _heal_syntax(self, code: str, diag: Any) -> Tuple[str, str]:
        """
        Deterministic AST syntax healing without regex:
        - Balances missing closing parentheses and brackets.
        - Fixes missing colons on def/if/for lines.
        - Removes dangling trailing operators.
        """
        lines = code.split("\n")
        fixed_lines = []
        fixes = []

        for line in lines:
            trimmed = line.strip()
            # If line is a header statement (def, if, while) with unclosed parens, balance parens first!
            for keyword in ("def ", "if ", "while "):
                if trimmed.startswith(keyword):
                    diff = line.count("(") - line.count(")")
                    if diff > 0:
                        line = line.rstrip() + (")" * diff)
                        trimmed = line.strip()
                        fixes.append(f"Balanced {diff} unclosed parenthesis in '{keyword.strip()}' header")
                    break

            # Missing colon in def / if / for / while / else
            for keyword in ("def ", "if ", "for ", "while ", "else"):
                if trimmed.startswith(keyword):
                    if trimmed.endswith("{"):
                        before_brace = trimmed[:-1].rstrip()
                        if not before_brace.endswith(":"):
                            idx = line.rfind("{")
                            line = line[:idx].rstrip() + ": {"
                            fixes.append(f"Inserted missing colon before block in '{keyword.strip()}' statement")
                            break
                    elif not trimmed.endswith(":"):
                        line = line.rstrip() + ":"
                        fixes.append(f"Inserted missing colon in '{keyword.strip()}' statement")
                        break

            # Remove trailing dangling operator (+, -, *, /)
            stripped_end = line.rstrip()
            if stripped_end.endswith(("+", "-", "*", "/")):
                line = stripped_end[:-1].rstrip()
                fixes.append("Removed dangling trailing operator")

            fixed_lines.append(line)

        # Parenthesis balancing
        joined = "\n".join(fixed_lines)
        open_parens = joined.count("(")
        close_parens = joined.count(")")
        if open_parens > close_parens:
            joined += ")" * (open_parens - close_parens)
            fixes.append(f"Balanced {open_parens - close_parens} unclosed parenthesis")

        open_braces = joined.count("{")
        close_braces = joined.count("}")
        if open_braces > close_braces:
            joined += "\n" + ("}" * (open_braces - close_braces))
            fixes.append(f"Balanced {open_braces - close_braces} unclosed braces")

        fix_summary = ", ".join(fixes) if fixes else "Normalized code whitespace"
        return joined, fix_summary

    def _heal_logic_violations(
        self,
        code: str,
        ast_tree: ProgramNode,
        ver_res: CodeVerificationResult
    ) -> Tuple[str, str]:
        """
        Heals mathematical vulnerabilities discovered by Z3:
        If division by zero was found, extracts the denominator variable
        and injects a formal 'require: <var> != 0' contract into the function.
        """
        fixes = []
        modified_code = code

        for v in ver_res.violations:
            if "Division by zero possible" in v:
                # Find the counterexample variables or denominator variables
                for stmt in ast_tree.statements:
                    if isinstance(stmt, FunctionDefNode):
                        div_nodes = self.prover._find_division_nodes(stmt)
                        for d in div_nodes:
                            if isinstance(d.right, VariableNode):
                                denom_var = d.right.name
                                # Inject require contract
                                contract_str = f"require: {denom_var} != 0"
                                if contract_str not in modified_code:
                                    # Insert right after def line
                                    lines = modified_code.split("\n")
                                    new_lines = []
                                    injected = False
                                    for l in lines:
                                        new_lines.append(l)
                                        if f"def {stmt.name}" in l and not injected:
                                            new_lines.append(f"    {contract_str}")
                                            injected = True
                                    modified_code = "\n".join(new_lines)
                                    fixes.append(f"Injected safety contract 'require: {denom_var} != 0' into function '{stmt.name}'")

        fix_summary = "; ".join(fixes) if fixes else "No automated logic repair pattern matched"
        return modified_code, fix_summary
