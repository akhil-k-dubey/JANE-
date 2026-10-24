"""Unit and Integration Tests for Lark AST Parsing, Z3 Formal Logic Proofs, and Self-Healing Code.

Verifies:
1. Formal Context-Free Grammar (Lark) parsing of Python logic and mathematical contracts.
2. Prevention of syntax hallucinations via AST structure.
3. Deterministic Z3 SMT mathematical proofs (division by zero, contract satisfaction, security audit).
4. Autonomous self-healing of syntax and logic violations.
5. APIGateway execution gating.
"""

import pytest
from src.symbolic_verifier.lark_python_parser import (
    LarkPythonLogicParser, ProgramNode, FunctionDefNode, ConstantNode
)
from src.symbolic_verifier.z3_code_verifier import (
    Z3CodeSafetyProver, CodeVerificationResult
)
from src.symbolic_verifier.self_healing_code import (
    SelfHealingCodeEngine, HealedCodeResult
)
from src.core.api_gateway import APIGateway, ActionASTNode
from src.symbolic_verifier.z3_interface import Z3LogicVerifier


def test_lark_python_parser_valid_syntax():
    """Verify Lark CFG successfully parses valid Python functions and arithmetic."""
    parser = LarkPythonLogicParser.get_instance()
    code = "def compute_power(voltage, current): return voltage * current"
    ast_tree, diag = parser.parse(code)

    assert diag.is_valid is True
    assert ast_tree is not None
    assert len(ast_tree.statements) == 1
    fdef = ast_tree.statements[0]
    assert isinstance(fdef, FunctionDefNode)
    assert fdef.name == "compute_power"
    assert fdef.params == ["voltage", "current"]


def test_lark_python_parser_formal_contracts():
    """Verify Lark CFG parses pre/post contracts (require, ensure)."""
    parser = LarkPythonLogicParser.get_instance()
    code = """
    def kinetic_energy(m, v): {
        require: m > 0
        require: v >= 0
        ensure: result >= 0
        result = 0.5 * m * v ** 2
        return result
    }
    """
    ast_tree, diag = parser.parse(code)
    assert diag.is_valid is True
    assert ast_tree is not None
    fdef = ast_tree.statements[0]
    assert len(fdef.contracts) == 3
    assert fdef.contracts[0].contract_type == "require"
    assert fdef.contracts[2].contract_type == "ensure"


def test_lark_python_parser_catches_hallucinations():
    """Verify Lark CFG catches malformed / hallucinated syntax deterministically."""
    parser = LarkPythonLogicParser.get_instance()
    bad_code = "def invalid_syntax(a b c: return a +"
    ast_tree, diag = parser.parse(bad_code)

    assert diag.is_valid is False
    assert ast_tree is None
    assert "Unexpected" in diag.error_message or "Grammar" in diag.error_message


def test_z3_prover_division_by_zero_counterexample():
    """Verify Z3 finds counterexample when division is unguarded."""
    parser = LarkPythonLogicParser.get_instance()
    prover = Z3CodeSafetyProver()

    code = "def calc_ratio(x, y): return x / y"
    ast_tree, diag = parser.parse(code)
    assert diag.is_valid is True

    res = prover.verify_program(ast_tree)
    assert res.is_safe is False
    assert any("Division by zero possible" in v for v in res.violations)
    assert len(res.counterexamples) > 0


def test_z3_prover_division_by_zero_guarded():
    """Verify Z3 mathematically proves division is safe when guarded by require contract."""
    parser = LarkPythonLogicParser.get_instance()
    prover = Z3CodeSafetyProver()

    code = """
    def calc_ratio_safe(x, y): {
        require: y != 0
        return x / y
    }
    """
    ast_tree, diag = parser.parse(code)
    assert diag.is_valid is True

    res = prover.verify_program(ast_tree)
    assert res.is_safe is True
    assert "PASSED" in res.proof_log


def test_z3_prover_contract_satisfaction():
    """Verify Z3 proves Pre => Post (contract guarantee)."""
    parser = LarkPythonLogicParser.get_instance()
    prover = Z3CodeSafetyProver()

    code = """
    def square_positive(x): {
        require: x > 0
        ensure: result > 0
        result = x * x
        return result
    }
    """
    ast_tree, _ = parser.parse(code)
    res = prover.verify_program(ast_tree)
    assert res.is_safe is True
    assert any("holds unconditionally" in inv for inv in res.proven_invariants)


def test_z3_prover_rejects_security_violations():
    """Verify static security audit rejects dangerous calls (eval, exec)."""
    parser = LarkPythonLogicParser.get_instance()
    prover = Z3CodeSafetyProver()

    code = "def dangerous(x): return eval(x)"
    ast_tree, _ = parser.parse(code)
    res = prover.verify_program(ast_tree)
    assert res.is_safe is False
    assert any("Forbidden function call: 'eval'" in v for v in res.violations)


def test_self_healing_code_engine_syntax():
    """Verify self-healing engine repairs missing colons and unclosed parens."""
    engine = SelfHealingCodeEngine()
    buggy_code = "def area_circle(radius\n    return 3.14159 * radius ** 2"

    healed = engine.process_and_verify(buggy_code)
    assert healed.is_verified is True
    assert "def area_circle(radius):" in healed.healed_code
    assert len(healed.repairs_applied) > 0


def test_self_healing_code_engine_logic():
    """Verify self-healing engine injects safety contracts to resolve division vulnerabilities."""
    engine = SelfHealingCodeEngine()
    vulnerable_code = "def compute_density(mass, volume): {\n    return mass / volume\n}"

    healed = engine.process_and_verify(vulnerable_code)
    assert healed.is_verified is True
    assert "require: volume != 0" in healed.healed_code
    assert "Z3 Proof PASSED" in healed.proof_log


def test_api_gateway_gated_execution():
    """Verify APIGateway executes proven safe code and blocks malicious code."""
    verifier = Z3LogicVerifier()
    gateway = APIGateway(z3_verifier=verifier)

    # 1. Safe verified execution
    safe_node = ActionASTNode(
        action_type="CODE_EXEC",
        payload="def add_nums(a, b): return a + b\nresult = add_nums(15, 25)"
    )
    res_safe = gateway.execute_verified_action(safe_node)
    assert res_safe.success is True
    assert res_safe.output["local_scope"]["result"] == 40

    # 2. Blocked malicious execution
    hack_node = ActionASTNode(
        action_type="CODE_EXEC",
        payload="import os; os.system('rm -rf /')"
    )
    res_hack = gateway.execute_verified_action(hack_node)
    assert res_hack.success is False
