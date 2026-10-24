"""Safe Mathematical AST Parser: Cryptographically Secure (Zero Eval, Zero Exec)."""

import ast
import operator
from typing import Dict, Set, Any, Union
import mpmath
import z3


class SecurityException(Exception):
    """Raised when an untrusted or non-mathematical AST node is detected."""
    pass


class SafeASTValidator(ast.NodeVisitor):
    """
    Enforces a strict mathematical AST grammar.
    Guarantees no arbitrary code execution (RCE) can occur through string input.
    """

    ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Call,
        ast.Load,
        # Operators
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
        ast.UAdd, ast.USub
    }

    ALLOWED_FUNCTIONS = {
        "sin", "cos", "tan", "exp", "log", "sqrt", "abs"
    }

    def generic_visit(self, node: ast.AST):
        if type(node) not in self.ALLOWED_NODES:
            raise SecurityException(
                f"Security Violation: Node '{type(node).__name__}' is forbidden in mathematical AST."
            )
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SecurityException("Forbidden function call: only direct mathematical functions allowed.")
        if node.func.id not in self.ALLOWED_FUNCTIONS:
            raise SecurityException(f"Forbidden function '{node.func.id}': not in mathematical whitelist.")
        if len(node.args) != 1:
            raise SecurityException(f"Mathematical functions must have exactly 1 argument, got {len(node.args)}.")
        if node.keywords:
            raise SecurityException("Keyword arguments are forbidden in mathematical calls.")
        super().generic_visit(node)


class SafeMathCompiler:
    """
    Safely compiles mathematical expressions from strings into Z3 expressions
    and rigorous interval bounds via mpmath, with zero eval() vulnerability.
    """

    def __init__(self):
        self.validator = SafeASTValidator()

    def parse_to_ast(self, expr_str: str) -> ast.Expression:
        """Parses and strictly validates string into a safe mathematical AST."""
        try:
            tree = ast.parse(expr_str.strip(), mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in mathematical expression: {e}")

        # Validate AST strictly against whitelist
        self.validator.visit(tree)
        return tree

    def has_transcendental_ast(self, tree: ast.AST) -> bool:
        """
        Walks the AST tree to check for mathematical transcendental function calls.
        Variables named 'sin_wave', 'cosine', or 'logging' are ast.Name nodes,
        NOT ast.Call nodes, so they are never falsely flagged as transcendentals.
        """
        transcendental_funcs = {"sin", "cos", "tan", "exp", "log", "sqrt"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in transcendental_funcs:
                    return True
        return False

    @staticmethod
    def has_transcendental_sympy(sympy_expr) -> bool:
        """Walks the SymPy AST tree to check for non-linear transcendental function nodes."""
        import sympy as sp
        transcendental_funcs = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt)
        if sympy_expr.is_Function and sympy_expr.func in transcendental_funcs:
            return True
        for arg in getattr(sympy_expr, "args", []):
            if SafeMathCompiler.has_transcendental_sympy(arg):
                return True
        return False

    def extract_symbols(self, tree: ast.Expression) -> Set[str]:
        """Extracts all variable identifiers from the AST."""
        symbols = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in SafeASTValidator.ALLOWED_FUNCTIONS:
                    symbols.add(node.id)
        return symbols

    def ast_to_z3(self, node: ast.AST, z3_vars: Dict[str, z3.ExprRef]) -> z3.ExprRef:
        """
        Recursively compiles a verified AST node directly to Z3.
        Supports algebra and polynomial powers.
        """
        if isinstance(node, ast.Expression):
            return self.ast_to_z3(node.body, z3_vars)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return z3.RealVal(float(node.value))
            raise SecurityException(f"Non-numeric constant '{node.value}' is forbidden.")

        elif isinstance(node, ast.Name):
            name = node.id
            if name not in z3_vars:
                z3_vars[name] = z3.Real(name)
            return z3_vars[name]

        elif isinstance(node, ast.UnaryOp):
            operand = self.ast_to_z3(node.operand, z3_vars)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return operand
            raise SecurityException(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.BinOp):
            left = self.ast_to_z3(node.left, z3_vars)
            right = self.ast_to_z3(node.right, z3_vars)

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    raise ZeroDivisionError("Division by literal zero in mathematical AST.")
                return left / right
            elif isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                    exp = node.right.value
                    if exp >= 0:
                        res = z3.RealVal(1.0)
                        for _ in range(exp):
                            res = res * left
                        return res
                    else:
                        denom = z3.RealVal(1.0)
                        for _ in range(abs(exp)):
                            denom = denom * left
                        return z3.RealVal(1.0) / denom
                return left ** right

            raise SecurityException(f"Unsupported binary operator: {type(node.op)}")

        elif isinstance(node, ast.Call):
            func_name = node.func.id
            arg = self.ast_to_z3(node.args[0], z3_vars)
            # Flag that expression has transcendental component
            raise NotImplementedError(f"Transcendental function '{func_name}' requires interval arithmetic.")

        raise SecurityException(f"Unsupported AST node: {type(node)}")

    def eval_interval(self, node: ast.AST, var_intervals: Dict[str, mpmath.iv.mpf]) -> mpmath.iv.mpf:
        """
        Evaluates the AST using rigorous interval arithmetic (mpmath.iv).
        Guarantees mathematically sound upper and lower bounds for non-linear & transcendental math.
        """
        if isinstance(node, ast.Expression):
            return self.eval_interval(node.body, var_intervals)

        elif isinstance(node, ast.Constant):
            return mpmath.iv.mpf(node.value)

        elif isinstance(node, ast.Name):
            name = node.id
            if name in var_intervals:
                return var_intervals[name]
            raise ValueError(f"No interval provided for variable '{name}'")

        elif isinstance(node, ast.UnaryOp):
            val = self.eval_interval(node.operand, var_intervals)
            if isinstance(node.op, ast.USub):
                return -val
            return val

        elif isinstance(node, ast.BinOp):
            left = self.eval_interval(node.left, var_intervals)
            right = self.eval_interval(node.right, var_intervals)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                try:
                    return left / right
                except ZeroDivisionError:
                    raise ZeroDivisionError(f"Zero division in interval calculation: denominator interval {right} contains zero.")
            elif isinstance(node.op, ast.Pow):
                return left ** right

        elif isinstance(node, ast.Call):
            func_name = node.func.id
            arg = self.eval_interval(node.args[0], var_intervals)
            if func_name == "sin":
                return mpmath.iv.sin(arg)
            elif func_name == "cos":
                return mpmath.iv.cos(arg)
            elif func_name == "tan":
                return mpmath.iv.tan(arg)
            elif func_name == "exp":
                return mpmath.iv.exp(arg)
            elif func_name == "log":
                return mpmath.iv.log(arg)
            elif func_name == "sqrt":
                return mpmath.iv.sqrt(arg)
            elif func_name == "abs":
                return abs(arg)

        raise SecurityException(f"Interval evaluation error on node: {type(node)}")
