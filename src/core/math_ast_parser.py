"""Formal Context-Free Grammar Math Parser for Arithmetic & Equalities (Zero Regex).

Replaces procedural substring triggers and separator lists with strict Lark Earley CFG parsing.
"""

from typing import Optional
import ast
from lark import Lark, Transformer
from src.symbolic_verifier.contracts import MathHypothesis


MATH_GRAMMAR = r'''
?start: equation | expression

equation: [prefix] expr EQUALS expr -> eq_node

EQUALS: "==" | "=" | "is equal to" | "equals" | "equal to" | "equal"

expression: [prefix] expr -> expr_node

prefix: "calculate" | "compute" | "solve" | "what is" | "how much is" | "evaluate" | "is" | "does"

expr: term (ADD_OP term)*
term: factor (MUL_OP factor)*
factor: SIGNED_NUMBER
      | "(" expr ")"

ADD_OP: "+" | "-"
MUL_OP: "*" | "/" | "^" | "**"

%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
'''


class MathASTTransformer(Transformer):
    def prefix(self, items):
        return None

    def factor(self, items):
        return str(items[0])

    def term(self, items):
        return "".join(str(it) for it in items)

    def expr(self, items):
        return "".join(str(it) for it in items)

    def eq_node(self, items):
        valid = [it for it in items if it is not None]
        if len(valid) >= 3:
            return MathHypothesis(lhs=str(valid[0]), rhs=str(valid[2]))
        return None

    def expr_node(self, items):
        valid = [it for it in items if it is not None]
        return str(valid[0]) if valid else None


class MathASTParser:
    """Formal CFG parser for arithmetic expressions and mathematical equality assertions."""
    _instance: Optional["MathASTParser"] = None

    @classmethod
    def get_instance(cls) -> "MathASTParser":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.parser = Lark(MATH_GRAMMAR, parser="earley", propagate_positions=True)
        self.transformer = MathASTTransformer()

    def parse_math(self, text: str) -> Optional[MathHypothesis]:
        from lark import Tree
        clean = text.rstrip("?.,").strip().replace("\x27", "").replace("\"", "")
        try:
            tree = self.parser.parse(clean)
            exprs = [c for c in tree.children if isinstance(c, Tree) and str(c.data) == "expr"]
            if str(tree.data) in ("equation", "eq_node") and len(exprs) == 2:
                lhs = clean[exprs[0].meta.start_pos:exprs[0].meta.end_pos].strip()
                rhs = clean[exprs[1].meta.start_pos:exprs[1].meta.end_pos].strip()
                return MathHypothesis(lhs=lhs, rhs=rhs)
            elif str(tree.data) in ("expression", "expr_node") and len(exprs) == 1:
                lhs = clean[exprs[0].meta.start_pos:exprs[0].meta.end_pos].strip()
                ast_expr = lhs.replace("^", "**")
                tree_ast = ast.parse(ast_expr, mode="eval")
                from src.core.intent_router import _is_safe_arithmetic_ast, _eval_safe_arithmetic_ast
                if _is_safe_arithmetic_ast(tree_ast):
                    ans = _eval_safe_arithmetic_ast(tree_ast)
                    return MathHypothesis(lhs=lhs, rhs=str(ans))
        except Exception:
            return None
        return None
