"""Formal Logic and AST Parsing via Lark Context-Free Grammar for Project Jane.

Eliminates syntax hallucinations by parsing Python code and formal contracts into
typed Abstract Syntax Tree (AST) nodes instead of treating code as arbitrary text.
Provides deterministic Earley parsing with exact token-level diagnostics.
Completely free of regular expressions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from lark import Lark, Transformer, Tree, Token
from lark.exceptions import LarkError, UnexpectedInput, UnexpectedToken

from src.utils.logger import get_logger

logger = get_logger("lark_python_parser")


# =====================================================================
# Structured Neuro-Symbolic AST Node Definitions
# =====================================================================

class ASTNode:
    """Base class for all neuro-symbolic AST nodes."""
    pass


@dataclass
class ConstantNode(ASTNode):
    value: Any
    type_name: str = "number"


@dataclass
class VariableNode(ASTNode):
    name: str


@dataclass
class UnaryOpNode(ASTNode):
    op: str
    operand: ASTNode


@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode


@dataclass
class CallNode(ASTNode):
    func: str
    args: List[ASTNode] = field(default_factory=list)


@dataclass
class SubscriptNode(ASTNode):
    value: ASTNode
    index: ASTNode


@dataclass
class AssignNode(ASTNode):
    target: str
    value: ASTNode
    op: str = "="


@dataclass
class AssertNode(ASTNode):
    condition: ASTNode
    message: Optional[str] = None


@dataclass
class ContractNode(ASTNode):
    contract_type: str  # "require", "ensure", "invariant"
    condition: ASTNode
    description: Optional[str] = None


@dataclass
class ReturnNode(ASTNode):
    value: Optional[ASTNode] = None


@dataclass
class IfNode(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
    orelse: List[ASTNode] = field(default_factory=list)


@dataclass
class ForNode(ASTNode):
    target: str
    iterable: ASTNode
    body: List[ASTNode]


@dataclass
class WhileNode(ASTNode):
    condition: ASTNode
    body: List[ASTNode]


@dataclass
class FunctionDefNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    contracts: List[ContractNode] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class ProgramNode(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class SyntaxErrorDiagnosis:
    """Structured diagnostic returned when syntax violates formal CFG grammar."""
    is_valid: bool
    error_message: str = ""
    line: int = 1
    column: int = 1
    expected_tokens: List[str] = field(default_factory=list)
    offending_token: str = ""


# =====================================================================
# Formal Lark Context-Free Grammar for Python Logic & Code
# =====================================================================

PYTHON_LOGIC_GRAMMAR = r"""
start: (stmt (";" | NEWLINE)?)*

?stmt: func_def
     | contract_stmt
     | assign_stmt
     | return_stmt
     | assert_stmt
     | if_stmt
     | for_stmt
     | while_stmt
     | expr_stmt

func_def: "def" CNAME "(" [param_list] ")" [ "->" CNAME ] ":" suite
param_list: param ("," param)*
param: CNAME [ ":" CNAME ]

contract_stmt: "require" ":" expr             -> require_contract
             | "ensure" ":" expr              -> ensure_contract
             | "invariant" ":" expr           -> invariant_contract

assign_stmt: CNAME ASSIGN_OP expr
ASSIGN_OP: "=" | "+=" | "-=" | "*=" | "/="

return_stmt: "return" [expr]
assert_stmt: "assert" expr ["," ESCAPED_STRING]
expr_stmt: expr

if_stmt: "if" expr ":" suite ["else" ":" suite]
for_stmt: "for" CNAME "in" expr ":" suite
while_stmt: "while" expr ":" suite

suite: "{" (stmt (";" | NEWLINE)?)* "}"
     | stmt

?expr: logical_or
?logical_or: logical_and ("or" logical_and)*
?logical_and: comparison ("and" comparison)*
?comparison: arith_expr (COMP_OP arith_expr)*
COMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">"

?arith_expr: term (ADD_OP term)*
ADD_OP: "+" | "-"

?term: power (MUL_OP power)*
MUL_OP: "*" | "/" | "//" | "%"

?power: factor ("**" factor)*

?factor: atom
       | ADD_OP factor
       | "not" factor

?atom: SIGNED_NUMBER           -> number
     | ESCAPED_STRING          -> string
     | "True"                  -> const_true
     | "False"                 -> const_false
     | "None"                  -> const_none
     | CNAME "(" [arg_list] ")" -> call
     | CNAME "[" expr "]"      -> subscript
     | CNAME                   -> var
     | "(" expr ")"

arg_list: expr ("," expr)*

NEWLINE: "\n" | "\r\n"

%import common.CNAME
%import common.SIGNED_NUMBER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


# =====================================================================
# Lark AST Transformer (ParseTree -> Typed Neuro-Symbolic AST)
# =====================================================================

class PythonLogicTransformer(Transformer):
    """Transforms Lark parse trees into typed PythonLogicAST nodes."""

    def start(self, items) -> ProgramNode:
        stmts = [it for it in items if isinstance(it, ASTNode)]
        return ProgramNode(statements=stmts)

    def number(self, items) -> ConstantNode:
        raw = str(items[0])
        val = float(raw) if ("." in raw or "e" in raw.lower()) else int(raw)
        return ConstantNode(value=val, type_name="float" if isinstance(val, float) else "int")

    def string(self, items) -> ConstantNode:
        s = str(items[0]).strip('"\'')
        return ConstantNode(value=s, type_name="str")

    def const_true(self, items) -> ConstantNode:
        return ConstantNode(value=True, type_name="bool")

    def const_false(self, items) -> ConstantNode:
        return ConstantNode(value=False, type_name="bool")

    def const_none(self, items) -> ConstantNode:
        return ConstantNode(value=None, type_name="NoneType")

    def var(self, items) -> VariableNode:
        return VariableNode(name=str(items[0]))

    def call(self, items) -> CallNode:
        func_name = str(items[0])
        args = list(items[1]) if len(items) > 1 and items[1] is not None else []
        return CallNode(func=func_name, args=args)

    def arg_list(self, items) -> List[ASTNode]:
        return [it for it in items if isinstance(it, ASTNode)]

    def subscript(self, items) -> SubscriptNode:
        var_node = VariableNode(name=str(items[0]))
        index_node = items[1]
        return SubscriptNode(value=var_node, index=index_node)

    def power(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        res = items[0]
        for it in items[1:]:
            res = BinaryOpNode(left=res, op="**", right=it)
        return res

    def term(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        res = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            res = BinaryOpNode(left=res, op=op, right=right)
            i += 2
        return res

    def arith_expr(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        res = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            res = BinaryOpNode(left=res, op=op, right=right)
            i += 2
        return res

    def comparison(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        res = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            right = items[i + 1]
            res = BinaryOpNode(left=res, op=op, right=right)
            i += 2
        return res

    def logical_and(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        res = items[0]
        for it in items[1:]:
            res = BinaryOpNode(left=res, op="and", right=it)
        return res

    def logical_or(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        res = items[0]
        for it in items[1:]:
            res = BinaryOpNode(left=res, op="or", right=it)
        return res

    def factor(self, items) -> ASTNode:
        if len(items) == 1:
            return items[0]
        if items[0] == "not":
            return UnaryOpNode(op="not", operand=items[1])
        if str(items[0]) in ("+", "-"):
            return UnaryOpNode(op=str(items[0]), operand=items[1])
        return items[0]

    def assign_stmt(self, items) -> AssignNode:
        target = str(items[0])
        op = str(items[1])
        val = items[2]
        return AssignNode(target=target, value=val, op=op)

    def return_stmt(self, items) -> ReturnNode:
        val = items[0] if items and isinstance(items[0], ASTNode) else None
        return ReturnNode(value=val)

    def assert_stmt(self, items) -> AssertNode:
        cond = items[0]
        msg = str(items[1]).strip('"\'') if len(items) > 1 and items[1] else None
        return AssertNode(condition=cond, message=msg)

    def require_contract(self, items) -> ContractNode:
        return ContractNode(contract_type="require", condition=items[0])

    def ensure_contract(self, items) -> ContractNode:
        return ContractNode(contract_type="ensure", condition=items[0])

    def invariant_contract(self, items) -> ContractNode:
        return ContractNode(contract_type="invariant", condition=items[0])

    def if_stmt(self, items) -> IfNode:
        cond = items[0]
        body = items[1] if isinstance(items[1], list) else [items[1]]
        orelse = []
        if len(items) > 2 and items[2]:
            orelse = items[2] if isinstance(items[2], list) else [items[2]]
        return IfNode(condition=cond, body=body, orelse=orelse)

    def for_stmt(self, items) -> ForNode:
        target = str(items[0])
        iterable = items[1]
        body = items[2] if isinstance(items[2], list) else [items[2]]
        return ForNode(target=target, iterable=iterable, body=body)

    def while_stmt(self, items) -> WhileNode:
        cond = items[0]
        body = items[1] if isinstance(items[1], list) else [items[1]]
        return WhileNode(condition=cond, body=body)

    def param(self, items) -> str:
        return str(items[0])

    def func_def(self, items) -> FunctionDefNode:
        name = str(items[0])
        params = items[1] if items[1] is not None and isinstance(items[1], list) else []
        suite_items = items[-1] if isinstance(items[-1], list) else [items[-1]]

        # Separate contracts from body
        contracts = [s for s in suite_items if isinstance(s, ContractNode)]
        body = [s for s in suite_items if not isinstance(s, ContractNode)]
        return FunctionDefNode(name=name, params=params, body=body, contracts=contracts)

    def param_list(self, items) -> List[str]:
        return [str(it) for it in items]

    def suite(self, items) -> List[ASTNode]:
        nodes = []
        for it in items:
            if isinstance(it, ASTNode):
                nodes.append(it)
            elif isinstance(it, list):
                nodes.extend([n for n in it if isinstance(n, ASTNode)])
        return nodes

    def expr_stmt(self, items) -> ASTNode:
        return items[0]


# =====================================================================
# Lark Python Logic Parser Class
# =====================================================================

class LarkPythonLogicParser:
    """
    Formal Lark Earley Context-Free Grammar Parser for Python Code & Logic.
    Translates raw text into deterministic, typed AST nodes with zero regex.
    """

    _instance: Optional["LarkPythonLogicParser"] = None

    @classmethod
    def get_instance(cls) -> "LarkPythonLogicParser":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.parser = Lark(PYTHON_LOGIC_GRAMMAR, parser="earley", propagate_positions=True)
        self.transformer = PythonLogicTransformer()

    def parse(self, code: str) -> Tuple[Optional[ProgramNode], SyntaxErrorDiagnosis]:
        """
        Parses code string into a ProgramNode AST.
        Returns: (ProgramNode or None, SyntaxErrorDiagnosis)
        """
        cleaned = code.strip()
        if not cleaned:
            return ProgramNode(statements=[]), SyntaxErrorDiagnosis(is_valid=True)

        try:
            tree = self.parser.parse(cleaned)
            ast_node = self.transformer.transform(tree)
            if isinstance(ast_node, ProgramNode):
                return ast_node, SyntaxErrorDiagnosis(is_valid=True)
            elif isinstance(ast_node, ASTNode):
                return ProgramNode(statements=[ast_node]), SyntaxErrorDiagnosis(is_valid=True)
            return None, SyntaxErrorDiagnosis(is_valid=False, error_message="Failed to transform AST tree")
        except UnexpectedToken as e:
            logger.debug("Lark syntax error (unexpected token): %s", e)
            return None, SyntaxErrorDiagnosis(
                is_valid=False,
                error_message=f"Unexpected token '{e.token}' at line {e.line}, column {e.column}",
                line=e.line,
                column=e.column,
                expected_tokens=list(e.expected) if hasattr(e, "expected") else [],
                offending_token=str(e.token)
            )
        except UnexpectedInput as e:
            logger.debug("Lark syntax error (unexpected input): %s", e)
            return None, SyntaxErrorDiagnosis(
                is_valid=False,
                error_message=f"Unexpected input at line {e.line}, column {e.column}",
                line=e.line,
                column=e.column
            )
        except LarkError as e:
            logger.debug("Lark error: %s", e)
            return None, SyntaxErrorDiagnosis(
                is_valid=False,
                error_message=f"Grammar parse error: {str(e)}"
            )
