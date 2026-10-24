"""Formal Context-Free Grammar AST Parser for Command Extraction (Zero Regex).

Uses Lark Earley Context-Free Grammar parsing to extract formal integer tokens
and semantic word tokens from natural human commands without procedural string splitting,
regex patterns, or heuristic .isdigit() checks.
"""

from typing import List, Tuple, Optional
from lark import Lark, Transformer


COMMAND_GRAMMAR = r'''
?start: command

command: (token | punct)+

token: INT -> int_val
     | CNAME -> word_val

punct: "." | "," | "?" | "!" | ":" | "-" | "'"

%import common.CNAME
%import common.INT
%import common.WS
%ignore WS
'''


class CommandASTTransformer(Transformer):
    def int_val(self, items):
        return ("INT", int(items[0]))

    def word_val(self, items):
        return ("WORD", str(items[0]))

    def command(self, items):
        return [it for it in items if isinstance(it, tuple)]


class CommandASTParser:
    """
    Formal Lark CFG Parser for Imperative Commands.
    Replaces procedural string splitting, regex, and isdigit heuristics
    with strict Context-Free Grammar AST tokens.
    """
    _instance: Optional["CommandASTParser"] = None

    @classmethod
    def get_instance(cls) -> "CommandASTParser":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.parser = Lark(COMMAND_GRAMMAR, parser="earley")
        self.transformer = CommandASTTransformer()

    def parse_command_tokens(self, text: str) -> Tuple[List[int], List[str]]:
        """Parses natural text into formal AST integers and word tokens."""
        clean = text.replace('"', '').replace("'", "")
        try:
            tree = self.parser.parse(clean)
            nodes = self.transformer.transform(tree)
            ints = [val for tag, val in nodes if tag == "INT"]
            words = [val for tag, val in nodes if tag == "WORD"]
            return ints, words
        except Exception:
            return [], []
