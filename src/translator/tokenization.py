"""Deterministic Lexical Tokenizer & Intermediate Logic Representation (Zero Regex)."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any


@dataclass(frozen=True)
class SemanticToken:
    """
    Formal semantic token.
    Separates linguistic identification from mathematical semantics.
    Constructed via deterministic character lexing with ZERO regex.
    """
    text: str
    token_id: int
    token_type: str = "WORD"        # "WORD", "SCALAR", "PERCENTAGE", "PUNCT", "SPECIAL"
    raw_value: Optional[float] = None


@dataclass(frozen=True)
class LogicASTNode:
    """Structured Intermediate Logic Representation (ILR) node emitted by the Transformer."""
    predicate: str
    args: List[Any]


from lark import Lark, Transformer
from transformers import AutoTokenizer


LOGIC_GRAMMAR = r"""
?start: term

term: CNAME "(" [arg ("," arg)*] ")"   -> compound
    | SIGNED_NUMBER                     -> number
    | CNAME                             -> atom

arg: term

%import common.CNAME
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""


class LogicASTTransformer(Transformer):
    def compound(self, items):
        predicate = str(items[0])
        args = list(items[1:])
        return LogicASTNode(predicate=predicate, args=args)

    def arg(self, items):
        return items[0]

    def atom(self, items):
        return LogicASTNode(predicate=str(items[0]), args=[])

    def number(self, items):
        return LogicASTNode(predicate=str(items[0]), args=[])


class LogicSyntaxParser:
    """
    Validates and parses the structured Intermediate Logic Representation (ILR)
    emitted by the Transformer using a formal Context-Free Grammar (Lark Earley parser).
    Zero regular expressions.
    """
    _parser = Lark(LOGIC_GRAMMAR, parser='earley')
    _transformer = LogicASTTransformer()

    @classmethod
    def parse_ilr(cls, logic_str: str) -> LogicASTNode:
        s = logic_str.strip()
        if not s:
            raise ValueError("Empty logic string.")
        try:
            tree = cls._parser.parse(s)
            return cls._transformer.transform(tree)
        except Exception as e:
            raise ValueError(f"Grammar syntax error parsing logic string: {e}")


class TranslatorPipeline:
    """
    Pre-trained BPE Tokenizer + Lark CFG Pipeline.
    Zero regular expressions.
    1. Pre-trained BPE Tokenizer handles fluid human language, slang, and grammar.
    2. Lark Context-Free Grammar handles strict math and logic extraction.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.bpe_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.grammar_parser = LogicSyntaxParser()

    def tokenize_text(self, raw_text: str):
        """Tokenizes English text into BPE token IDs and attention mask."""
        return self.bpe_tokenizer(raw_text, return_tensors="pt")

    def parse_logic(self, logic_str: str) -> LogicASTNode:
        """Parses logic output string into formal AST via Context-Free Grammar."""
        return self.grammar_parser.parse_ilr(logic_str)


class SemanticTokenizer:
    """
    Deterministic Lexical Tokenizer.
    Operates strictly character-by-character. Completely free of regular expressions.
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.num_token = "<num>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.num_token]
        self.w2i: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.i2w: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.special_tokens)}

    def build_vocab(self, texts: List[str]):
        for text in texts:
            lexemes = self._lex_characters(text)
            for lex, _ in lexemes:
                if lex not in self.w2i and len(self.w2i) < self.vocab_size:
                    idx = len(self.w2i)
                    self.w2i[lex] = idx
                    self.i2w[idx] = lex

    def _lex_characters(self, text: str) -> List[Tuple[str, str]]:
        """
        Pure character-by-character scanner (Zero Regex).
        Returns list of (lexeme, token_type).
        """
        lexemes: List[Tuple[str, str]] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            # 1. Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # 2. Number lexing (handles signed numbers, floating points, percentages)
            if ch.isdigit() or (ch in ("+", "-") and i + 1 < n and (text[i + 1].isdigit() or text[i + 1] == ".")):
                start = i
                i += 1
                has_dot = (ch == ".")
                while i < n and (text[i].isdigit() or (text[i] == "." and not has_dot)):
                    if text[i] == ".":
                        has_dot = True
                    i += 1

                # Check trailing percentage
                if i < n and text[i] == "%":
                    i += 1
                    raw_num_str = text[start:i]
                    lexemes.append((raw_num_str, "PERCENTAGE"))
                else:
                    raw_num_str = text[start:i]
                    lexemes.append((raw_num_str, "SCALAR"))
                continue

            # 3. Word / Identifier lexing
            if ch.isalnum() or ch == "_":
                start = i
                while i < n and (text[i].isalnum() or text[i] == "_"):
                    i += 1
                word = text[start:i].lower()
                lexemes.append((word, "WORD"))
                continue

            # 4. Punctuation & individual symbols
            lexemes.append((ch, "PUNCT"))
            i += 1

        return lexemes

    def encode(self, text: str, auto_add: bool = False) -> List[SemanticToken]:
        """
        Encodes text into a sequence of typed SemanticTokens.
        Zero regex, pure character lexing.
        """
        lexemes = self._lex_characters(text)
        tokens: List[SemanticToken] = []

        tokens.append(SemanticToken(
            text=self.bos_token,
            token_id=self.w2i[self.bos_token],
            token_type="SPECIAL"
        ))

        for lex, ltype in lexemes:
            if ltype in ("SCALAR", "PERCENTAGE"):
                num_str = lex[:-1] if ltype == "PERCENTAGE" else lex
                try:
                    val = float(num_str)
                    num_id = self.w2i.get(self.num_token, self.w2i[self.unk_token])
                    tokens.append(SemanticToken(
                        text=lex,
                        token_id=num_id,
                        token_type=ltype,
                        raw_value=val  # Raw value is preserved without mutation
                    ))
                    continue
                except ValueError:
                    pass

            # Word / Punctuation
            if auto_add and lex not in self.w2i and len(self.w2i) < self.vocab_size:
                idx = len(self.w2i)
                self.w2i[lex] = idx
                self.i2w[idx] = lex

            tid = self.w2i.get(lex, self.w2i[self.unk_token])
            tokens.append(SemanticToken(
                text=lex,
                token_id=tid,
                token_type=ltype
            ))

        tokens.append(SemanticToken(
            text=self.eos_token,
            token_id=self.w2i[self.eos_token],
            token_type="SPECIAL"
        ))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            tok = self.i2w.get(tid, self.unk_token)
            if tok in (self.pad_token, self.bos_token, self.eos_token):
                continue
            words.append(tok)
        return " ".join(words)


# Backward-compatible alias
SimpleTokenizer = SemanticTokenizer
