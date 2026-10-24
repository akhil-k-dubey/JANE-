"""LaTeX Mathematical Formula Normalizer for Project Jane.

Converts raw LaTeX mathematical expressions from scientific papers, abstracts,
and textbooks into clean, canonical AST formulas compatible with SymPy and Z3 SMT.

Constructed via a Deterministic Recursive-Descent Lexer and AST Parser.
ZERO Regular Expressions.
"""

from typing import List, Dict, Optional, Tuple, Any


class LaTeXMathNormalizer:
    """
    Robust, deterministic LaTeX math lexer and AST normalizer.
    Completely regex-free.
    """

    GREEK_AND_PHYSICAL_SYMBOLS = {
        "\\hbar": "hbar",
        "\\lambda": "wavelength",
        "\\nu": "frequency",
        "\\omega": "angular_velocity",
        "\\sigma": "stefan_boltzmann_constant",
        "\\rho": "density",
        "\\mu": "permeability",
        "\\epsilon": "permittivity",
        "\\theta": "angle",
        "\\phi": "phi",
        "\\Phi_0": "flux_quantum",
        "\\Phi": "magnetic_flux",
        "\\tau": "time_constant",
        "\\alpha": "fine_structure_constant",
        "\\beta": "beta",
        "\\gamma": "lorentz_factor",
        "\\partial": "partial_"
    }

    IGNORE_COMMANDS = {
        "\\mathrm", "\\mathbf", "\\mathit", "\\text", "\\operatorname",
        "\\left", "\\right", "\\displaystyle", "\\,", "\\;", "\\!", "\\quad", "\\qquad"
    }

    OPERATOR_MAP = {
        "\\cdot": "*",
        "\\times": "*",
        "\\ge": ">=",
        "\\geq": ">=",
        "\\le": "<=",
        "\\leq": "<=",
        "\\approx": "==",
        "\\sim": "==",
        "\\equiv": "==",
        "\\pm": "+-"
    }

    @classmethod
    def extract_latex_blocks(cls, text: str) -> List[str]:
        """
        Extracts all LaTeX mathematical expressions from text using character scanning.
        Zero regular expressions.
        """
        blocks: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            # 1. Check for environments: \begin{equation}, \begin{align}, etc.
            if text.startswith("\\begin{", i):
                brace_end = text.find("}", i + 7)
                if brace_end != -1:
                    env_name = text[i + 7:brace_end].rstrip("*")
                    if env_name in ("equation", "align", "gather"):
                        end_tag = f"\\end{{{text[i + 7:brace_end]}}}"
                        end_pos = text.find(end_tag, brace_end + 1)
                        if end_pos != -1:
                            block_content = text[brace_end + 1:end_pos].strip()
                            if block_content and block_content not in blocks:
                                blocks.append(block_content)
                            i = end_pos + len(end_tag)
                            continue

            # 2. Check for bracket display math: \[ ... \]
            if text.startswith("\\[", i):
                end_pos = text.find("\\]", i + 2)
                if end_pos != -1:
                    block_content = text[i + 2:end_pos].strip()
                    if block_content and block_content not in blocks:
                        blocks.append(block_content)
                    i = end_pos + 2
                    continue

            # 3. Check for double dollar display math: $$ ... $$
            if text.startswith("$$", i):
                end_pos = text.find("$$", i + 2)
                if end_pos != -1:
                    block_content = text[i + 2:end_pos].strip()
                    if block_content and block_content not in blocks:
                        blocks.append(block_content)
                    i = end_pos + 2
                    continue

            # 4. Check for inline dollar math: $ ... $
            if text[i] == "$" and (i == 0 or text[i - 1] != "\\") and not text.startswith("$$", i):
                end_pos = text.find("$", i + 1)
                if end_pos != -1 and not text.startswith("$$", end_pos) and text[end_pos - 1] != "\\":
                    candidate = text[i + 1:end_pos].strip()
                    # Filter out empty or monetary amounts (e.g. $100)
                    is_money = candidate and all(c.isdigit() or c == "." for c in candidate)
                    if candidate and not is_money:
                        # Must contain math operators or LaTeX command
                        if any(c in candidate for c in ("=", "+", "-", "*", "/", "\\", "^", "_")):
                            if candidate not in blocks:
                                blocks.append(candidate)
                    i = end_pos + 1
                    continue

            i += 1

        return blocks

    @classmethod
    def _extract_group(cls, s: str, start: int) -> Tuple[str, int]:
        """
        Extracts balanced group enclosed by '{' and '}'.
        Returns (content, next_index).
        """
        if start >= len(s) or s[start] != "{":
            # Single character group
            if start < len(s):
                return s[start], start + 1
            return "", start

        depth = 0
        i = start
        while i < len(s):
            if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
                depth += 1
            elif s[i] == "}" and (i == 0 or s[i - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    return s[start + 1:i], i + 1
            i += 1
        return s[start + 1:], len(s)

    @classmethod
    def _extract_bracket_group(cls, s: str, start: int) -> Tuple[str, int]:
        """Extracts balanced group enclosed by '[' and ']'."""
        if start >= len(s) or s[start] != "[":
            return "", start
        depth = 0
        i = start
        while i < len(s):
            if s[i] == "[":
                depth += 1
            elif s[i] == "]":
                depth -= 1
                if depth == 0:
                    return s[start + 1:i], i + 1
            i += 1
        return s[start + 1:], len(s)

    @classmethod
    def normalize(cls, latex_str: str) -> str:
        """
        Translates a raw LaTeX formula string into a canonical, evaluable math AST string.
        Completely deterministic, zero regex.
        """
        s = latex_str.strip().strip("$")
        i = 0
        n = len(s)
        out: List[str] = []

        while i < n:
            ch = s[i]

            # 1. Skip whitespace
            if ch.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
                i += 1
                continue

            # 2. LaTeX Backslash Commands
            if ch == "\\":
                cmd_start = i
                i += 1
                while i < n and (s[i].isalpha() or s[i] in ",;!"):
                    i += 1
                cmd = s[cmd_start:i]

                # Check ignore formatting commands
                if cmd in cls.IGNORE_COMMANDS:
                    continue

                # Check operator mapping
                if cmd in cls.OPERATOR_MAP:
                    out.append(f" {cls.OPERATOR_MAP[cmd]} ")
                    continue

                # Check fraction: \frac{num}{den}
                if cmd == "\\frac":
                    while i < n and s[i].isspace():
                        i += 1
                    num_raw, i = cls._extract_group(s, i)
                    while i < n and s[i].isspace():
                        i += 1
                    den_raw, i = cls._extract_group(s, i)
                    norm_num = cls.normalize(num_raw)
                    norm_den = cls.normalize(den_raw)
                    out.append(f"(({norm_num}) / ({norm_den}))")
                    continue

                # Check square root: \sqrt{arg} or \sqrt[n]{arg}
                if cmd == "\\sqrt":
                    while i < n and s[i].isspace():
                        i += 1
                    root_idx = ""
                    if i < n and s[i] == "[":
                        root_idx, i = cls._extract_bracket_group(s, i)
                    while i < n and s[i].isspace():
                        i += 1
                    arg_raw, i = cls._extract_group(s, i)
                    norm_arg = cls.normalize(arg_raw)
                    if root_idx:
                        norm_idx = cls.normalize(root_idx)
                        out.append(f"(({norm_arg})**(1/({norm_idx})))")
                    else:
                        out.append(f"sqrt({norm_arg})")
                    continue

                # Check Delta variable: \Delta x -> delta_x
                if cmd == "\\Delta":
                    while i < n and s[i].isspace():
                        i += 1
                    var_name = ""
                    while i < n and (s[i].isalnum() or s[i] == "_"):
                        var_name += s[i]
                        i += 1
                    out.append(f"delta_{var_name}" if var_name else "delta")
                    continue

                # Check Greek / physical symbols (e.g. \Phi_0, \hbar, etc.)
                # Check for subscript attached immediately like \Phi_0
                full_sym = cmd
                if i < n and s[i] == "_":
                    sub_raw, next_i = cls._extract_group(s, i + 1)
                    cand = f"{cmd}_{sub_raw}"
                    if cand in cls.GREEK_AND_PHYSICAL_SYMBOLS:
                        full_sym = cand
                        i = next_i

                if full_sym in cls.GREEK_AND_PHYSICAL_SYMBOLS:
                    out.append(cls.GREEK_AND_PHYSICAL_SYMBOLS[full_sym])
                    continue
                elif cmd in cls.GREEK_AND_PHYSICAL_SYMBOLS:
                    out.append(cls.GREEK_AND_PHYSICAL_SYMBOLS[cmd])
                    continue
                else:
                    # Strip leading slash and emit clean identifier
                    out.append(cmd.lstrip("\\"))
                    continue

            # 3. Exponent handling: ^2 or ^{...}
            if ch == "^":
                exp_raw, i = cls._extract_group(s, i + 1)
                norm_exp = cls.normalize(exp_raw)
                out.append(f"**({norm_exp})" if len(norm_exp) > 1 else f"**{norm_exp}")
                continue

            # 4. Subscript handling: _1 or _{...}
            if ch == "_":
                sub_raw, i = cls._extract_group(s, i + 1)
                norm_sub = cls.normalize(sub_raw)
                out.append(f"_{norm_sub}")
                continue

            # 5. Operators: =, >=, <=, etc.
            if ch == "=":
                if (i > 0 and s[i - 1] in "<>!=") or (i + 1 < n and s[i + 1] == "="):
                    out.append("=")
                else:
                    out.append(" == ")
                i += 1
                continue

            # 6. Basic characters, numbers, parentheses
            out.append(ch)
            i += 1

        # Post-process: insert implied multiplications for juxtaposed terms
        # (e.g. 2e -> 2 * e, m c -> m * c)
        raw_res = "".join(out).strip()
        tokens: List[str] = []
        k = 0
        m = len(raw_res)

        while k < m:
            c = raw_res[k]
            if c.isspace():
                if tokens and tokens[-1] != " ":
                    tokens.append(" ")
                k += 1
                continue

            # Multi-char operators or words
            if c.isdigit() or (c == "." and k + 1 < m and raw_res[k + 1].isdigit()):
                num_str = ""
                while k < m and (raw_res[k].isdigit() or raw_res[k] == "."):
                    num_str += raw_res[k]
                    k += 1
                tokens.append(num_str)
                continue

            if c.isalpha() or c == "_":
                ident_str = ""
                while k < m and (raw_res[k].isalnum() or raw_res[k] == "_"):
                    ident_str += raw_res[k]
                    k += 1
                tokens.append(ident_str)
                continue

            # Operators
            if raw_res.startswith("**", k):
                tokens.append("**")
                k += 2
                continue
            if raw_res.startswith("==", k):
                tokens.append("==")
                k += 2
                continue
            if raw_res.startswith(">=", k):
                tokens.append(">=")
                k += 2
                continue
            if raw_res.startswith("<=", k):
                tokens.append("<=")
                k += 2
                continue
            if raw_res.startswith("+-", k):
                tokens.append("+-")
                k += 2
                continue

            tokens.append(c)
            k += 1

        # Insert multiplication between juxtaposed tokens
        final_tokens: List[str] = []
        for idx, tok in enumerate(tokens):
            if idx > 0 and tok != " ":
                prev = final_tokens[-1] if final_tokens else ""
                prev_non_space = final_tokens[-2] if len(final_tokens) >= 2 and final_tokens[-1] == " " else prev

                # Juxtaposed cases:
                # 1. Number followed directly by Identifier (e.g. '2', 'e' -> '2 * e')
                # 2. Identifier followed by Identifier separated by space (e.g. 'm', 'c' -> 'm * c')
                is_num_prev = prev_non_space and prev_non_space[0].isdigit()
                is_ident_prev = prev_non_space and (prev_non_space[0].isalpha() or prev_non_space[0] == "_")
                is_ident_curr = tok and (tok[0].isalpha() or tok[0] == "_")

                if (is_num_prev and is_ident_curr) or (is_ident_prev and is_ident_curr and prev == " "):
                    if final_tokens[-1] == " ":
                        final_tokens[-1] = " * "
                    else:
                        final_tokens.append(" * ")

            final_tokens.append(tok)

        # Clean spaces and return
        res_str = "".join(final_tokens)
        cleaned_words = [w for w in res_str.split(" ") if w]
        return " ".join(cleaned_words)
