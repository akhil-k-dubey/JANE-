"""Read-Only Codebase Self-Introspection Engine for Project Jane.

Provides Jane with dynamic, grounded self-awareness of her own source code:
1. Pure AST Static Analysis (ast.parse):
   - Zero eval, zero exec, zero subprocess execution.
   - Strictly read-only inspection of module structures, classes, methods, and AST nodes.
2. Sandboxed File Traversal:
   - Restricts code inspection strictly to the project root directory.
   - Prevents path traversal outside the project workspace.
3. Symbol & Module Resolver:
   - Indexes all modules in src/ (core, reasoning_engine, symbolic_verifier, perception, translator, etc.).
   - Resolves classes, functions, docstrings, argument signatures, and call dependencies.
4. Natural Introspective Grounding:
   - Connects to SelfModelEngine and JanePipeline so Jane can explain her actual
     source code, algorithms, and invariants dynamically rather than relying on hallucinated memory.
"""

import ast
import os
from typing import Dict, List, Any, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("codebase_introspector")


class CodebaseIntrospector:
    """
    Read-Only AST Codebase Introspector.
    Inspects, indexes, and queries Project Jane's own source code.
    """

    def __init__(self, project_root: Optional[str] = None):
        if project_root:
            self.project_root = os.path.abspath(project_root)
        else:
            # Auto-detect project root from current file location (../../)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.abspath(os.path.join(current_dir, "../.."))

        self.src_dir = os.path.join(self.project_root, "src")
        self._module_cache: Dict[str, str] = {}
        self._parsed_ast_cache: Dict[str, Dict[str, Any]] = {}
        self.refresh_index()

    def refresh_index(self):
        """Scans the src/ directory and indexes all Python modules."""
        self._module_cache.clear()
        self._parsed_ast_cache.clear()

        if not os.path.exists(self.src_dir):
            logger.warning("Source directory '%s' does not exist.", self.src_dir)
            return

        for root, _, files in os.walk(self.src_dir):
            for f in files:
                if f.endswith(".py") and not f.startswith("."):
                    full_path = os.path.abspath(os.path.join(root, f))
                    rel_path = os.path.relpath(full_path, self.project_root)

                    # Store multiple access keys:
                    # 1. Base filename without .py (e.g. "api_gateway")
                    stem = os.path.splitext(f)[0]
                    self._module_cache[stem.lower()] = full_path

                    # 2. Relative path (e.g. "src/core/api_gateway.py")
                    self._module_cache[rel_path.lower()] = full_path

                    # 3. Dotted module name (e.g. "src.core.api_gateway")
                    dotted = rel_path.replace(os.sep, ".").rstrip(".py")
                    self._module_cache[dotted.lower()] = full_path

    def _is_safe_path(self, target_path: str) -> bool:
        """Ensures target path is safely within the project root."""
        abs_path = os.path.abspath(target_path)
        return abs_path.startswith(self.project_root)

    def resolve_module_path(self, identifier: str) -> Optional[str]:
        """Resolves a module identifier, filename, or symbol to its absolute path."""
        clean = identifier.strip().lower().replace(" ", "_")
        if clean.endswith(".py"):
            clean = clean[:-3]

        # 1. Direct dictionary match
        if clean in self._module_cache:
            return self._module_cache[clean]

        # 2. Partial match on stem or path
        for key, path in self._module_cache.items():
            if clean in key or key in clean:
                return path

        # 3. Direct path check within project root
        candidate = os.path.join(self.project_root, identifier)
        if not candidate.endswith(".py"):
            candidate += ".py"
        if os.path.exists(candidate) and self._is_safe_path(candidate):
            return os.path.abspath(candidate)

        return None

    def introspect_module(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Parses a module into an AST and extracts structured metadata:
        docstring, classes, methods, functions, and line metrics.
        """
        full_path = self.resolve_module_path(identifier)
        if not full_path or not os.path.exists(full_path):
            return None

        if not self._is_safe_path(full_path):
            logger.warning("Attempted path traversal rejected: %s", full_path)
            return None

        if full_path in self._parsed_ast_cache:
            return self._parsed_ast_cache[full_path]

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                code_text = f.read()

            tree = ast.parse(code_text, filename=full_path)
            module_doc = ast.get_docstring(tree) or ""

            classes = []
            functions = []
            imports = []

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    cls_doc = ast.get_docstring(node) or ""
                    bases = [ast.unparse(b) for b in node.bases]
                    methods = []

                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            m_doc = ast.get_docstring(item) or ""
                            args = [a.arg for a in item.args.args]
                            decorators = [ast.unparse(d) for d in item.decorator_list]
                            methods.append({
                                "name": item.name,
                                "args": args,
                                "docstring": m_doc,
                                "decorators": decorators,
                                "is_async": isinstance(item, ast.AsyncFunctionDef),
                                "line_start": item.lineno
                            })

                    classes.append({
                        "name": node.name,
                        "bases": bases,
                        "docstring": cls_doc,
                        "methods": methods,
                        "line_start": node.lineno
                    })

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    f_doc = ast.get_docstring(node) or ""
                    args = [a.arg for a in node.args.args]
                    decorators = [ast.unparse(d) for d in node.decorator_list]
                    functions.append({
                        "name": node.name,
                        "args": args,
                        "docstring": f_doc,
                        "decorators": decorators,
                        "line_start": node.lineno
                    })

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for alias in node.names:
                        imports.append(f"{mod}.{alias.name}" if mod else alias.name)

            rel_path = os.path.relpath(full_path, self.project_root)
            line_count = len(code_text.splitlines())

            meta = {
                "file_path": rel_path,
                "absolute_path": full_path,
                "docstring": module_doc,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "lines_of_code": line_count
            }
            self._parsed_ast_cache[full_path] = meta
            return meta

        except Exception as e:
            logger.error("Failed to parse AST for %s: %s", full_path, e, exc_info=True)
            return None

    def find_symbol(self, symbol_name: str) -> List[Dict[str, Any]]:
        """
        Searches the entire codebase for a class or function matching symbol_name.
        """
        results: List[Dict[str, Any]] = []
        sym_clean = symbol_name.strip().lower()

        # Iterate over all indexed modules
        seen_paths = set(self._module_cache.values())
        for path in seen_paths:
            mod_data = self.introspect_module(path)
            if not mod_data:
                continue

            # Check classes
            for cls_info in mod_data["classes"]:
                cls_lower = cls_info["name"].lower()
                if sym_clean == cls_lower or (len(sym_clean) >= 6 and (sym_clean in cls_lower or cls_lower in sym_clean)):
                    results.append({
                        "type": "class",
                        "name": cls_info["name"],
                        "file_path": mod_data["file_path"],
                        "line_start": cls_info["line_start"],
                        "docstring": cls_info["docstring"],
                        "bases": cls_info["bases"],
                        "methods": cls_info["methods"]
                    })

                # Check methods inside classes
                for m in cls_info["methods"]:
                    m_lower = m["name"].lower()
                    if sym_clean == m_lower:
                        results.append({
                            "type": "method",
                            "name": f"{cls_info['name']}.{m['name']}",
                            "file_path": mod_data["file_path"],
                            "line_start": m["line_start"],
                            "docstring": m["docstring"],
                            "args": m["args"],
                            "class_name": cls_info["name"]
                        })

            # Check top-level functions
            for fn_info in mod_data["functions"]:
                fn_lower = fn_info["name"].lower()
                if sym_clean == fn_lower:
                    results.append({
                        "type": "function",
                        "name": fn_info["name"],
                        "file_path": mod_data["file_path"],
                        "line_start": fn_info["line_start"],
                        "docstring": fn_info["docstring"],
                        "args": fn_info["args"]
                    })

        return results

    def format_module_report(self, identifier: str) -> str:
        """Formats a human-readable markdown breakdown of a module's AST structure."""
        data = self.introspect_module(identifier)
        if not data:
            return f"Module '{identifier}' was not found in Jane's codebase index."

        lines = [
            f"### Module: `{data['file_path']}` ({data['lines_of_code']} LOC)",
            ""
        ]

        if data["docstring"]:
            doc_snippet = data["docstring"].strip().split("\n\n")[0]
            lines.append(f"> **Overview**: {doc_snippet}\n")

        if data["classes"]:
            lines.append("**Classes Defined:**")
            for c in data["classes"]:
                bases_str = f"({', '.join(c['bases'])})" if c["bases"] else ""
                lines.append(f"- `class {c['name']}{bases_str}` (Line {c['line_start']})")
                if c["docstring"]:
                    first_line = c["docstring"].strip().split("\n")[0]
                    lines.append(f"  - *{first_line}*")
                for m in c["methods"]:
                    args_str = ", ".join(m["args"])
                    lines.append(f"  - `def {m['name']}({args_str})`")
            lines.append("")

        if data["functions"]:
            lines.append("**Top-Level Functions:**")
            for f in data["functions"]:
                args_str = ", ".join(f["args"])
                lines.append(f"- `def {f['name']}({args_str})` (Line {f['line_start']})")
            lines.append("")

        return "\n".join(lines)

    def format_symbol_report(self, symbol_name: str) -> str:
        """Formats a markdown report for a specific class or function."""
        matches = self.find_symbol(symbol_name)
        if not matches:
            return f"Symbol '{symbol_name}' was not found in Jane's codebase."

        lines = [f"### Codebase AST Symbol Match: `{symbol_name}`", ""]
        for item in matches[:3]:
            t = item["type"].upper()
            lines.append(f"**[{t}] `{item['name']}`** in `{item['file_path']}:{item['line_start']}`")
            if item.get("docstring"):
                lines.append(f"> {item['docstring'].strip()}")
            if item.get("args"):
                lines.append(f"- Signature: `({', '.join(item['args'])})`")
            if item.get("methods"):
                lines.append("- Methods:")
                for m in item["methods"]:
                    lines.append(f"  - `{m['name']}({', '.join(m['args'])})`")
            lines.append("")

        return "\n".join(lines)

    def get_codebase_overview(self) -> Dict[str, Any]:
        """Provides an authoritative overview of all modules, classes, and packages."""
        packages: Dict[str, List[str]] = {}
        total_classes = 0
        total_functions = 0
        total_loc = 0

        seen_paths = set(self._module_cache.values())
        for path in seen_paths:
            data = self.introspect_module(path)
            if not data:
                continue

            rel_path = data["file_path"]
            parts = rel_path.split(os.sep)
            pkg = parts[1] if len(parts) > 1 else "root"

            if pkg not in packages:
                packages[pkg] = []
            packages[pkg].append(parts[-1])

            total_classes += len(data["classes"])
            total_functions += len(data["functions"])
            total_loc += data["lines_of_code"]

        return {
            "total_modules": len(seen_paths),
            "packages": packages,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_loc": total_loc
        }

    def answer_codebase_query(self, query: str) -> Optional[str]:
        """
        Translates a natural language query about Jane's code into an authoritative,
        AST-grounded explanation.
        """
        q_lower = query.lower()

        # 1. Direct module name query (e.g. "api_gateway" or "curiosity_daemon")
        clean_q = q_lower.strip().replace(".py", "").replace(" ", "_")
        if clean_q in self._module_cache:
            return self.format_module_report(clean_q)

        # 2. Targeted module queries - requires specific module mention
        module_names = [
            "api_gateway", "curiosity_daemon", "anticipatory_engine", "hebbian_manifold",
            "mode_router", "z3_interface", "pipeline", "self_model", "episodic_ledger",
            "active_learner", "paper_ingestor", "wikipedia_ingestor", "codebase_introspector",
            "interactive_chat", "energy_constraints", "nalu_arithmetic", "dnc_memory"
        ]

        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()
        sim_codebase = m.similarity(query, "codebase_introspection")

        matched_mod = None
        for mod_name in module_names:
            mod_clean = mod_name.replace("_", " ")
            if mod_name in q_lower or mod_clean in q_lower:
                matched_mod = mod_name
                break

        if matched_mod and sim_codebase >= 0.20:
            return self.format_module_report(matched_mod)

        # 3. Overall codebase structure / architecture queries
        if sim_codebase >= 0.65 and not matched_mod:
            overview = self.get_codebase_overview()
            pkg_lines = []
            for pkg, mods in sorted(overview["packages"].items()):
                pkg_lines.append(f"  • **{pkg}**: {', '.join(mods)}")
            return (
                f"Jane Source Codebase Architecture (Read-Only AST Introspection):\n"
                f"  • Total Modules: {overview['total_modules']} files ({overview['total_loc']} lines of code)\n"
                f"  • Defined Classes: {overview['total_classes']} | Functions: {overview['total_functions']}\n"
                f"Packages:\n" + "\n".join(pkg_lines)
            )

        # 3. Check for specific class/symbol queries
        # Deterministic token scanning: extract words/identifiers without regular expressions
        tokens = []
        curr = []
        for ch in query:
            if ch.isalnum() or ch == '_':
                curr.append(ch)
            else:
                if curr:
                    tokens.append("".join(curr))
                    curr = []
        if curr:
            tokens.append("".join(curr))

        explicit_symbol_match = []
        pascal_case_words = []
        for i, token in enumerate(tokens):
            if token.lower() in ("class", "function", "def", "symbol") and i + 1 < len(tokens):
                explicit_symbol_match.append(tokens[i + 1])
            # Check for PascalCase / CamelCase: starts uppercase, has both upper and lower
            if len(token) >= 2 and token[0].isupper() and any(c.islower() for c in token) and any(c.isupper() for c in token[1:]):
                pascal_case_words.append(token)

        candidate_symbols = explicit_symbol_match + pascal_case_words

        if not candidate_symbols and sim_codebase >= 0.50:
            candidate_symbols = tokens

        for sym in candidate_symbols:
            matches = self.find_symbol(sym)
            if matches:
                return self.format_symbol_report(sym)

        return None
