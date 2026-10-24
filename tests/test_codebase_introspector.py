"""Unit and Integration Tests for Project Jane's Read-Only Codebase Self-Introspection Engine."""

import os
import pytest
from src.core.codebase_introspector import CodebaseIntrospector
from src.reasoning_engine.self_model import SelfModelEngine


def test_codebase_introspector_initialization_and_indexing():
    """Verify that CodebaseIntrospector correctly discovers and indexes modules in src/."""
    introspector = CodebaseIntrospector()
    assert os.path.exists(introspector.project_root)
    assert os.path.exists(introspector.src_dir)

    overview = introspector.get_codebase_overview()
    assert overview["total_modules"] >= 30, f"Expected >= 30 modules, found {overview['total_modules']}"
    assert overview["total_classes"] >= 50, f"Expected >= 50 classes, found {overview['total_classes']}"
    assert overview["total_loc"] >= 5000, f"Expected >= 5000 LOC, found {overview['total_loc']}"

    # Verify key modules exist in cache
    for key in ("api_gateway", "pipeline", "self_model", "curiosity_daemon", "mode_router"):
        assert key in introspector._module_cache, f"Module '{key}' missing from cache"


def test_codebase_introspector_sandboxing_path_traversal():
    """Verify that path traversal outside the project root is strictly blocked."""
    introspector = CodebaseIntrospector()

    # Outside paths must fail safety check
    assert not introspector._is_safe_path("/etc/passwd")
    assert not introspector._is_safe_path("/var/log")
    assert not introspector._is_safe_path(os.path.abspath(os.path.join(introspector.project_root, "../..")))

    # Introspecting an outside path returns None
    res = introspector.introspect_module("../../../../etc/passwd")
    assert res is None

    res2 = introspector.introspect_module("/etc/hosts")
    assert res2 is None


def test_module_ast_parsing_and_structure():
    """Verify pure AST extraction of classes, methods, arguments, imports, and docstrings."""
    introspector = CodebaseIntrospector()
    data = introspector.introspect_module("api_gateway")

    assert data is not None
    assert "file_path" in data
    assert "src/core/api_gateway.py" in data["file_path"].replace("\\", "/")
    assert data["lines_of_code"] > 0
    assert len(data["classes"]) >= 1

    # Check APIGateway class details
    class_names = [c["name"] for c in data["classes"]]
    assert "APIGateway" in class_names

    api_gw = next(c for c in data["classes"] if c["name"] == "APIGateway")
    method_names = [m["name"] for m in api_gw["methods"]]
    assert "execute_verified_action" in method_names

    exec_m = next(m for m in api_gw["methods"] if m["name"] == "execute_verified_action")
    assert "ast_node" in exec_m["args"]
    assert exec_m["line_start"] > 0


def test_find_symbol_classes_and_methods():
    """Verify symbol search accurately resolves classes and functions across files."""
    introspector = CodebaseIntrospector()

    # Search for class APIGateway
    matches = introspector.find_symbol("APIGateway")
    assert len(matches) > 0
    top = matches[0]
    assert top["name"] == "APIGateway"
    assert top["type"] == "class"
    assert "api_gateway" in top["file_path"]

    # Search for method execute_verified_action
    matches_m = introspector.find_symbol("execute_verified_action")
    assert len(matches_m) > 0
    assert any("execute_verified_action" in m["name"] for m in matches_m)

    # Search for non-existent symbol
    matches_none = introspector.find_symbol("NonExistentSymbol12345xyz")
    assert len(matches_none) == 0


def test_formatting_reports():
    """Verify human-readable markdown generation for modules and symbols."""
    introspector = CodebaseIntrospector()

    mod_report = introspector.format_module_report("curiosity_daemon")
    assert "### Module:" in mod_report
    assert "CuriosityDaemon" in mod_report

    sym_report = introspector.format_symbol_report("CuriosityDaemon")
    assert "### Codebase AST Symbol Match:" in sym_report
    assert "CuriosityDaemon" in sym_report

    missing_report = introspector.format_module_report("totally_fake_module_abc")
    assert "was not found" in missing_report


def test_natural_language_codebase_query_resolution():
    """Verify natural language queries are routed to appropriate AST introspections."""
    introspector = CodebaseIntrospector()

    # Overview query
    overview_ans = introspector.answer_codebase_query("give me a codebase overview")
    assert overview_ans is not None
    assert "Total Modules:" in overview_ans
    assert "Packages:" in overview_ans

    # Module query
    mod_ans = introspector.answer_codebase_query("how is api_gateway implemented")
    assert mod_ans is not None
    assert "APIGateway" in mod_ans

    # Symbol query
    sym_ans = introspector.answer_codebase_query("what is CuriosityDaemon")
    assert sym_ans is not None
    assert "CuriosityDaemon" in sym_ans


def test_self_model_codebase_integration():
    """Verify SelfModelEngine detects codebase queries and returns grounded AST responses."""
    introspector = CodebaseIntrospector()
    self_model = SelfModelEngine(codebase_introspector=introspector)

    # Intent detection
    assert self_model.is_self_query("tell me about your codebase")
    assert self_model.is_self_query("what modules do you have")
    assert self_model.is_self_query("inspect module api_gateway")

    # Grounded response
    resp = self_model.answer_self_query("tell me about your codebase")
    assert "Total Modules:" in resp
    assert "Packages:" in resp

    resp_mod = self_model.answer_self_query("how is api_gateway implemented")
    assert "APIGateway" in resp_mod


def test_pipeline_introspect_codebase():
    """Verify JanePipeline.introspect_codebase delegating correctly."""
    from src.core.pipeline import JanePipeline

    # Instantiate light pipeline on CPU without pretrained translator to keep test rapid
    pipeline = JanePipeline(use_pretrained_translator=False, device="cpu")
    assert hasattr(pipeline, "codebase_introspector")
    assert pipeline.codebase_introspector is not None

    overview = pipeline.introspect_codebase()
    assert "Total Modules:" in overview

    gw_report = pipeline.introspect_codebase("api_gateway")
    assert "APIGateway" in gw_report
