"""Core execution pipeline tying all modules together."""

from .config import (
    JaneConfig,
    ManifoldConfig,
    TheoryOfMindConfig,
    AudioConfig,
    EmbodimentConfig,
    PipelineConfig,
    DEFAULT_CONFIG
)

_LAZY_EXPORTS = {
    "JanePipeline": (".pipeline", "JanePipeline"),
    "DecoupledAsynchronousEmbodiment": (".async_embodiment", "DecoupledAsynchronousEmbodiment"),
    "SharedDNCPlanState": (".async_embodiment", "SharedDNCPlanState"),
    "ActionExecutorThread": (".async_embodiment", "ActionExecutorThread"),
    "AudioEncoderThread": (".async_embodiment", "AudioEncoderThread"),
    "DeliberationEngineThread": (".async_embodiment", "DeliberationEngineThread"),
    "ModeRouter": (".mode_router", "ModeRouter"),
    "OperatingMode": (".mode_router", "OperatingMode"),
    "JaneTerminalChat": (".interactive_chat", "JaneTerminalChat"),
    "ReactiveTemporalEngine": (".reactive_temporal_loop", "ReactiveTemporalEngine"),
    "ReactiveExecutionAudit": (".reactive_temporal_loop", "ReactiveExecutionAudit"),
    "APIGateway": (".api_gateway", "APIGateway"),
    "ActionASTNode": (".api_gateway", "ActionASTNode"),
    "GatewayResult": (".api_gateway", "GatewayResult"),
    "AnticipatoryEngine": (".anticipatory_engine", "AnticipatoryEngine"),
    "EgoNode": (".anticipatory_engine", "EgoNode"),
    "CuriosityDaemon": (".curiosity_daemon", "CuriosityDaemon"),
    "CodebaseIntrospector": (".codebase_introspector", "CodebaseIntrospector")
}

__all__ = [
    "JaneConfig",
    "ManifoldConfig",
    "TheoryOfMindConfig",
    "AudioConfig",
    "EmbodimentConfig",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    *_LAZY_EXPORTS.keys()
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        module_path, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

