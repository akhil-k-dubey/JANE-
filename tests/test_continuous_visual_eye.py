"""Unit and integration test suite for Project Jane's Continuous Visual Eye,
OS Screen Perception, and Dynamic Visual Design Learning Engine.

Zero regex invariants enforced.
"""

import time
import torch
import pytest

from src.perception.continuous_visual_eye import (
    ContinuousVisualEye,
    ScreenCaptureEngine,
    VisualDesignLearner,
    DesignKnowledgeStore,
    ScreenFrame,
    LearnedDesignArchetype,
    ObservedVisualContext
)
from src.symbolic_verifier.neuro_symbolic_synthesizer import (
    MathematicalStyleEngine,
    NeuroSymbolicProgramSynthesizer
)


def test_screen_capture_engine_capture_and_injection():
    """Verify ScreenCaptureEngine captures normalized [1, 3, 128, 128] frames without crashing."""
    engine = ScreenCaptureEngine()

    # 1. Default capture (X11 live or simulated fallback)
    frame = engine.capture_frame()
    assert isinstance(frame, ScreenFrame)
    assert frame.tensor.shape == (1, 3, 128, 128)
    assert frame.tensor.min() >= 0.0
    assert frame.tensor.max() <= 1.0
    assert frame.source in ("x11_screen", "virtual_buffer")

    # 2. Injected custom frame
    custom_tensor = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)
    engine.inject_frame(custom_tensor)
    injected_frame = engine.capture_frame()
    assert injected_frame.source == "injected_feed"
    assert injected_frame.tensor.shape == (1, 3, 128, 128)
    assert pytest.approx(injected_frame.mean_luminance, rel=1e-2) == 0.5

    # Reset injection
    engine.inject_frame(None)


def test_screen_capture_modes():
    """Verify high-fidelity simulated frame synthesis across active desktop contexts."""
    engine = ScreenCaptureEngine()

    for mode in ("reading_book", "coding_ide", "watching_video", "ui_design_exploration", "system_idle"):
        engine.set_simulated_mode(mode)
        f = engine.capture_frame()
        assert f.tensor.shape == (1, 3, 128, 128)
        assert f.mean_luminance > 0.0
        assert f.contrast_metric >= 0.0


def test_visual_design_learner_from_frame():
    """Verify VisualDesignLearner extracts color palettes, typography, layout, and proves WCAG AAA."""
    eye = ContinuousVisualEye.get_instance()
    engine = eye.capture_engine
    learner = eye.design_learner
    store = DesignKnowledgeStore.get_instance()
    store.clear()

    # Capture a simulated video / UI frame with high contrast
    engine.set_simulated_mode("watching_video")
    frame = engine.capture_frame()

    archetype = learner.learn_from_frame(frame, source_label="youtube_tutorial")

    assert isinstance(archetype, LearnedDesignArchetype)
    assert archetype.source_label == "youtube_tutorial"
    assert 0.0 <= archetype.primary_hue <= 360.0
    assert archetype.palette["primary"].startswith("#")
    assert archetype.palette["background"].startswith("#")
    assert "radial-gradient" in archetype.synthesized_css

    # Formally verified WCAG 2.1 AAA contrast
    assert archetype.wcag_contrast_ratio >= 7.0
    assert archetype.wcag_aaa_compliant is True

    # Golden ratio typography and spacing
    assert len(archetype.typography_scale) > 0
    assert len(archetype.spatial_spacing) > 0
    assert any("rem" in t for t in archetype.typography_scale)
    assert any("px" in s for s in archetype.spatial_spacing)

    # Crystallized in persistent store
    assert len(store.list_archetypes()) == 1
    assert store.get_latest_archetype() == archetype


def test_continuous_visual_eye_activity_observation():
    """Verify ContinuousVisualEye classifies active user context and generates companion guidance."""
    eye = ContinuousVisualEye.get_instance()
    engine = eye.capture_engine

    # 1. Reading Book Observation
    engine.set_simulated_mode("reading_book")
    ctx_reading = eye.observe_screen(task_context="reading a physics book")
    assert isinstance(ctx_reading, ObservedVisualContext)
    assert "reading" in ctx_reading.companion_assistance.lower()
    assert len(ctx_reading.detected_entities) > 0

    # 2. Coding IDE Observation
    engine.set_simulated_mode("coding_ide")
    ctx_coding = eye.observe_screen(task_context="writing python algorithm in vscode")
    assert isinstance(ctx_coding, ObservedVisualContext)
    assert any(term in ctx_coding.companion_assistance.lower() for term in ("coding", "verifier", "invariants"))

    # 3. Watching Video / YouTube Observation
    engine.set_simulated_mode("watching_video")
    ctx_video = eye.observe_screen(task_context="watching youtube web design tutorial")
    assert isinstance(ctx_video, ObservedVisualContext)
    assert any(term in ctx_video.companion_assistance.lower() for term in ("video", "youtube", "design"))


def test_continuous_visual_eye_background_daemon():
    """Verify non-blocking continuous observer daemon starts, monitors, and stops cleanly."""
    eye = ContinuousVisualEye.get_instance()

    assert eye.is_observer_running() is False

    eye.start_background_observer(interval=0.1)
    assert eye.is_observer_running() is True

    # Give daemon time to perform at least one observation tick
    time.sleep(0.35)

    latest = eye.get_latest_observation()
    assert latest is not None
    assert isinstance(latest, ObservedVisualContext)

    eye.stop_background_observer()
    assert eye.is_observer_running() is False


def test_mathematical_style_engine_dynamic_background_variation():
    """Verify MathematicalStyleEngine synthesizes distinct, dynamic backgrounds across different intents."""
    # Intent 1: Physics simulation portfolio
    rules_physics = MathematicalStyleEngine.synthesize_rules(
        task_description="quantum physics and manifold energy simulation"
    )
    css_physics = "\n".join(r.render() for r in rules_physics)

    # Intent 2: Minimalist artistic portfolio
    rules_art = MathematicalStyleEngine.synthesize_rules(
        task_description="organic nature floral watercolor visual portfolio"
    )
    css_art = "\n".join(r.render() for r in rules_art)

    # Verify both contain mathematical radial gradients
    assert "radial-gradient" in css_physics
    assert "radial-gradient" in css_art

    # Extract body rule declarations
    def get_decl(rules, sel, prop):
        for r in rules:
            if r.selector == sel:
                for d in r.declarations:
                    if d.property_name == prop:
                        return d.property_value
        return None

    bg_phys = get_decl(rules_physics, ":root", "--bg-deep")
    bg_art = get_decl(rules_art, ":root", "--bg-deep")

    assert bg_phys is not None
    assert bg_art is not None
    # Verify different intents yield different dynamic backgrounds
    assert bg_phys != bg_art
