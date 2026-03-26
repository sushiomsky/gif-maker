"""Smoke-tests for gif-maker.py.

The module uses a hyphen in its filename, so it is loaded via importlib
rather than a normal import statement.
"""
import importlib.util
import os
import subprocess
import sys

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Load the module
# ---------------------------------------------------------------------------
_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gif-maker.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("gif_maker", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gm = _load_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_img():
    """32×32 red RGBA image — small enough for fast tests."""
    return Image.new("RGBA", (32, 32), (200, 50, 50, 255))


@pytest.fixture
def img_path(tmp_path, small_img):
    p = tmp_path / "input.png"
    small_img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# Effects — every effect must return exactly n RGBA frames of the same size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(gm.EFFECTS.keys()))
def test_effect_returns_n_rgba_frames(small_img, key):
    fn = gm.EFFECTS[key][1]
    n = 4
    frames = fn(small_img, n)
    assert len(frames) == n, f"{key}: expected {n} frames, got {len(frames)}"
    for f in frames:
        assert f.mode == "RGBA", f"{key}: frame mode is {f.mode}"
        assert f.size == small_img.size, f"{key}: size mismatch"


# ---------------------------------------------------------------------------
# Overlays — must not change frame count or dimensions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(gm.OVERLAYS.keys()))
def test_overlay_preserves_frame_count_and_size(small_img, key):
    frames = [small_img.copy() for _ in range(4)]
    original_size = frames[0].size
    fn = gm.OVERLAYS[key][1]
    fn(frames)
    assert len(frames) == 4, f"{key}: overlay changed frame count"
    for f in frames:
        assert f.size == original_size, f"{key}: overlay changed frame size"


# ---------------------------------------------------------------------------
# Text animations
# ---------------------------------------------------------------------------
def test_apply_text_all_styles(small_img):
    for style in gm.TEXT_STYLES:
        frames = [small_img.copy() for _ in range(4)]
        gm.apply_text(frames, "HI", style=style)  # must not raise


# ---------------------------------------------------------------------------
# save_optimized — output must exist and be ≤ 2 MB
# ---------------------------------------------------------------------------
def test_save_optimized_under_2mb(small_img, tmp_path):
    frames = [small_img.copy() for _ in range(3)]
    out = str(tmp_path / "out.gif")
    sz, side, nf, cols = gm.save_optimized(frames, out)
    assert os.path.exists(out)
    assert sz <= 2_000_000


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def test_generate_default(img_path, tmp_path):
    cfg = gm.default_cfg()
    cfg["source_path"] = img_path
    cfg["output_path"] = str(tmp_path / "out.gif")
    cfg["n_frames"] = 3
    out, sz = gm.generate(cfg)
    assert os.path.exists(out)
    assert sz > 0


def test_generate_all_effects(img_path, tmp_path):
    for key in gm.EFFECTS:
        cfg = gm.default_cfg()
        cfg["source_path"] = img_path
        cfg["output_path"] = str(tmp_path / f"out_{key}.gif")
        cfg["effect"] = key
        cfg["n_frames"] = 2
        out, sz = gm.generate(cfg)
        assert os.path.exists(out), f"effect '{key}' did not produce output"


def test_generate_with_text_and_overlays(img_path, tmp_path):
    cfg = gm.default_cfg()
    cfg["source_path"] = img_path
    cfg["output_path"] = str(tmp_path / "out_combo.gif")
    cfg["effect"] = "glitch"
    cfg["overlays"] = ["sparkles", "snow"]
    cfg["text"] = "TEST"
    cfg["text_style"] = "bounce"
    cfg["n_frames"] = 3
    out, sz = gm.generate(cfg)
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Preset system — save / load round-trip
# ---------------------------------------------------------------------------
def test_preset_round_trip(tmp_path, monkeypatch):
    preset_file = str(tmp_path / "presets.json")
    monkeypatch.setattr(gm, "PRESETS_FILE", preset_file)

    cfg = gm.default_cfg()
    cfg["effect"] = "neon"
    cfg["overlays"] = ["sparkles"]
    gm.save_preset("my_preset", cfg)

    loaded = gm.load_presets()
    assert "my_preset" in loaded
    assert loaded["my_preset"]["effect"] == "neon"
    assert loaded["my_preset"]["overlays"] == ["sparkles"]
    # source_path must not be persisted
    assert "source_path" not in loaded["my_preset"]


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------
def _run(*args):
    return subprocess.run(
        [sys.executable, _SCRIPT, *args],
        capture_output=True,
        text=True,
    )


def test_cli_list_effects():
    r = _run("--list-effects")
    assert r.returncode == 0
    assert "glitch" in r.stdout
    assert "neon" in r.stdout


def test_cli_list_overlays():
    r = _run("--list-overlays")
    assert r.returncode == 0
    assert "sparkles" in r.stdout


def test_cli_version():
    r = _run("--version")
    assert r.returncode == 0
    assert "1.0" in r.stdout or "1.0" in r.stderr


def test_cli_no_image_non_interactive_exits_nonzero():
    r = _run("-n")
    assert r.returncode != 0


def test_cli_generate(img_path, tmp_path):
    out = str(tmp_path / "result.gif")
    r = _run(img_path, "-e", "none", "-n", "-O", out)
    assert r.returncode == 0
    assert os.path.exists(out)


def test_cli_generate_with_effect_and_overlay(img_path, tmp_path):
    out = str(tmp_path / "result2.gif")
    r = _run(img_path, "-e", "glitch", "-o", "sparkles", "--frames", "3", "-n", "-O", out)
    assert r.returncode == 0
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Sticker system
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(gm.STICKERS.keys()))
def test_builtin_sticker_returns_n_rgba_frames(key):
    fn = gm.STICKERS[key][1]
    n = 4
    frames = fn(n, 40)
    assert len(frames) == n, f"{key}: expected {n} frames"
    for f in frames:
        assert f.mode == "RGBA", f"{key}: frame mode is {f.mode}"
        assert f.size == (40, 40), f"{key}: unexpected size {f.size}"


@pytest.mark.parametrize("key", list(gm.STICKERS.keys()))
def test_apply_sticker_does_not_change_frame_count_or_size(small_img, key):
    frames = [small_img.copy() for _ in range(4)]
    original_size = frames[0].size
    gm.apply_sticker(frames, key, position="center", scale=0.4)
    assert len(frames) == 4, f"{key}: sticker changed frame count"
    for f in frames:
        assert f.size == original_size, f"{key}: sticker changed frame size"


@pytest.mark.parametrize("pos", gm.STICKER_POSITIONS)
def test_apply_sticker_all_positions(small_img, pos):
    frames = [small_img.copy() for _ in range(2)]
    gm.apply_sticker(frames, "star", position=pos, scale=0.3)
    assert len(frames) == 2


def test_apply_sticker_unknown_key_does_not_raise(small_img):
    frames = [small_img.copy() for _ in range(2)]
    gm.apply_sticker(frames, "nonexistent_sticker_xyz")
    assert len(frames) == 2


def test_apply_sticker_from_png_file(small_img, tmp_path):
    # Save a small transparent PNG and use it as a sticker
    sticker_file = str(tmp_path / "my_sticker.png")
    Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(sticker_file)
    frames = [small_img.copy() for _ in range(3)]
    gm.apply_sticker(frames, sticker_file, position="center", scale=0.4)
    assert len(frames) == 3


def test_generate_with_sticker(img_path, tmp_path):
    cfg = gm.default_cfg()
    cfg["source_path"] = img_path
    cfg["output_path"] = str(tmp_path / "out_sticker.gif")
    cfg["n_frames"] = 3
    cfg["stickers"] = [{"key": "heart", "position": "center", "scale": 0.4}]
    out, sz = gm.generate(cfg)
    assert os.path.exists(out)
    assert sz > 0


def test_cli_list_stickers():
    r = _run("--list-stickers")
    assert r.returncode == 0
    assert "star" in r.stdout
    assert "heart" in r.stdout


def test_cli_generate_with_sticker(img_path, tmp_path):
    out = str(tmp_path / "result_sticker.gif")
    r = _run(img_path, "--sticker", "star", "--sticker-pos", "top-right",
             "--sticker-scale", "0.3", "--frames", "3", "-n", "-O", out)
    assert r.returncode == 0
    assert os.path.exists(out)


def test_sticker_preset_round_trip(tmp_path, monkeypatch):
    preset_file = str(tmp_path / "presets.json")
    monkeypatch.setattr(gm, "PRESETS_FILE", preset_file)

    cfg = gm.default_cfg()
    cfg["stickers"] = [{"key": "crown", "position": "top-left", "scale": 0.25}]
    gm.save_preset("sticker_preset", cfg)

    loaded = gm.load_presets()
    assert "sticker_preset" in loaded
    assert loaded["sticker_preset"]["stickers"][0]["key"] == "crown"
    assert "source_path" not in loaded["sticker_preset"]
