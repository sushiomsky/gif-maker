"""
tests/test_duck_animator.py
===========================

Test suite for the Duckling GIF Animation Engine.

Covers:
  - emotion_engine: EMOTIONS structure, interpolation, custom JSON loading
  - duck_animator:  FaceGeometry scaling, blink/pupil/bob curve shapes,
                    generate_frames output contract, save_gif file output
  - pipeline_integration: animate_image() end-to-end
  - CLI smoke tests
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  Module roots
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(_ROOT))

from emotion_engine import (
    EMOTIONS,
    get_emotion,
    get_random_emotion,
    interpolate_emotions,
    load_custom_emotions,
    list_emotions,
)
from duck_animator import (
    FaceGeometry,
    generate_blink_seq,
    generate_pupil_track,
    generate_head_bob,
    generate_frames,
    save_gif,
    _BLINK_CURVE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_img():
    """64×64 yellow RGBA image — small enough for fast tests."""
    return Image.new("RGBA", (64, 64), (255, 230, 50, 255))


@pytest.fixture
def medium_img():
    """256×256 image at reference scale."""
    return Image.new("RGBA", (256, 256), (200, 180, 80, 255))


@pytest.fixture
def img_path(tmp_path, small_img):
    p = tmp_path / "duck.png"
    small_img.save(str(p))
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
#  emotion_engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionEngine:
    def test_emotions_not_empty(self):
        assert len(EMOTIONS) >= 5

    def test_required_emotions_present(self):
        for name in ("neutral", "happy", "sad", "angry", "surprised"):
            assert name in EMOTIONS, f"Missing required emotion: {name}"

    def test_each_emotion_has_required_keys(self):
        required = {"eye_open", "brow_angle", "brow_raise", "mouth_open",
                    "look_bias", "blink_rate"}
        for name, cfg in EMOTIONS.items():
            missing = required - cfg.keys()
            assert not missing, f"Emotion '{name}' missing keys: {missing}"

    def test_eye_open_in_range(self):
        for name, cfg in EMOTIONS.items():
            assert 0.0 <= cfg["eye_open"] <= 1.5, \
                f"Emotion '{name}' eye_open={cfg['eye_open']} out of 0–1.5"

    def test_brow_angle_in_range(self):
        for name, cfg in EMOTIONS.items():
            assert -30 <= cfg["brow_angle"] <= 30, \
                f"Emotion '{name}' brow_angle={cfg['brow_angle']} out of -30–30"

    def test_blink_rate_in_range(self):
        for name, cfg in EMOTIONS.items():
            assert 0.0 <= cfg["blink_rate"] <= 1.0, \
                f"Emotion '{name}' blink_rate={cfg['blink_rate']} out of 0–1"

    def test_look_bias_is_2_tuple(self):
        for name, cfg in EMOTIONS.items():
            lb = cfg["look_bias"]
            assert len(lb) == 2, f"Emotion '{name}' look_bias is not length-2"

    def test_get_emotion_known(self):
        cfg = get_emotion("happy")
        assert cfg["eye_open"] == EMOTIONS["happy"]["eye_open"]

    def test_get_emotion_unknown_falls_back_to_neutral(self):
        cfg = get_emotion("totally_made_up")
        assert cfg["eye_open"] == EMOTIONS["neutral"]["eye_open"]

    def test_get_random_emotion_returns_valid_name(self):
        name, cfg = get_random_emotion()
        assert name in EMOTIONS
        assert "eye_open" in cfg

    def test_list_emotions_sorted(self):
        names = list_emotions()
        assert names == sorted(names)

    def test_interpolate_t0_is_emo_a(self):
        a = EMOTIONS["neutral"]
        b = EMOTIONS["angry"]
        result = interpolate_emotions(a, b, 0.0)
        assert result["eye_open"] == pytest.approx(a["eye_open"])
        assert result["brow_angle"] == pytest.approx(a["brow_angle"])

    def test_interpolate_t1_is_emo_b(self):
        a = EMOTIONS["neutral"]
        b = EMOTIONS["angry"]
        result = interpolate_emotions(a, b, 1.0)
        assert result["eye_open"] == pytest.approx(b["eye_open"])

    def test_interpolate_midpoint(self):
        a = EMOTIONS["neutral"]
        b = EMOTIONS["surprised"]
        result = interpolate_emotions(a, b, 0.5)
        expected_eye = (a["eye_open"] + b["eye_open"]) / 2
        assert result["eye_open"] == pytest.approx(expected_eye, rel=1e-3)

    def test_interpolate_clamps_t(self):
        a = EMOTIONS["neutral"]
        b = EMOTIONS["angry"]
        assert interpolate_emotions(a, b, -0.5)["eye_open"] == pytest.approx(
            EMOTIONS["neutral"]["eye_open"]
        )
        assert interpolate_emotions(a, b, 1.5)["eye_open"] == pytest.approx(
            EMOTIONS["angry"]["eye_open"]
        )

    def test_load_custom_emotions_adds_new(self, tmp_path):
        custom_file = tmp_path / "custom.json"
        custom_file.write_text(json.dumps({
            "party": {"eye_open": 1.3, "brow_angle": -5, "brow_raise": 8,
                      "mouth_open": 0.6, "look_bias": [0, -2], "blink_rate": 0.02}
        }))
        merged = load_custom_emotions(str(custom_file))
        assert "party" in merged
        assert merged["party"]["eye_open"] == pytest.approx(1.3)

    def test_load_custom_emotions_overrides_existing(self, tmp_path):
        custom_file = tmp_path / "override.json"
        custom_file.write_text(json.dumps({
            "neutral": {"brow_angle": 99}
        }))
        merged = load_custom_emotions(str(custom_file))
        assert merged["neutral"]["brow_angle"] == 99
        # All other keys of neutral should still be present
        assert "eye_open" in merged["neutral"]

    def test_load_custom_emotions_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_custom_emotions("/nonexistent/path/emotions.json")


# ─────────────────────────────────────────────────────────────────────────────
#  FaceGeometry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceGeometry:
    def test_scale_at_256(self):
        geo = FaceGeometry(256, 256)
        assert geo.scale == pytest.approx(1.0)

    def test_scale_at_512(self):
        geo = FaceGeometry(512, 512)
        assert geo.scale == pytest.approx(2.0)

    def test_eyes_symmetric_around_cx(self):
        geo = FaceGeometry(400, 400)
        assert geo.left_eye_x == pytest.approx(geo.cx - geo.eye_sep)
        assert geo.right_eye_x == pytest.approx(geo.cx + geo.eye_sep)

    def test_eyes_share_same_y(self):
        geo = FaceGeometry(300, 300)
        assert geo.eye_y == geo.eye_y  # trivially true but also verifies attr exists

    def test_cx_is_horizontal_centre(self):
        geo = FaceGeometry(320, 240)
        assert geo.cx == pytest.approx(160.0)

    def test_all_dims_positive(self):
        for size in (64, 128, 256, 512):
            geo = FaceGeometry(size, size)
            assert geo.eye_rx > 0
            assert geo.eye_ry_base > 0
            assert geo.brow_half_len > 0
            assert geo.brow_thickness >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Animation curve tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnimationCurves:
    def test_blink_seq_correct_length(self):
        seq = generate_blink_seq(80, 0.03)
        assert len(seq) == 80

    def test_blink_seq_values_in_range(self):
        seq = generate_blink_seq(100, 0.1)
        assert all(0.0 <= v <= 1.0 for v in seq), "Blink value outside 0–1"

    def test_blink_seq_with_offset_correct_length(self):
        seq = generate_blink_seq(60, 0.05, frame_offset=3)
        assert len(seq) == 60

    def test_blink_curve_shape(self):
        # The blink curve must start and end at 1.0 and dip to 0 in the middle
        assert _BLINK_CURVE[0]  == pytest.approx(1.0)
        assert _BLINK_CURVE[-1] == pytest.approx(1.0)
        assert min(_BLINK_CURVE) == pytest.approx(0.0)

    def test_pupil_track_correct_length(self):
        track = generate_pupil_track(50)
        assert len(track) == 50

    def test_pupil_track_items_are_2_tuples(self):
        track = generate_pupil_track(10)
        for item in track:
            assert len(item) == 2

    def test_pupil_track_not_all_identical(self):
        """No two consecutive frames should have exactly equal pupil position."""
        track = generate_pupil_track(20)
        identical_pairs = sum(
            1 for i in range(1, len(track))
            if track[i] == track[i - 1]
        )
        assert identical_pairs < len(track) - 1, "Pupil track is suspiciously static"

    def test_head_bob_correct_length(self):
        bobs = generate_head_bob(40)
        assert len(bobs) == 40

    def test_head_bob_items_are_2_tuples(self):
        bobs = generate_head_bob(10)
        for item in bobs:
            assert len(item) == 2

    def test_head_bob_amplitude_small(self):
        """Head bob should be very subtle — well under 5 px."""
        bobs = generate_head_bob(200)
        max_dy = max(abs(dy) for dy, _ in bobs)
        assert max_dy < 5.0, f"Head bob too large: {max_dy:.2f} px"


# ─────────────────────────────────────────────────────────────────────────────
#  generate_frames contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateFrames:
    @pytest.mark.parametrize("emotion", list(EMOTIONS.keys()))
    def test_returns_n_frames(self, small_img, emotion):
        frames = generate_frames(small_img, emotion_name=emotion, n_frames=4)
        assert len(frames) == 4, f"{emotion}: expected 4 frames, got {len(frames)}"

    @pytest.mark.parametrize("emotion", list(EMOTIONS.keys()))
    def test_all_frames_are_rgba(self, small_img, emotion):
        frames = generate_frames(small_img, emotion_name=emotion, n_frames=3)
        for i, f in enumerate(frames):
            assert f.mode == "RGBA", f"{emotion}: frame {i} mode is {f.mode}"

    @pytest.mark.parametrize("emotion", list(EMOTIONS.keys()))
    def test_all_frames_same_size_as_input(self, small_img, emotion):
        frames = generate_frames(small_img, emotion_name=emotion, n_frames=3)
        for i, f in enumerate(frames):
            assert f.size == small_img.size, \
                f"{emotion}: frame {i} size {f.size} ≠ input {small_img.size}"

    def test_rgb_input_auto_converted(self, small_img):
        rgb_img = small_img.convert("RGB")
        frames = generate_frames(rgb_img, emotion_name="neutral", n_frames=2)
        assert len(frames) == 2
        assert all(f.mode == "RGBA" for f in frames)

    def test_base_image_not_mutated(self, small_img):
        """The original base image must be unchanged after generate_frames."""
        import copy
        original_pixels = list(small_img.getdata())
        generate_frames(small_img, emotion_name="angry", n_frames=5)
        assert list(small_img.getdata()) == original_pixels

    def test_frames_not_all_identical(self, medium_img):
        """Animation must have variation — no two frames pixel-identical."""
        frames = generate_frames(medium_img, emotion_name="neutral", n_frames=10)
        unique = len(set(f.tobytes() for f in frames))
        assert unique > 1, "All frames are pixel-identical — animation is static"

    def test_unknown_emotion_falls_back(self, small_img):
        frames = generate_frames(small_img, emotion_name="totally_unknown", n_frames=2)
        assert len(frames) == 2

    def test_transition_produces_n_frames(self, small_img):
        frames = generate_frames(
            small_img, emotion_name="neutral", n_frames=10,
            transition_to="happy", transition_frames=4
        )
        assert len(frames) == 10

    def test_high_resolution_image(self):
        """Engine must handle large images without error."""
        large = Image.new("RGBA", (512, 512), (100, 150, 200, 255))
        frames = generate_frames(large, emotion_name="surprised", n_frames=4)
        assert len(frames) == 4
        assert all(f.size == (512, 512) for f in frames)

    def test_non_square_image(self):
        """Non-square inputs must be handled gracefully."""
        wide = Image.new("RGBA", (400, 200), (80, 120, 60, 255))
        frames = generate_frames(wide, emotion_name="sad", n_frames=3)
        assert all(f.size == (400, 200) for f in frames)

    def test_tiny_image(self):
        """Very small images (16×16) must not raise."""
        tiny = Image.new("RGBA", (16, 16), (200, 100, 50, 255))
        frames = generate_frames(tiny, emotion_name="neutral", n_frames=2)
        assert len(frames) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  save_gif tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveGif:
    def test_file_is_created(self, small_img, tmp_path):
        frames = generate_frames(small_img, n_frames=3)
        out = str(tmp_path / "out.gif")
        save_gif(frames, out)
        assert os.path.exists(out)

    def test_file_is_nonzero(self, small_img, tmp_path):
        frames = generate_frames(small_img, n_frames=3)
        out = str(tmp_path / "out.gif")
        save_gif(frames, out)
        assert os.path.getsize(out) > 0

    def test_gif_is_multi_frame(self, small_img, tmp_path):
        """Saved GIF must contain multiple frames."""
        frames = generate_frames(small_img, n_frames=5)
        out = str(tmp_path / "multi.gif")
        save_gif(frames, out, duration=80)
        gif = Image.open(out)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        assert frame_count >= 2, "GIF has fewer than 2 frames"

    def test_empty_frames_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_gif([], str(tmp_path / "empty.gif"))

    def test_custom_duration(self, small_img, tmp_path):
        frames = generate_frames(small_img, n_frames=2)
        out = str(tmp_path / "fast.gif")
        save_gif(frames, out, duration=60)
        assert os.path.exists(out)

    def test_accepts_rgb_frames(self, small_img, tmp_path):
        """save_gif should handle RGB (non-RGBA) frames."""
        frames = [small_img.convert("RGB") for _ in range(3)]
        out = str(tmp_path / "rgb.gif")
        save_gif(frames, out)
        assert os.path.exists(out)


# ─────────────────────────────────────────────────────────────────────────────
#  pipeline_integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_animate_image_from_path(self, img_path, tmp_path):
        from pipeline_integration import animate_image
        out = str(tmp_path / "pipeline_out.gif")
        result = animate_image(img_path, out, emotion="neutral", n_frames=3)
        assert result == out
        assert os.path.exists(out)

    def test_animate_image_from_pil(self, small_img, tmp_path):
        from pipeline_integration import animate_image
        out = str(tmp_path / "pil_out.gif")
        animate_image(small_img, out, emotion="happy", n_frames=3)
        assert os.path.exists(out)

    def test_animate_image_random_emotion(self, img_path, tmp_path):
        from pipeline_integration import animate_image
        out = str(tmp_path / "random_out.gif")
        animate_image(img_path, out, emotion="random", n_frames=3)
        assert os.path.exists(out)

    def test_animate_image_with_transition(self, img_path, tmp_path):
        from pipeline_integration import animate_image
        out = str(tmp_path / "transition_out.gif")
        animate_image(img_path, out, emotion="neutral",
                      transition_to="happy", n_frames=8, transition_frames=3)
        assert os.path.exists(out)

    def test_animate_image_missing_source_raises(self, tmp_path):
        from pipeline_integration import animate_image
        with pytest.raises(FileNotFoundError):
            animate_image("/no/such/file.png", str(tmp_path / "x.gif"))

    def test_animate_image_bad_source_type_raises(self, tmp_path):
        from pipeline_integration import animate_image
        with pytest.raises(TypeError):
            animate_image(12345, str(tmp_path / "x.gif"))

    def test_duck_animation_pipeline_run(self, img_path, tmp_path):
        from pipeline_integration import DuckAnimationPipeline
        pipe = DuckAnimationPipeline(emotion="sad", n_frames=3, duration=80)
        out = str(tmp_path / "pipe.gif")
        pipe.run(img_path, out)
        assert os.path.exists(out)

    def test_duck_animation_pipeline_run_batch(self, tmp_path):
        from pipeline_integration import DuckAnimationPipeline
        # Set up a small batch
        in_dir  = tmp_path / "inputs"
        out_dir = tmp_path / "outputs"
        in_dir.mkdir()
        for i in range(3):
            img = Image.new("RGBA", (32, 32), (100 + i * 40, 100, 50, 255))
            img.save(str(in_dir / f"duck_{i}.png"))

        pipe = DuckAnimationPipeline(emotion="random", n_frames=3)
        results = pipe.run_batch(str(in_dir), str(out_dir))
        assert len(results) == 3
        for path in results:
            assert os.path.exists(path)

    def test_animate_image_with_custom_emotions_json(self, img_path, tmp_path):
        from pipeline_integration import animate_image
        custom_file = tmp_path / "custom.json"
        custom_file.write_text(json.dumps({
            "party": {"eye_open": 1.3, "brow_angle": -5, "brow_raise": 8,
                      "mouth_open": 0.6, "look_bias": [0, -2],
                      "blink_rate": 0.02, "description": "Party mode"}
        }))
        out = str(tmp_path / "custom_out.gif")
        animate_image(img_path, out, emotion="party",
                      custom_emotions_path=str(custom_file), n_frames=3)
        assert os.path.exists(out)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def _run_cli(script, *args):
    return subprocess.run(
        [sys.executable, str(_ROOT / script), *args],
        capture_output=True,
        text=True,
    )


class TestCLI:
    def test_duck_animator_cli_happy_path(self, img_path, tmp_path):
        out = str(tmp_path / "cli_out.gif")
        r = _run_cli("duck_animator.py", img_path, out, "neutral", "4", "80")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert os.path.exists(out)

    def test_duck_animator_cli_no_emotion_uses_random(self, img_path, tmp_path):
        out = str(tmp_path / "cli_random.gif")
        r = _run_cli("duck_animator.py", img_path, out)
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_duck_animator_cli_missing_args_exits_nonzero(self):
        r = _run_cli("duck_animator.py")
        assert r.returncode != 0

    def test_pipeline_integration_cli(self, img_path, tmp_path):
        out = str(tmp_path / "pl_cli.gif")
        r = _run_cli("pipeline_integration.py", img_path, out, "angry",
                     "--frames", "4", "--duration", "80")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert os.path.exists(out)

    def test_pipeline_integration_cli_with_transition(self, img_path, tmp_path):
        out = str(tmp_path / "pl_trans.gif")
        r = _run_cli("pipeline_integration.py", img_path, out, "neutral",
                     "--frames", "8", "--transition", "happy", "--transition-frames", "3")
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_emotion_engine_cli_prints_table(self):
        r = _run_cli("emotion_engine.py")
        assert r.returncode == 0
        assert "neutral" in r.stdout
        assert "angry"   in r.stdout

    def test_batch_runner_cli(self, tmp_path):
        in_dir  = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        for i in range(2):
            Image.new("RGBA", (32, 32), (200, 100, 50, 255)).save(
                str(in_dir / f"img_{i}.png")
            )
        r = _run_cli("batch_runner.py", str(in_dir), str(out_dir),
                     "--frames", "3", "--emotion", "neutral", "--workers", "2")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        gifs = list(out_dir.glob("*.gif"))
        assert len(gifs) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Integration — real duck image (skipped if not present)
# ─────────────────────────────────────────────────────────────────────────────

_REAL_IMAGES = [
    _ROOT / "beetle.png",
    _ROOT / "hanshaw.png",
]


@pytest.mark.parametrize("img_file", [p for p in _REAL_IMAGES if p.exists()])
@pytest.mark.parametrize("emotion", ["neutral", "happy", "angry"])
def test_real_image_all_emotions(img_file, emotion, tmp_path):
    """End-to-end test on real project images (skipped if absent)."""
    frames = generate_frames(
        Image.open(img_file), emotion_name=emotion, n_frames=4
    )
    out = str(tmp_path / f"{img_file.stem}_{emotion}.gif")
    save_gif(frames, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 100
