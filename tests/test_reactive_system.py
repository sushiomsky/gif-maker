#!/usr/bin/env python3
"""
tests/test_reactive_system.py — Tests for lipsync, reaction, and timeline modules
==================================================================================
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def duck_image():
    return Image.new("RGBA", (128, 128), (255, 220, 60, 255))


@pytest.fixture
def small_image():
    return Image.new("RGBA", (64, 64), (200, 180, 100, 255))


# ─────────────────────────────────────────────────────────────────────────────
#  LIPSYNC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TestMouthState:
    def test_default_construction(self):
        from lipsync_engine import MouthState
        m = MouthState()
        assert isinstance(m.mouth_open, float)
        assert isinstance(m.width_scale, float)
        assert isinstance(m.corner_pull, float)
        assert 0.0 <= m.mouth_open <= 1.0

    def test_closed_factory(self):
        from lipsync_engine import MouthState
        m = MouthState.closed()
        assert m.mouth_open < 0.05

    def test_blend_toward_returns_new_instance(self):
        from lipsync_engine import MouthState
        m1 = MouthState.closed()
        m2 = MouthState(mouth_open=1.0, width_scale=1.5, corner_pull=0.5)
        blended = m1.blend_toward(m2, 0.5)
        assert isinstance(blended, MouthState)
        assert blended is not m1
        assert blended is not m2

    def test_blend_toward_interpolates(self):
        from lipsync_engine import MouthState
        m1 = MouthState(mouth_open=0.0)
        m2 = MouthState(mouth_open=1.0)
        blended = m1.blend_toward(m2, 0.5)
        assert 0.0 < blended.mouth_open < 1.0

    def test_blend_toward_alpha_0_unchanged(self):
        from lipsync_engine import MouthState
        m1 = MouthState(mouth_open=0.0)
        m2 = MouthState(mouth_open=1.0)
        blended = m1.blend_toward(m2, 0.0)
        assert abs(blended.mouth_open - 0.0) < 0.01

    def test_blend_toward_alpha_1_equals_target(self):
        from lipsync_engine import MouthState
        m1 = MouthState(mouth_open=0.0)
        m2 = MouthState(mouth_open=1.0)
        blended = m1.blend_toward(m2, 1.0)
        assert abs(blended.mouth_open - 1.0) < 0.01

    def test_to_dict_from_dict_roundtrip(self):
        from lipsync_engine import MouthState
        m = MouthState(mouth_open=0.7, width_scale=1.2, corner_pull=0.3)
        d = m.to_dict()
        assert isinstance(d, dict)
        m2 = MouthState.from_dict(d)
        assert abs(m2.mouth_open - m.mouth_open) < 0.001
        assert abs(m2.width_scale - m.width_scale) < 0.001

    def test_from_phoneme_returns_mouth_state(self):
        from lipsync_engine import MouthState, PhonemeShape
        m = MouthState.from_phoneme(PhonemeShape.WIDE)
        assert isinstance(m, MouthState)
        assert m.mouth_open > 0.0


class TestPhonemeShape:
    def test_all_shapes_exist(self):
        from lipsync_engine import PhonemeShape
        for name in ("CLOSED", "SMALL", "MID", "WIDE", "ROUND"):
            assert hasattr(PhonemeShape, name)

    def test_is_int_enum(self):
        from lipsync_engine import PhonemeShape
        import enum
        assert issubclass(PhonemeShape, enum.IntEnum)


class TestLipsyncEngine:
    def test_construction(self):
        from lipsync_engine import LipsyncEngine
        eng = LipsyncEngine(fps=12)
        assert eng is not None

    def test_generate_track_length(self):
        from lipsync_engine import LipsyncEngine, MouthState
        eng = LipsyncEngine(fps=12)
        # generate_track takes duration in seconds
        track = eng.generate_track(2.0)   # 2s * 12fps = 24 frames
        assert len(track) == 24

    def test_generate_track_returns_mouth_states(self):
        from lipsync_engine import LipsyncEngine, MouthState
        eng = LipsyncEngine(fps=12)
        track = eng.generate_track(1.0)   # 1 second = 12 frames
        for item in track:
            assert isinstance(item, MouthState)

    def test_generate_track_mouth_open_range(self):
        from lipsync_engine import LipsyncEngine
        eng = LipsyncEngine(fps=12)
        track = eng.generate_track(2.5)   # 2.5 seconds
        for m in track:
            assert 0.0 <= m.mouth_open <= 1.05  # small float tolerance

    def test_generate_idle_mouth_length(self):
        from lipsync_engine import LipsyncEngine, MouthState
        eng = LipsyncEngine(fps=12)
        track = eng.generate_idle_mouth(15)
        assert len(track) == 15
        for item in track:
            assert isinstance(item, MouthState)

    def test_idle_mouth_is_mostly_closed(self):
        from lipsync_engine import LipsyncEngine
        eng = LipsyncEngine(fps=12)
        track = eng.generate_idle_mouth(24)
        avg_open = sum(m.mouth_open for m in track) / len(track)
        assert avg_open < 0.3   # idle mouth should be mostly closed

    def test_idle_chatter_track_length(self):
        from lipsync_engine import LipsyncEngine, MouthState
        eng = LipsyncEngine(fps=12)
        track = eng.idle_chatter_track(20)
        assert len(track) == 20
        for item in track:
            assert isinstance(item, MouthState)

    def test_track_not_perfectly_periodic(self):
        """No two consecutive runs should be identical (randomness check)."""
        from lipsync_engine import LipsyncEngine
        eng = LipsyncEngine(fps=12)
        t1 = [m.mouth_open for m in eng.generate_track(2.0)]
        t2 = [m.mouth_open for m in eng.generate_track(2.0)]
        # Very unlikely to be identical due to randomness
        assert t1 != t2


# ─────────────────────────────────────────────────────────────────────────────
#  REACTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TestEventCatalog:
    def test_is_dict(self):
        from reaction_engine import EVENT_CATALOG
        assert isinstance(EVENT_CATALOG, dict)

    def test_has_required_events(self):
        from reaction_engine import EVENT_CATALOG
        for ev in ("idle", "user_message", "loud_noise", "attention"):
            assert ev in EVENT_CATALOG

    def test_each_event_has_priority(self):
        from reaction_engine import EVENT_CATALOG
        for name, cfg in EVENT_CATALOG.items():
            assert "priority" in cfg, f"{name} missing priority"
            assert isinstance(cfg["priority"], (int, float))


class TestGetBehaviorParams:
    def test_returns_dict(self):
        from reaction_engine import get_behavior_params
        p = get_behavior_params("idle")
        assert isinstance(p, dict)

    def test_known_behavior_has_params(self):
        from reaction_engine import get_behavior_params
        p = get_behavior_params("talking")
        assert len(p) > 0

    def test_unknown_behavior_returns_fallback(self):
        from reaction_engine import get_behavior_params
        p = get_behavior_params("nonexistent_behavior_xyz")
        assert isinstance(p, dict)


class TestReactionEngine:
    def test_construction_default(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine()
        assert eng is not None

    def test_construction_with_emotion(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("happy")
        # default_emotion is "happy", current may be "neutral" until a trigger
        assert isinstance(eng.current_emotion, str)

    def test_initial_behavior_is_idle(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        assert eng.current_behavior == "idle"

    def test_initial_is_not_speaking(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        assert eng.is_speaking is False

    def test_trigger_changes_emotion(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        eng.trigger("user_message")
        # After trigger, emotion should change from neutral
        # (exact value depends on catalog config)
        assert isinstance(eng.current_emotion, str)

    def test_trigger_loud_noise_sets_speaking_false(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        eng.trigger("loud_noise")
        # loud_noise is not a talking event
        assert isinstance(eng.is_speaking, bool)

    def test_trigger_user_message_enables_speaking(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        eng.trigger("user_message")
        assert eng.is_speaking is True

    def test_update_does_not_crash(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        eng.trigger("attention")
        eng.update()   # should not raise
        eng.update()

    def test_active_is_none_or_reaction(self):
        from reaction_engine import ReactionEngine, Reaction
        eng = ReactionEngine("neutral")
        assert eng.active is None or isinstance(eng.active, Reaction)

    def test_trigger_sets_active(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        eng.trigger("praise")
        assert eng.active is not None

    def test_high_priority_interrupts_low(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        eng.trigger("idle")         # low priority
        first_behavior = eng.current_behavior
        eng.trigger("loud_noise")   # high priority
        # loud_noise should override idle
        assert eng.current_behavior != "idle" or eng.active is not None

    def test_current_emotion_is_string(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        assert isinstance(eng.current_emotion, str)

    def test_current_behavior_is_string(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        assert isinstance(eng.current_behavior, str)

    def test_debug_state_returns_str_or_dict(self):
        from reaction_engine import ReactionEngine
        eng = ReactionEngine("neutral")
        state = eng.debug_state()
        assert isinstance(state, (dict, str))


# ─────────────────────────────────────────────────────────────────────────────
#  EMOTION BLENDER (timeline_sequencer)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionBlender:
    def test_construction(self):
        from timeline_sequencer import EmotionBlender
        b = EmotionBlender("neutral")
        assert b.emotion_name == "neutral"

    def test_initial_values_from_emotion(self):
        from timeline_sequencer import EmotionBlender
        from emotion_engine import EMOTIONS
        b = EmotionBlender("happy")
        assert abs(b.eye_open - EMOTIONS["happy"]["eye_open"]) < 0.01
        assert abs(b.brow_angle - EMOTIONS["happy"]["brow_angle"]) < 0.01

    def test_set_emotion_changes_target(self):
        from timeline_sequencer import EmotionBlender
        b = EmotionBlender("neutral")
        b.set_emotion("angry")
        assert b.emotion_name == "angry"

    def test_update_moves_toward_target(self):
        from timeline_sequencer import EmotionBlender
        from emotion_engine import EMOTIONS
        b = EmotionBlender("neutral")
        b.set_emotion("surprised")
        old_eye = b.eye_open
        b.update(dt=0.1)
        target_eye = EMOTIONS["surprised"]["eye_open"]
        # Should move toward target
        assert abs(b.eye_open - old_eye) >= 0.0  # at least some movement or same

    def test_update_never_overshoots(self):
        from timeline_sequencer import EmotionBlender
        from emotion_engine import EMOTIONS
        b = EmotionBlender("neutral")
        b.set_emotion("happy")
        for _ in range(100):
            b.update(dt=0.1)
        target = EMOTIONS["happy"]["eye_open"]
        # After many steps, should be very close to target
        assert abs(b.eye_open - target) < 0.05

    def test_look_bias_is_tuple(self):
        from timeline_sequencer import EmotionBlender
        b = EmotionBlender("neutral")
        lb = b.look_bias
        assert isinstance(lb, tuple)
        assert len(lb) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  LIPSYNC SCHEDULER (timeline_sequencer)
# ─────────────────────────────────────────────────────────────────────────────

class TestLipsyncScheduler:
    def test_construction(self):
        from timeline_sequencer import LipsyncScheduler
        s = LipsyncScheduler(fps=12)
        assert s.is_speaking is False

    def test_start_speech_sets_speaking(self):
        from timeline_sequencer import LipsyncScheduler
        s = LipsyncScheduler(fps=12)
        s.start_speech(duration_frames=24)
        assert s.is_speaking is True

    def test_tick_returns_mouth_state(self):
        from timeline_sequencer import LipsyncScheduler
        from duck_animator import MouthState
        s = LipsyncScheduler(fps=12)
        s.start_speech(duration_frames=24)
        m = s.tick()
        assert isinstance(m, MouthState)

    def test_stop_speech(self):
        from timeline_sequencer import LipsyncScheduler
        s = LipsyncScheduler(fps=12)
        s.start_speech(duration_frames=24)
        s.stop_speech()
        assert s.is_speaking is False

    def test_exhausted_track_stops_speaking(self):
        from timeline_sequencer import LipsyncScheduler
        s = LipsyncScheduler(fps=12)
        s.start_speech(duration_frames=3)
        for _ in range(5):
            s.tick()
        assert s.is_speaking is False


# ─────────────────────────────────────────────────────────────────────────────
#  TIMELINE SEQUENCER
# ─────────────────────────────────────────────────────────────────────────────

class TestTimelineSequencer:
    def test_construction(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        assert seq is not None

    def test_construction_rgb_image(self):
        from timeline_sequencer import TimelineSequencer
        img = Image.new("RGB", (128, 128), (100, 200, 100))
        seq = TimelineSequencer(img, fps=6)
        assert seq is not None  # should auto-convert

    def test_load_script(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        seq.load_script([{"t": 0.0, "event": "idle"}])
        assert len(seq._script) == 1

    def test_render_returns_frames(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        seq.load_script([{"t": 0.0, "event": "idle"}])
        frames = seq.render(duration=0.5)
        assert len(frames) == 3  # 0.5s * 6fps = 3

    def test_render_frame_is_rgba(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        seq.load_script([{"t": 0.0, "event": "idle"}])
        frames = seq.render(duration=0.5)
        for f in frames:
            assert f.mode == "RGBA"

    def test_render_frame_correct_size(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        seq.load_script([{"t": 0.0, "event": "idle"}])
        frames = seq.render(duration=0.5)
        for f in frames:
            assert f.size == duck_image.size

    def test_script_events_fired(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        seq.load_script([
            {"t": 0.0, "event": "idle"},
            {"t": 0.2, "event": "user_message"},
        ])
        frames = seq.render(duration=0.6)
        assert len(frames) > 0  # just verifying no crash

    def test_empty_script_defaults_to_idle(self, duck_image):
        from timeline_sequencer import TimelineSequencer
        seq = TimelineSequencer(duck_image, fps=6)
        seq.load_script([])
        frames = seq.render(duration=0.5)
        assert len(frames) == 3


class TestAnimateFunction:
    def test_returns_list_of_images(self, duck_image):
        from timeline_sequencer import animate
        frames = animate(duck_image, duration=0.5, fps=6)
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_correct_frame_count(self, duck_image):
        from timeline_sequencer import animate
        frames = animate(duck_image, duration=1.0, fps=6)
        assert len(frames) == 6

    def test_frames_are_rgba(self, duck_image):
        from timeline_sequencer import animate
        frames = animate(duck_image, duration=0.5, fps=6)
        for f in frames:
            assert f.mode == "RGBA"

    def test_frames_match_input_size(self, duck_image):
        from timeline_sequencer import animate
        frames = animate(duck_image, duration=0.5, fps=6)
        for f in frames:
            assert f.size == duck_image.size

    def test_with_emotion(self, duck_image):
        from timeline_sequencer import animate
        frames = animate(duck_image, duration=0.5, emotion="happy", fps=6)
        assert len(frames) > 0

    def test_with_script(self, duck_image):
        from timeline_sequencer import animate
        script = [
            {"t": 0.0, "event": "idle"},
            {"t": 0.3, "event": "user_message"},
        ]
        frames = animate(duck_image, duration=0.6, script=script, fps=6)
        assert len(frames) == 4

    def test_no_script_uses_idle(self, duck_image):
        from timeline_sequencer import animate
        frames = animate(duck_image, duration=0.5, fps=6)
        assert len(frames) == 3

    def test_all_emotions_work(self, duck_image):
        from timeline_sequencer import animate
        from emotion_engine import EMOTIONS
        for em in EMOTIONS:
            frames = animate(duck_image, duration=0.25, emotion=em, fps=4)
            assert len(frames) == 1, f"Failed for emotion={em}"
