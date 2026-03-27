"""
tests/test_world_simulation.py
================================

Comprehensive tests for the world simulation subsystem:
  - duck_entity:        DuckEntity, PersonalityProfile
  - interaction_engine: InteractionEngine, ConversationSession
  - renderer:           WorldRenderer, _DuckRenderState
  - world_simulation:   WorldSimulation, save_gif
  - persistence_layer:  save_world, load_world, create_world,
                        load_or_create_world, snapshot
"""

import sys
import os
from pathlib import Path

import pytest
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from duck_entity import DuckEntity, PersonalityProfile, create_duck
from interaction_engine import InteractionEngine, ConversationSession
from relationship_engine import RelationshipEngine
from renderer import WorldRenderer, _DuckRenderState
from world_simulation import WorldSimulation, save_gif
from persistence_layer import (
    save_world,
    load_world,
    create_world,
    load_or_create_world,
    snapshot,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def test_image():
    return Image.new("RGBA", (128, 128), (255, 220, 60, 255))


@pytest.fixture
def duck():
    return DuckEntity("d0", name="Quackers", position=(0.5, 0.5))


@pytest.fixture
def two_ducks():
    return [
        DuckEntity("d0", name="Alice", position=(0.3, 0.4)),
        DuckEntity("d1", name="Bob",   position=(0.7, 0.6)),
    ]


@pytest.fixture
def world(test_image):
    return WorldSimulation.create_new(num_ducks=2, canvas_size=(400, 300))


# ─────────────────────────────────────────────────────────────────────────────
#  TestPersonalityProfile
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonalityProfile:
    def test_default_shyness(self):
        p = PersonalityProfile()
        assert p.shyness == pytest.approx(0.30)

    def test_default_sociability(self):
        p = PersonalityProfile()
        assert p.sociability == pytest.approx(0.50)

    def test_default_energy(self):
        p = PersonalityProfile()
        assert p.energy == pytest.approx(0.50)

    def test_default_caution(self):
        p = PersonalityProfile()
        assert p.caution == pytest.approx(0.40)

    def test_random_creates_instance(self):
        p = PersonalityProfile.random()
        assert isinstance(p, PersonalityProfile)

    def test_random_traits_in_range(self):
        for _ in range(20):
            p = PersonalityProfile.random()
            assert 0.0 <= p.shyness <= 1.0
            assert 0.0 <= p.sociability <= 1.0
            assert 0.0 <= p.energy <= 1.0
            assert 0.0 <= p.caution <= 1.0

    def test_random_different_each_time(self):
        profiles = [PersonalityProfile.random() for _ in range(10)]
        shyness_values = [p.shyness for p in profiles]
        # Very unlikely all 10 are identical
        assert len(set(round(v, 5) for v in shyness_values)) > 1

    def test_label_returns_string(self):
        p = PersonalityProfile()
        assert isinstance(p.label, str)
        assert p.label in ("shy", "social", "energetic", "cautious", "balanced")

    def test_label_shy_when_high_shyness(self):
        p = PersonalityProfile(shyness=0.9, sociability=0.2, energy=0.3, caution=0.4)
        assert p.label == "shy"

    def test_to_dict_round_trip(self):
        p = PersonalityProfile(shyness=0.6, sociability=0.4, energy=0.7, caution=0.2)
        d = p.to_dict()
        p2 = PersonalityProfile.from_dict(d)
        assert p2.shyness == pytest.approx(p.shyness)
        assert p2.sociability == pytest.approx(p.sociability)
        assert p2.energy == pytest.approx(p.energy)
        assert p2.caution == pytest.approx(p.caution)

    def test_evolve_does_not_raise(self):
        p = PersonalityProfile()
        for exp in ("positive_interaction", "ignored", "startled",
                    "praised", "scolded", "long_friendship", "isolation"):
            p.evolve(exp, intensity=0.3)


# ─────────────────────────────────────────────────────────────────────────────
#  TestDuckEntity
# ─────────────────────────────────────────────────────────────────────────────

class TestDuckEntity:
    def test_id_stored(self, duck):
        assert duck.id == "d0"

    def test_name_stored(self, duck):
        assert duck.name == "Quackers"

    def test_position_stored(self, duck):
        assert duck.position == [0.5, 0.5]

    def test_position_is_mutable_list(self, duck):
        assert isinstance(duck.position, list)

    def test_initial_is_speaking_false(self, duck):
        assert duck.is_speaking is False

    def test_initial_is_listening_false(self, duck):
        assert duck.is_listening is False

    def test_initial_gaze_target_none(self, duck):
        assert duck.gaze_target_id is None

    def test_update_does_not_raise(self, duck):
        duck.update(dt=0.1)

    def test_update_with_other_ducks(self, two_ducks):
        for d in two_ducks:
            d.update(dt=0.1, other_ducks=two_ducks)

    def test_get_character_state_returns_object(self, duck):
        duck.update(dt=0.1)
        cs = duck.get_character_state()
        assert cs is not None

    def test_get_character_state_has_emotion_name(self, duck):
        duck.update(dt=0.1)
        cs = duck.get_character_state()
        assert isinstance(cs.emotion_name, str)

    def test_trigger_does_not_raise(self, duck):
        duck.trigger("user_message")

    def test_trigger_valid_event(self, duck):
        duck.trigger("greeting")
        duck.update(dt=0.1)

    def test_to_dict_returns_dict(self, duck):
        assert isinstance(duck.to_dict(), dict)

    def test_to_dict_has_id(self, duck):
        assert duck.to_dict()["id"] == "d0"

    def test_to_dict_has_position(self, duck):
        assert "position" in duck.to_dict()

    def test_from_dict_round_trip_id(self, duck):
        d = duck.to_dict()
        duck2 = DuckEntity.from_dict(d)
        assert duck2.id == duck.id

    def test_from_dict_round_trip_name(self, duck):
        d = duck.to_dict()
        duck2 = DuckEntity.from_dict(d)
        assert duck2.name == duck.name

    def test_from_dict_round_trip_position(self, duck):
        d = duck.to_dict()
        duck2 = DuckEntity.from_dict(d)
        assert duck2.position[0] == pytest.approx(duck.position[0])
        assert duck2.position[1] == pytest.approx(duck.position[1])

    def test_from_dict_round_trip_is_speaking(self, duck):
        d = duck.to_dict()
        duck2 = DuckEntity.from_dict(d)
        assert duck2.is_speaking == duck.is_speaking

    def test_repr_is_string(self, duck):
        assert isinstance(repr(duck), str)

    def test_create_duck_factory(self):
        d = create_duck(duck_id="test_001", name="Pip", position=(0.2, 0.8))
        assert d.id == "test_001"
        assert d.name == "Pip"

    def test_update_multiple_steps(self, duck):
        for _ in range(10):
            duck.update(dt=0.1)
        cs = duck.get_character_state()
        assert cs is not None

    def test_position_stays_in_bounds_after_update(self, duck):
        for _ in range(50):
            duck.update(dt=0.05)
        assert 0.0 <= duck.position[0] <= 1.0
        assert 0.0 <= duck.position[1] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  TestInteractionEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractionEngine:
    def test_instantiation(self):
        ie = InteractionEngine()
        assert ie is not None

    def test_update_returns_list(self, two_ducks):
        ie = InteractionEngine()
        rels = RelationshipEngine()
        result = ie.update(two_ducks, rels, dt=0.1, sim_time=0.0)
        assert isinstance(result, list)

    def test_update_with_empty_ducks(self):
        ie = InteractionEngine()
        rels = RelationshipEngine()
        result = ie.update([], rels, dt=0.1, sim_time=0.0)
        assert result == []

    def test_update_with_single_duck(self):
        ie = InteractionEngine()
        rels = RelationshipEngine()
        ducks = [DuckEntity("d0", position=(0.5, 0.5))]
        result = ie.update(ducks, rels, dt=0.1, sim_time=0.0)
        assert isinstance(result, list)

    def test_update_multiple_times_does_not_crash(self, two_ducks):
        ie = InteractionEngine()
        rels = RelationshipEngine()
        for i in range(10):
            ie.update(two_ducks, rels, dt=0.1, sim_time=float(i) * 0.1)

    def test_update_result_items_are_dicts(self, two_ducks):
        ie = InteractionEngine()
        rels = RelationshipEngine()
        # Move ducks close together to encourage interaction events
        two_ducks[0].position = [0.3, 0.3]
        two_ducks[1].position = [0.32, 0.32]
        # Run many steps to trigger an event
        results = []
        for i in range(50):
            results.extend(ie.update(two_ducks, rels, dt=0.1, sim_time=float(i) * 0.1))
        for item in results:
            assert isinstance(item, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  TestWorldRenderer
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldRenderer:
    def test_instantiation_with_canvas_size(self):
        renderer = WorldRenderer(canvas_size=(400, 300))
        assert renderer.canvas_w == 400
        assert renderer.canvas_h == 300

    def test_add_duck_does_not_raise(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)

    def test_render_frame_returns_image(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)
        duck = DuckEntity("d0", position=(0.5, 0.5))
        frame = renderer.render_frame([duck])
        assert isinstance(frame, Image.Image)

    def test_render_frame_is_rgba(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)
        duck = DuckEntity("d0", position=(0.5, 0.5))
        frame = renderer.render_frame([duck])
        assert frame.mode == "RGBA"

    def test_render_frame_correct_size(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)
        duck = DuckEntity("d0", position=(0.5, 0.5))
        frame = renderer.render_frame([duck])
        assert frame.size == (400, 300)

    def test_render_frame_empty_ducks(self):
        renderer = WorldRenderer(canvas_size=(200, 150))
        frame = renderer.render_frame([])
        assert frame.size == (200, 150)

    def test_render_frame_multiple_ducks(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)
        renderer.add_duck("d1", test_image)
        ducks = [
            DuckEntity("d0", position=(0.3, 0.4)),
            DuckEntity("d1", position=(0.7, 0.6)),
        ]
        frame = renderer.render_frame(ducks)
        assert frame.size == (400, 300)
        assert frame.mode == "RGBA"

    def test_registered_ducks_after_add(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)
        assert "d0" in renderer.registered_ducks()

    def test_canvas_pos_normalised_to_pixel(self):
        renderer = WorldRenderer(canvas_size=(400, 300))
        px, py = renderer.canvas_pos(0.5, 0.5)
        assert px == 200
        assert py == 150

    def test_canvas_pos_origin(self):
        renderer = WorldRenderer(canvas_size=(400, 300))
        px, py = renderer.canvas_pos(0.0, 0.0)
        assert px == 0
        assert py == 0

    def test_render_frame_consistent_size_across_calls(self, test_image):
        renderer = WorldRenderer(canvas_size=(400, 300))
        renderer.add_duck("d0", test_image)
        duck = DuckEntity("d0", position=(0.5, 0.5))
        for _ in range(5):
            frame = renderer.render_frame([duck])
            assert frame.size == (400, 300)


# ─────────────────────────────────────────────────────────────────────────────
#  TestWorldSimulation
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldSimulation:
    def test_create_new_returns_instance(self):
        world = WorldSimulation.create_new(num_ducks=2, canvas_size=(400, 300))
        assert isinstance(world, WorldSimulation)

    def test_create_new_has_correct_duck_count(self):
        world = WorldSimulation.create_new(num_ducks=3, canvas_size=(400, 300))
        assert len(world.ducks) == 3

    def test_create_new_canvas_size_stored(self):
        world = WorldSimulation.create_new(num_ducks=2, canvas_size=(400, 300))
        assert world.canvas_size == (400, 300)

    def test_step_returns_image(self, world):
        frame = world.step(dt=0.1)
        assert isinstance(frame, Image.Image)

    def test_step_returns_rgba_image(self, world):
        frame = world.step(dt=0.1)
        assert frame.mode == "RGBA"

    def test_step_returns_correct_size(self, world):
        frame = world.step(dt=0.1)
        assert frame.size == (400, 300)

    def test_step_multiple_times(self, world):
        for _ in range(5):
            frame = world.step(dt=0.1)
            assert frame.size == (400, 300)

    def test_simulate_returns_list(self, world):
        frames = world.simulate(duration=0.5, fps=6)
        assert isinstance(frames, list)

    def test_simulate_correct_frame_count(self, world):
        frames = world.simulate(duration=0.5, fps=6)
        assert len(frames) == 3  # 0.5s * 6fps = 3

    def test_simulate_frames_are_rgba(self, world):
        frames = world.simulate(duration=0.5, fps=6)
        for f in frames:
            assert f.mode == "RGBA"

    def test_simulate_frames_correct_size(self, world):
        frames = world.simulate(duration=0.5, fps=6)
        for f in frames:
            assert f.size == (400, 300)

    def test_simulate_returns_pil_images(self, world):
        frames = world.simulate(duration=0.5, fps=6)
        for f in frames:
            assert isinstance(f, Image.Image)

    def test_to_dict_returns_dict(self, world):
        assert isinstance(world.to_dict(), dict)

    def test_to_dict_has_canvas_size(self, world):
        d = world.to_dict()
        assert "canvas_size" in d
        assert d["canvas_size"] == [400, 300]

    def test_to_dict_has_ducks(self, world):
        d = world.to_dict()
        assert "ducks" in d

    def test_to_dict_has_clock(self, world):
        d = world.to_dict()
        assert "clock" in d

    def test_from_dict_round_trip_canvas_size(self, world):
        d = world.to_dict()
        world2 = WorldSimulation.from_dict(d, images={})
        assert world2.canvas_size == world.canvas_size

    def test_from_dict_round_trip_duck_count(self, world):
        d = world.to_dict()
        world2 = WorldSimulation.from_dict(d, images={})
        assert len(world2.ducks) == len(world.ducks)

    def test_from_dict_preserves_duck_ids(self, world):
        d = world.to_dict()
        world2 = WorldSimulation.from_dict(d, images={})
        assert set(world2.ducks.keys()) == set(world.ducks.keys())

    def test_add_duck_increases_count(self, test_image):
        world = WorldSimulation.create_new(num_ducks=1, canvas_size=(400, 300))
        new_duck = DuckEntity("new", position=(0.5, 0.5))
        world.add_duck("new", new_duck, test_image)
        assert len(world.ducks) == 2

    def test_remove_duck_decreases_count(self, world):
        initial = len(world.ducks)
        first_id = next(iter(world.ducks))
        world.remove_duck(first_id)
        assert len(world.ducks) == initial - 1

    def test_frame_count_increments(self, world):
        initial = world._frame_count
        world.step(dt=0.1)
        assert world._frame_count == initial + 1


# ─────────────────────────────────────────────────────────────────────────────
#  TestSaveGif
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveGif:
    def test_save_gif_creates_file(self, tmp_path):
        frames = [Image.new("RGBA", (64, 64), (255, 0, 0, 255)) for _ in range(3)]
        out = str(tmp_path / "test.gif")
        save_gif(frames, out, fps=6)
        assert os.path.exists(out)

    def test_save_gif_file_is_nonzero(self, tmp_path):
        frames = [Image.new("RGBA", (64, 64), (0, 255, 0, 255)) for _ in range(3)]
        out = str(tmp_path / "test.gif")
        save_gif(frames, out, fps=6)
        assert os.path.getsize(out) > 0

    def test_save_gif_readable_as_image(self, tmp_path):
        frames = [Image.new("RGBA", (64, 64), (0, 0, 255, 255)) for _ in range(3)]
        out = str(tmp_path / "test.gif")
        save_gif(frames, out, fps=6)
        loaded = Image.open(out)
        assert loaded.format == "GIF"

    def test_save_gif_empty_raises(self, tmp_path):
        out = str(tmp_path / "empty.gif")
        with pytest.raises((ValueError, Exception)):
            save_gif([], out)

    def test_save_gif_single_frame(self, tmp_path):
        frames = [Image.new("RGBA", (64, 64), (128, 128, 128, 255))]
        out = str(tmp_path / "single.gif")
        save_gif(frames, out, fps=12)
        assert os.path.exists(out)


# ─────────────────────────────────────────────────────────────────────────────
#  TestPersistenceLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistenceLayer:
    def test_save_world_creates_file(self, tmp_path, world):
        path = str(tmp_path / "world.json")
        save_world(world, path)
        assert os.path.exists(path)

    def test_save_world_file_is_json(self, tmp_path, world):
        import json
        path = str(tmp_path / "world.json")
        save_world(world, path)
        with open(path) as f:
            doc = json.load(f)
        assert isinstance(doc, dict)

    def test_save_world_has_version(self, tmp_path, world):
        import json
        path = str(tmp_path / "world.json")
        save_world(world, path)
        with open(path) as f:
            doc = json.load(f)
        assert "version" in doc

    def test_load_world_returns_instance(self, tmp_path, world):
        path = str(tmp_path / "world.json")
        save_world(world, path)
        world2 = load_world(path)
        assert isinstance(world2, WorldSimulation)

    def test_load_world_preserves_duck_count(self, tmp_path, world):
        path = str(tmp_path / "world.json")
        save_world(world, path)
        world2 = load_world(path)
        assert len(world2.ducks) == len(world.ducks)

    def test_load_world_preserves_canvas_size(self, tmp_path, world):
        path = str(tmp_path / "world.json")
        save_world(world, path)
        world2 = load_world(path)
        assert world2.canvas_size == world.canvas_size

    def test_load_world_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_world(str(tmp_path / "nonexistent.json"))

    def test_create_world_returns_instance(self):
        w = create_world(num_ducks=2, canvas_size=(400, 300))
        assert isinstance(w, WorldSimulation)

    def test_create_world_correct_duck_count(self):
        w = create_world(num_ducks=2, canvas_size=(400, 300))
        assert len(w.ducks) == 2

    def test_create_world_with_save_path(self, tmp_path):
        path = str(tmp_path / "new_world.json")
        w = create_world(num_ducks=2, save_path=path, canvas_size=(400, 300))
        assert os.path.exists(path)
        assert isinstance(w, WorldSimulation)

    def test_load_or_create_world_creates_when_missing(self, tmp_path):
        path = str(tmp_path / "missing.json")
        w = load_or_create_world(path, num_ducks=2, canvas_size=(400, 300))
        assert isinstance(w, WorldSimulation)
        assert os.path.exists(path)

    def test_load_or_create_world_loads_existing(self, tmp_path, world):
        path = str(tmp_path / "existing.json")
        save_world(world, path)
        w2 = load_or_create_world(path, num_ducks=99, canvas_size=(400, 300))
        # Should load existing, not create new with num_ducks=99
        assert len(w2.ducks) == len(world.ducks)

    def test_load_or_create_world_preserves_canvas_size(self, tmp_path, world):
        path = str(tmp_path / "world.json")
        save_world(world, path)
        w2 = load_or_create_world(path)
        assert w2.canvas_size == world.canvas_size

    def test_snapshot_returns_frames_list(self, tmp_path, world):
        path = str(tmp_path / "snap.gif")
        frames = snapshot(world, path, duration=0.5, fps=6)
        assert isinstance(frames, list)
        assert len(frames) == 3

    def test_snapshot_creates_gif_file(self, tmp_path, world):
        path = str(tmp_path / "snap.gif")
        snapshot(world, path, duration=0.5, fps=6)
        assert os.path.exists(path)

    def test_snapshot_gif_is_readable(self, tmp_path, world):
        path = str(tmp_path / "snap.gif")
        snapshot(world, path, duration=0.5, fps=6)
        img = Image.open(path)
        assert img.format == "GIF"

    def test_snapshot_frames_are_rgba(self, tmp_path, world):
        path = str(tmp_path / "snap.gif")
        frames = snapshot(world, path, duration=0.5, fps=6)
        for f in frames:
            assert f.mode == "RGBA"

    def test_snapshot_frames_correct_size(self, tmp_path, world):
        path = str(tmp_path / "snap.gif")
        frames = snapshot(world, path, duration=0.5, fps=6)
        for f in frames:
            assert f.size == (400, 300)
