#!/usr/bin/env python3
"""
renderer.py — Multi-duck world canvas renderer
================================================

Draws all DuckEntity objects onto a single canvas image each simulation frame.

Design decisions
────────────────
• Ducks with higher Y (closer to the bottom) are drawn on top (painter's
  algorithm / fake depth).
• Each duck is composited at a depth-scaled size: farther = smaller.
• The overlay (eyes, eyebrows, beak) is applied to the scaled image so all
  proportions remain correct at any resolution.
• An optional soft shadow is drawn beneath each duck for grounding.
• The renderer is stateless between calls except for per-duck cached geometry
  and blink / pupil curve state (to make animations smooth across frames).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter

from duck_animator import (
    FaceGeometry,
    CharacterState,
    MouthState,
    generate_blink_seq,
    generate_pupil_track,
    generate_head_bob,
    render_single_frame,
    _generate_reactive_blink_seq,
)


# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH / LAYOUT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Duck image height at world Y=0 (back of scene) and Y=1 (front)
# as a fraction of canvas height.
DEPTH_SCALE_MIN = 0.14    # far away   (y ≈ 0)
DEPTH_SCALE_MAX = 0.38    # up close   (y ≈ 1)

# Background colour (RGBA) — soft grassy green
DEFAULT_BG_COLOR = (162, 210, 130, 255)

# Shadow parameters
SHADOW_COLOR     = (0, 0, 0, 60)
SHADOW_ELLIPSE_W = 0.80   # relative to duck image width
SHADOW_ELLIPSE_H = 0.12   # relative to duck image height


# ─────────────────────────────────────────────────────────────────────────────
#  PER-DUCK RENDER STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _DuckRenderState:
    """
    Per-duck animation state maintained by the renderer between frames.

    Blink sequences and pupil tracks are pre-generated for a buffer window
    and regenerated when exhausted — this gives smooth, continuous animation.
    """
    entity_id:  str
    base_image: Image.Image     # original full-res RGBA duck image
    geo:        FaceGeometry    # geometry for the full-res image
    scaled_geo: Optional[FaceGeometry] = None   # geometry for current depth size

    # Animation curves (refilled when depleted)
    blink_l:   list[float] = field(default_factory=list)
    blink_r:   list[float] = field(default_factory=list)
    pupil_xy:  list[tuple] = field(default_factory=list)
    bob_xy:    list[tuple] = field(default_factory=list)
    curve_idx: int = 0

    # Pre-scaled image cache (key = (w, h))
    _scale_cache: dict = field(default_factory=dict)

    CURVE_BUFFER = 120   # frames per buffer window

    def get_scaled_image(self, target_size: tuple[int, int]) -> Image.Image:
        """Return the base image scaled to (w, h), cached."""
        if target_size not in self._scale_cache:
            if len(self._scale_cache) > 8:
                self._scale_cache.clear()
            self._scale_cache[target_size] = self.base_image.resize(
                target_size, Image.LANCZOS
            )
        return self._scale_cache[target_size]

    def advance(self, blink_rate: float = 0.03) -> tuple[float, float, tuple, tuple]:
        """
        Advance one frame and return (blink_l, blink_r, (pupil_x, pupil_y), (bob_dy, _)).
        Regenerates curves when the buffer is exhausted.
        """
        if self.curve_idx >= len(self.blink_l) - 1:
            self._refill(blink_rate)

        i = self.curve_idx
        bl = self.blink_l[i]
        br = self.blink_r[i]
        pp = self.pupil_xy[i]
        bb = self.bob_xy[i]
        self.curve_idx += 1
        return bl, br, pp, bb

    def _refill(self, blink_rate: float = 0.03) -> None:
        n              = self.CURVE_BUFFER
        rates          = [blink_rate] * n
        self.blink_l   = _generate_reactive_blink_seq(rates, frame_offset=0)
        self.blink_r   = _generate_reactive_blink_seq(rates,
                                                       frame_offset=random.randint(1, 3))
        self.pupil_xy  = generate_pupil_track(n)
        self.bob_xy    = generate_head_bob(n)
        self.curve_idx = 0


# ─────────────────────────────────────────────────────────────────────────────
#  WORLD RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class WorldRenderer:
    """
    Renders all ducks in the world onto a single canvas each frame.

    Usage
    ─────
        renderer = WorldRenderer(canvas_size=(800, 500))
        renderer.add_duck("duck_0", duck_entity, duck_base_image)
        …
        frame_image = renderer.render_frame(ducks, states)
    """

    def __init__(
        self,
        canvas_size:    tuple[int, int] = (800, 500),
        background:     Image.Image | None = None,
        bg_color:       tuple             = DEFAULT_BG_COLOR,
        draw_shadows:   bool              = True,
        draw_names:     bool              = False,
        draw_debug:     bool              = False,
        draw_beak:      bool              = True,
    ) -> None:
        self.canvas_w, self.canvas_h = canvas_size
        self.background   = background
        self.bg_color     = bg_color
        self.draw_shadows = draw_shadows
        self.draw_names   = draw_names
        self.draw_debug   = draw_debug
        self.draw_beak    = draw_beak

        # Per-duck render state keyed by duck id
        self._states: dict[str, _DuckRenderState] = {}

    # ── Duck registration ─────────────────────────────────────────────────────

    def add_duck(
        self,
        duck_id:    str,
        base_image: Image.Image,
    ) -> None:
        """Register a duck with its base portrait image."""
        if base_image.mode != "RGBA":
            base_image = base_image.convert("RGBA")
        geo = FaceGeometry(base_image.width, base_image.height)
        self._states[duck_id] = _DuckRenderState(
            entity_id  = duck_id,
            base_image = base_image,
            geo        = geo,
        )

    def remove_duck(self, duck_id: str) -> None:
        self._states.pop(duck_id, None)

    # ── Main render ───────────────────────────────────────────────────────────

    def render_frame(
        self,
        ducks:  list,   # list[DuckEntity]
        states: dict[str, CharacterState] | None = None,
    ) -> Image.Image:
        """
        Render one animation frame.

        ducks   List of DuckEntity objects (position / state data).
        states  Optional pre-built CharacterState per duck id.
                If None, each duck's get_character_state() is called.

        Returns a new RGBA canvas Image.
        """
        # Create or copy background
        canvas = self._make_canvas()

        # Sort ducks back-to-front by Y so nearer ducks overdraw farther ones
        sorted_ducks = sorted(ducks, key=lambda d: d.position[1])

        for duck in sorted_ducks:
            duck_id = duck.id
            if duck_id not in self._states:
                # Auto-register with a solid yellow placeholder
                placeholder = Image.new("RGBA", (128, 128), (255, 220, 60, 255))
                self.add_duck(duck_id, placeholder)

            rs     = self._states[duck_id]
            cstate = (states or {}).get(duck_id) or duck.get_character_state()

            self._composite_duck(canvas, duck, rs, cstate)

        return canvas

    # ── Internal compositing ──────────────────────────────────────────────────

    def _make_canvas(self) -> Image.Image:
        """Create the background canvas for this frame."""
        if self.background is not None:
            bg = self.background.copy().convert("RGBA")
            if bg.size != (self.canvas_w, self.canvas_h):
                bg = bg.resize((self.canvas_w, self.canvas_h), Image.LANCZOS)
            return bg

        canvas = Image.new("RGBA", (self.canvas_w, self.canvas_h), self.bg_color)

        # Simple gradient: lighten toward the top (sky feel)
        draw = ImageDraw.Draw(canvas)
        for y in range(self.canvas_h // 2):
            alpha = int(30 * (1.0 - y / (self.canvas_h // 2)))
            draw.line(
                [(0, y), (self.canvas_w, y)],
                fill=(255, 255, 255, alpha),
            )
        return canvas

    def _composite_duck(
        self,
        canvas: Image.Image,
        duck,              # DuckEntity
        rs:     _DuckRenderState,
        cstate: CharacterState,
    ) -> None:
        """Scale, animate, and paste one duck onto the canvas."""
        # ── Depth-based size ─────────────────────────────────────────────────
        y_norm    = max(0.0, min(1.0, duck.position[1]))
        depth_s   = DEPTH_SCALE_MIN + (DEPTH_SCALE_MAX - DEPTH_SCALE_MIN) * y_norm
        duck_h    = int(self.canvas_h * depth_s)
        aspect    = rs.base_image.width / rs.base_image.height
        duck_w    = max(1, int(duck_h * aspect))
        duck_size = (duck_w, duck_h)

        # ── Scaled geometry ───────────────────────────────────────────────────
        if rs.scaled_geo is None or rs.scaled_geo.width != duck_w:
            rs.scaled_geo = FaceGeometry(duck_w, duck_h)

        scaled_base = rs.get_scaled_image(duck_size)

        # ── Per-frame animation values ────────────────────────────────────────
        blink_l, blink_r, (px, py), (bob_dy, _) = rs.advance(cstate.blink_rate)

        # Blend in live look_bias from CharacterState
        bias_x, bias_y = cstate.look_bias
        pdx = px + bias_x * 0.4 + random.gauss(0.0, 0.12 * cstate.eye_jitter_scale)
        pdy = py + bias_y * 0.4 + random.gauss(0.0, 0.12 * cstate.eye_jitter_scale)

        bob_dy *= cstate.head_bob_scale

        # ── Render the duck frame ─────────────────────────────────────────────
        duck_frame = render_single_frame(
            base_image = scaled_base,
            state      = cstate,
            geo        = rs.scaled_geo,
            blink_l    = blink_l,
            blink_r    = blink_r,
            pupil_dx   = pdx,
            pupil_dy   = pdy,
            bob_dy     = bob_dy,
            draw_beak  = self.draw_beak,
        )

        # ── Canvas position ───────────────────────────────────────────────────
        cx = int(duck.position[0] * self.canvas_w)
        cy = int(duck.position[1] * self.canvas_h)

        # Anchor: centre-bottom of duck image at (cx, cy)
        paste_x = cx - duck_w // 2
        paste_y = cy - duck_h

        # ── Shadow ────────────────────────────────────────────────────────────
        if self.draw_shadows:
            self._draw_shadow(canvas, cx, cy, duck_w, duck_h)

        # ── Paste duck ────────────────────────────────────────────────────────
        # Use the duck frame's alpha channel as mask for clean compositing
        canvas.paste(duck_frame, (paste_x, paste_y), mask=duck_frame.split()[3])

        # ── Name label (optional) ──────────────────────────────────────────────
        if self.draw_names and hasattr(duck, "name"):
            self._draw_name(canvas, duck.name, cx, paste_y - 4)

        # ── Debug overlay ─────────────────────────────────────────────────────
        if self.draw_debug:
            self._draw_debug(canvas, duck, cstate, cx, cy)

    def _draw_shadow(
        self,
        canvas: Image.Image,
        cx: int, cy: int,
        duck_w: int, duck_h: int,
    ) -> None:
        """Draw a soft ellipse shadow under the duck."""
        sw = int(duck_w * SHADOW_ELLIPSE_W)
        sh = max(4, int(duck_h * SHADOW_ELLIPSE_H))
        sx = cx - sw // 2
        sy = cy - sh // 2

        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sdraw        = ImageDraw.Draw(shadow_layer)
        sdraw.ellipse((sx, sy, sx + sw, sy + sh), fill=SHADOW_COLOR)
        # Blur the shadow for softness
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=3))
        canvas.alpha_composite(shadow_layer)

    def _draw_name(
        self,
        canvas: Image.Image,
        name:   str,
        cx:     int,
        y:      int,
    ) -> None:
        """Draw the duck's name above its head in small text."""
        draw     = ImageDraw.Draw(canvas)
        text_col = (30, 20, 10, 200)
        draw.text((cx, y), name, fill=text_col, anchor="ms")

    def _draw_debug(
        self,
        canvas: Image.Image,
        duck,
        cstate: CharacterState,
        cx: int, cy: int,
    ) -> None:
        """Draw a small debug bubble showing emotion / behavior."""
        draw = ImageDraw.Draw(canvas)
        line = f"{duck.id}|{cstate.emotion_name}|{cstate.behavior}"
        draw.rectangle(
            (cx - 45, cy + 2, cx + 45, cy + 14),
            fill=(0, 0, 0, 100),
        )
        draw.text((cx, cy + 8), line, fill=(255, 255, 200, 220), anchor="mm")

    # ── Utility ───────────────────────────────────────────────────────────────

    def set_background(self, image: Image.Image) -> None:
        self.background = image.convert("RGBA")

    def registered_ducks(self) -> list[str]:
        return list(self._states.keys())

    def canvas_pos(
        self, x_norm: float, y_norm: float
    ) -> tuple[int, int]:
        """Convert normalised world coords to canvas pixel coords."""
        return (
            int(x_norm * self.canvas_w),
            int(y_norm * self.canvas_h),
        )
