#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         GIF MAKER  v1.0  —  Acrolite Tools               ║
║   Turn still images into animated GIFs from the CLI.     ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python3 gif-maker.py                        # interactive mode
  python3 gif-maker.py image.png              # load image, then interactive
  python3 gif-maker.py image.png -e glitch    # non-interactive, one effect
  python3 gif-maker.py image.png -e neon -o sparkles coins -t "ACROLITE" -n

Effects:  none glitch neon kenburns matrix hud holographic fire pixelsort
          lightning vignette colorgrade zoompulse tvstatic ripple

Overlays: sparkles stars fire_particles snow confetti lightning coins hearts
          fireworks rain bubbles smoke arrows crosshair emoji_party neon_border

Text styles: typewriter neon_glow glitch_text bounce scroll fade
"""

import os, sys, math, random, json, shutil, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

VERSION = "1.0"

# ── paths & library ───────────────────────────────────────────────────────────
LIBRARY_DIR  = os.path.expanduser("~/.gif-maker/library")
PRESETS_FILE = os.path.expanduser("~/.gif-maker/presets.json")
os.makedirs(LIBRARY_DIR, exist_ok=True)

# ── ANSI colours ──────────────────────────────────────────────────────────────
RST = "\033[0m";  BLD = "\033[1m";  DIM = "\033[2m"
RED = "\033[31m"; GRN = "\033[32m"; YEL = "\033[33m"
BLU = "\033[34m"; MAG = "\033[35m"; CYN = "\033[36m"
WHT = "\033[37m"

TERM_W = shutil.get_terminal_size((80, 24)).columns
BOX_W  = min(TERM_W, 62)

def clr():           print("\033[2J\033[H", end="", flush=True)
def hr(c="─"):       return c * BOX_W
def c(s, code):      return f"{code}{s}{RST}"

# ── fonts ─────────────────────────────────────────────────────────────────────
_font_cache: dict = {}

def get_font(size=18):
    if size in _font_cache:
        return _font_cache[size]
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                fnt = ImageFont.truetype(fp, size)
                _font_cache[size] = fnt
                return fnt
            except Exception:
                pass
    fnt = ImageFont.load_default()
    _font_cache[size] = fnt
    return fnt

# ── image utils ───────────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")

def resize_fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def blank(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))

def save_optimized(frames, path, dur=80, target=2_000_000):
    """Save GIF with progressive quality reduction until < target bytes."""
    TIERS = [
        (480, None, 64), (480, 40, 64), (480, 30, 56),
        (400, 24, 48),   (360, 20, 40), (320, 16, 32), (280, 12, 28),
    ]
    for max_side, cap, colors in TIERS:
        fs = frames[:cap] if cap else frames
        rs = [resize_fit(f, max_side) for f in fs]
        pals = [
            f.convert("RGBA").quantize(
                colors=colors,
                method=Image.Quantize.FASTOCTREE,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
            for f in rs
        ]
        pals[0].save(
            path, save_all=True, append_images=pals[1:],
            duration=dur, loop=0, optimize=True, disposal=2,
        )
        sz = os.path.getsize(path)
        if sz <= target:
            return sz, max_side, len(fs), colors
    return sz, max_side, len(fs), colors  # best effort

# ══════════════════════════════════════════════════════════════════════════════
#  EFFECTS  — each fn(img: RGBA Image, n_frames: int) -> list[RGBA Image]
# ══════════════════════════════════════════════════════════════════════════════

def efx_none(img, n):
    return [img.copy() for _ in range(n)]

def efx_glitch(img, n):
    base = np.array(img, dtype=np.float32)
    frames = []
    for i in range(n):
        t = i / n
        arr = base.copy()
        shift = int(math.sin(t * math.pi * 4) * 10)
        arr[:, :, 0] = np.roll(base[:, :, 0], shift,  axis=1)
        arr[:, :, 2] = np.roll(base[:, :, 2], -shift, axis=1)
        line = int((t * arr.shape[0] * 2) % arr.shape[0])
        arr[max(0, line-2):line+2, :, :3] = np.clip(
            arr[max(0, line-2):line+2, :, :3] * 2, 0, 255)
        frames.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA"))
    return frames

def efx_neon(img, n):
    base = img.convert("RGB")
    frames = []
    for i in range(n):
        t = i / n
        hsv = base.convert("HSV")
        h, s, v = hsv.split()
        h_arr = ((np.array(h, dtype=np.int32) + int(t * 256)) % 256).astype(np.uint8)
        shifted = Image.merge("HSV", (Image.fromarray(h_arr), s, v)).convert("RGB")
        pulse = 0.75 + 0.5 * math.sin(t * math.pi * 2)
        frames.append(ImageEnhance.Brightness(shifted).enhance(pulse).convert("RGBA"))
    return frames

def efx_kenburns(img, n):
    base = img.convert("RGBA")
    W, H = base.size
    frames = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        zoom = 1 + 0.15 * t
        cx, cy = W * (0.5 - 0.1 * t), H * 0.5
        cw, ch = int(W / zoom), int(H / zoom)
        x0 = max(0, int(cx - cw // 2)); y0 = max(0, int(cy - ch // 2))
        x1 = min(W, x0 + cw);           y1 = min(H, y0 + ch)
        frames.append(base.crop((x0, y0, x1, y1)).resize((W, H), Image.LANCZOS))
    return frames

def efx_matrix(img, n):
    base = img.convert("RGBA")
    W, H = base.size
    chars = "01ACROLITE$#!?"
    rng = random.Random(7)
    cols = max(8, W // 14)
    drops = [rng.randint(0, H // 14) for _ in range(cols)]
    fnt = get_font(13)
    frames = []
    for i in range(n):
        ov = blank(W, H)
        draw = ImageDraw.Draw(ov)
        for c_idx in range(cols):
            x = c_idx * (W // cols)
            y = drops[c_idx] * 14
            draw.text((x, y), rng.choice(chars), fill=(0, 255, 70, 210), font=fnt)
            drops[c_idx] = (drops[c_idx] + 1) % (H // 14 + 5)
        frames.append(Image.alpha_composite(base, ov))
    return frames

def efx_hud(img, n):
    base = img.convert("RGBA")
    W, H = base.size
    fnt = get_font(11)
    frames = []
    for i in range(n):
        t = i / n
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        for y in range(0, H, 4):
            draw.line([(0, y), (W, y)], fill=(0, 255, 100, 15))
        L = 28; col = (0, 255, 100, 200)
        for cx, cy, sx, sy in [(5,5,1,1),(W-5,5,-1,1),(5,H-5,1,-1),(W-5,H-5,-1,-1)]:
            draw.line([(cx, cy), (cx + sx*L, cy)], fill=col, width=2)
            draw.line([(cx, cy), (cx, cy + sy*L)], fill=col, width=2)
        if int(t * 8) % 2 == 0:
            draw.text((W//2 - 38, H - 22), "◉ ACROLITE", fill=(0, 255, 100, 220), font=fnt)
        frames.append(frame)
    return frames

def efx_holographic(img, n):
    base = np.array(img.convert("RGBA"), dtype=np.float32)
    H, W = base.shape[:2]
    ys = np.arange(H, dtype=np.float32)
    frames = []
    for i in range(n):
        t = i / n
        arr = base.copy()
        p = (ys / H + t) % 1.0
        bl = 0.35
        r_row = (0.5 + 0.5 * np.sin(p * 2 * math.pi))[:, np.newaxis] * 255 * bl
        g_row = (0.5 + 0.5 * np.sin(p * 2 * math.pi + 2.094))[:, np.newaxis] * 255 * bl
        b_row = (0.5 + 0.5 * np.sin(p * 2 * math.pi + 4.189))[:, np.newaxis] * 255 * bl
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 - bl) + r_row, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 - bl) + g_row, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - bl) + b_row, 0, 255)
        frames.append(Image.fromarray(arr.astype(np.uint8), "RGBA"))
    return frames

def efx_fire(img, n):
    base = np.array(img.convert("RGBA"), dtype=np.float32)
    H, W = base.shape[:2]
    rng = random.Random(42)
    frames = []
    for i in range(n):
        t = i / n
        arr = base.copy()
        glow = 0.85 + 0.15 * abs(math.sin(t * math.pi * 3))
        arr[:,:,0] = np.clip(arr[:,:,0] * (glow + 0.1), 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * (glow - 0.05), 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * (glow - 0.2),  0, 255)
        fi = Image.fromarray(arr.astype(np.uint8), "RGBA")
        draw = ImageDraw.Draw(fi)
        for _ in range(14):
            ex = rng.randint(0, W-1); ey = rng.randint(int(H*0.3), H-1)
            r  = rng.randint(2, 6);   a  = rng.randint(150, 255)
            draw.ellipse([ex-r, ey-r, ex+r, ey+r],
                         fill=(255, rng.randint(60, 160), 0, a))
        frames.append(fi)
    return frames

def efx_pixelsort(img, n):
    base = np.array(img.convert("RGBA"), dtype=np.uint8)
    H, W = base.shape[:2]
    bri = base[:,:,0]*0.299 + base[:,:,1]*0.587 + base[:,:,2]*0.114
    frames = []
    for i in range(n):
        arr = base.copy()
        cap = min(int(W * i / n), W)
        if cap > 0:
            idx = np.argsort(bri[:, :cap], axis=0)
            arr[:, :cap, :] = arr[idx, np.arange(cap)[np.newaxis, :], :]
        frames.append(Image.fromarray(arr, "RGBA"))
    return frames

def _midpoint_bolt(rng, x1, y1, x2, y2, depth=6):
    """Generate a midpoint-displacement lightning bolt path."""
    pts = [(x1, y1), (x2, y2)]
    for _ in range(depth):
        npts = []
        for j in range(len(pts) - 1):
            ax, ay = pts[j]; bx, by = pts[j + 1]
            mx = (ax + bx) / 2 + rng.uniform(-1, 1) * 0.5 * abs(bx - ax)
            my = (ay + by) / 2 + rng.uniform(-1, 1) * 0.5 * abs(by - ay)
            npts.extend([(ax, ay), (mx, my)])
        npts.append(pts[-1])
        pts = npts
    return [(int(x), int(y)) for x, y in pts]

def efx_lightning(img, n):
    base = img.convert("RGBA")
    W, H = base.size
    rng = random.Random(99)
    frames = []
    for i in range(n):
        f = base.copy()
        if i % 5 in (0, 1):
            draw = ImageDraw.Draw(f)
            pts = _midpoint_bolt(rng, W//2 + rng.randint(-W//4, W//4), 0,
                                 W//2 + rng.randint(-W//8, W//8), H)
            for wd, col in [(4,(200,220,255,80)),(2,(220,240,255,160)),(1,(255,255,255,255))]:
                draw.line(pts, fill=col, width=wd)
        frames.append(f)
    return frames

def efx_vignette(img, n):
    base = img.convert("RGBA")
    W, H = base.size
    ys = np.arange(H)[:, np.newaxis]
    xs = np.arange(W)[np.newaxis, :]
    dist = np.minimum(np.minimum(ys, H - 1 - ys), np.minimum(xs, W - 1 - xs))
    vig = np.clip(dist * 6, 0, 255).astype(np.uint8)
    vig_img = Image.fromarray(vig, "L")
    rng = random.Random(13)
    stars = [(rng.randint(0,W-1), rng.randint(0,H-1), rng.uniform(0,1)) for _ in range(25)]
    frames = []
    for i in range(n):
        t = i / n; f = base.copy()
        dk = Image.new("RGBA", (W, H), (0, 0, 0, 120))
        dk.putalpha(vig_img)
        f = Image.alpha_composite(f, dk)
        draw = ImageDraw.Draw(f)
        for sx, sy, ph in stars:
            bri = int(255 * abs(math.sin((t+ph) * math.pi * 2))); r = 2
            draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(255, 255, 200, bri))
        frames.append(f)
    return frames

def efx_colorgrade(img, n):
    base = np.array(img.convert("RGBA"), dtype=np.float32)
    H, W = base.shape[:2]
    frames = []
    for i in range(n):
        t = i / n; arr = base.copy()
        phase = math.sin(t * math.pi * 2)
        arr[:,:,0] = np.clip(arr[:,:,0] * (1 + 0.15*phase), 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * (1 - 0.10*phase), 0, 255)
        grad = np.linspace(0, 0.25, H)[:, np.newaxis]
        arr[:,:,0] = np.clip(arr[:,:,0] + grad*50, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] + grad*80, 0, 255)
        frames.append(Image.fromarray(arr.astype(np.uint8), "RGBA"))
    return frames

def efx_zoompulse(img, n):
    base = img.convert("RGBA")
    W, H = base.size
    frames = []
    for i in range(n):
        t = i / n
        scale = 1 + 0.08 * math.sin(t * math.pi * 2)
        nw, nh = int(W * scale), int(H * scale)
        zoomed = base.resize((nw, nh), Image.LANCZOS)
        x = (nw - W) // 2; y = (nh - H) // 2
        frames.append(zoomed.crop((x, y, x+W, y+H)))
    return frames

def efx_tvstatic(img, n):
    base = np.array(img.convert("RGBA"), dtype=np.float32)
    H, W = base.shape[:2]
    rng = random.Random(3)
    frames = []
    for i in range(n):
        arr = base.copy()
        noise = np.random.randint(0, 40, (H, W, 1), dtype=np.int16)
        arr[:,:,:3] = np.clip(arr[:,:,:3] + noise, 0, 255)
        if rng.random() < 0.4:
            y = rng.randint(0, H-8); h = rng.randint(2, 8)
            arr[y:y+h, :, :3] = np.clip(
                arr[y:y+h, :, :3] + rng.randint(-60, 60), 0, 255)
        frames.append(Image.fromarray(arr.astype(np.uint8), "RGBA"))
    return frames

def efx_ripple(img, n):
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        return efx_none(img, n)
    base = np.array(img.convert("RGBA"), dtype=np.float32)
    H, W = base.shape[:2]
    cx, cy = W // 2, H // 2
    ys, xs = np.mgrid[0:H, 0:W]
    frames = []
    for i in range(n):
        t = i / n
        dist   = np.sqrt((xs-cx)**2 + (ys-cy)**2)
        angle  = dist/30 - t*2*math.pi
        strength = np.clip(18*(1 - dist/max(W,H)), 0, 18)
        sx = np.clip(xs + strength*np.cos(angle), 0, W-1)
        sy = np.clip(ys + strength*np.sin(angle), 0, H-1)
        channels = [
            map_coordinates(base[:,:,c], [sy, sx], order=1, mode='reflect')
            for c in range(4)
        ]
        arr = np.stack(channels, axis=2)
        frames.append(Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGBA"))
    return frames

# Registry
EFFECTS: dict[str, tuple[str, callable]] = {
    "none":        ("No effect (static frames)",  efx_none),
    "glitch":      ("Glitch Scan",                efx_glitch),
    "neon":        ("Neon Hue Cycle",             efx_neon),
    "kenburns":    ("Ken Burns Zoom + Pan",       efx_kenburns),
    "matrix":      ("Matrix Code Rain",           efx_matrix),
    "hud":         ("HUD Scanlines + Brackets",   efx_hud),
    "holographic": ("Holographic Foil Shimmer",   efx_holographic),
    "fire":        ("Fire Glow + Embers",         efx_fire),
    "pixelsort":   ("Pixel Sort Sweep",           efx_pixelsort),
    "lightning":   ("Lightning Strike",           efx_lightning),
    "vignette":    ("Vignette + Twinkling Stars", efx_vignette),
    "colorgrade":  ("Cinematic Color Grade",      efx_colorgrade),
    "zoompulse":   ("Zoom Pulse",                 efx_zoompulse),
    "tvstatic":    ("TV Static / Noise",          efx_tvstatic),
    "ripple":      ("Liquid Ripple Warp",         efx_ripple),
}

# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAYS  — each fn(frames, seed=int) modifies frames list in-place
# ══════════════════════════════════════════════════════════════════════════════

def _star_shape(draw, cx, cy, r, fill):
    pts = []
    for k in range(5):
        ao = math.pi/2 + k * 2*math.pi/5
        ai = ao + math.pi/5
        pts += [(cx + r*math.cos(ao), cy - r*math.sin(ao)),
                (cx + (r/2.5)*math.cos(ai), cy - (r/2.5)*math.sin(ai))]
    draw.polygon(pts, fill=fill)

def ovl_sparkles(frames, seed=42):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    sparks = [(rng.randint(5,W-5), rng.randint(5,H-5), rng.uniform(0,1)) for _ in range(22)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for sx, sy, ph in sparks:
            bri = abs(math.sin((t+ph)*math.pi*4))
            if bri > 0.45:
                r = int(2+5*bri); a = int(255*bri)
                for dx, dy in [(r,0),(-r,0),(0,r),(0,-r)]:
                    draw.line([(sx,sy),(sx+dx,sy+dy)], fill=(255,255,200,a), width=1)
                for dx, dy in [(r//2,r//2),(-r//2,-r//2),(r//2,-r//2),(-r//2,r//2)]:
                    draw.line([(sx,sy),(sx+dx,sy+dy)], fill=(255,255,220,a//2), width=1)

def ovl_stars(frames, seed=13):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    stars = [(rng.randint(0,W-1), rng.randint(0,H-1), rng.uniform(0,1)) for _ in range(35)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for sx, sy, ph in stars:
            bri = int(255*abs(math.sin((t+ph)*math.pi*2))); r = 2
            draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(255,255,200,bri))

def ovl_fire_particles(frames, seed=7):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    particles = [(rng.randint(0,W-1), rng.uniform(0,1)) for _ in range(25)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for px, phase in particles:
            py = H - int(((t+phase)%1.0) * H * 1.2)
            if 0 <= py < H:
                a = rng.randint(150,255); r = rng.randint(2,5)
                draw.ellipse([px-r, py-r, px+r, py+r],
                             fill=(255, rng.randint(60,160), 0, a))

def ovl_snow(frames, seed=3):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    flakes = [(rng.randint(0,W-1), rng.uniform(0,1), rng.uniform(0.3,1.0)) for _ in range(40)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for fx, phase, speed in flakes:
            fy = int(((t*speed+phase)%1.0) * H); r = rng.randint(1,3)
            draw.ellipse([fx-r, fy-r, fx+r, fy+r], fill=(255,255,255,rng.randint(150,230)))

def ovl_confetti(frames, seed=5):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    cols = [(255,50,50),(50,255,50),(50,50,255),(255,255,50),(255,50,255),(50,255,255)]
    pieces = [(rng.randint(0,W), rng.uniform(0,1), rng.choice(cols), rng.uniform(-15,15))
              for _ in range(30)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for px, phase, col, angle in pieces:
            py = int(((t+phase)%1.0) * H)
            r = 6; a_rad = math.radians(angle + t*180)
            dx = r*math.cos(a_rad); dy = r*math.sin(a_rad)
            draw.polygon([(px-dx,py-dy),(px+dx,py+dy),(px+dy,py-dx),(px-dy,py+dx)],
                         fill=col+(200,))

def ovl_lightning_flash(frames, seed=99):
    rng = random.Random(seed)
    W, H = frames[0].size
    for i, f in enumerate(frames):
        if i % 6 in (0, 1):
            draw = ImageDraw.Draw(f)
            pts = _midpoint_bolt(rng, rng.randint(W//4, 3*W//4), 0,
                                 rng.randint(W//4, 3*W//4), H, depth=5)
            for wd, col in [(4,(200,220,255,60)),(2,(220,240,255,140)),(1,(255,255,255,255))]:
                draw.line(pts, fill=col, width=wd)

def ovl_coins(frames, seed=17):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    fnt = get_font(10)
    coins = [(rng.randint(12,W-12), rng.uniform(0,1)) for _ in range(5)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for cx, phase in coins:
            cy = int(H*0.85 - abs(math.sin((t+phase)*math.pi*2)) * H*0.3)
            r = 11
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(255,215,0,220),
                         outline=(200,150,0,255), width=1)
            draw.text((cx-4, cy-6), "$", fill=(140,80,0,255), font=fnt)

def ovl_hearts(frames, seed=21):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    def draw_heart(draw, cx, cy, r, col):
        draw.ellipse([cx-r, cy-r, cx, cy+r//2],    fill=col)
        draw.ellipse([cx,   cy-r, cx+r, cy+r//2],  fill=col)
        draw.polygon([(cx-r,cy+r//3),(cx+r,cy+r//3),(cx,cy+r+r//2)], fill=col)
    hearts = [(rng.randint(10,W-10), rng.uniform(0,1)) for _ in range(8)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for hx, phase in hearts:
            hy = H - int(((t*0.5+phase)%1.0) * H * 1.2)
            if 0 <= hy < H:
                r = int(5 + 3*abs(math.sin((t+phase)*math.pi*2)))
                draw_heart(draw, hx, hy, r, (255,50,100,200))

def ovl_fireworks(frames, seed=31):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    bursts = [(rng.randint(W//5, 4*W//5), rng.randint(H//5, 2*H//3), rng.uniform(0,1))
              for _ in range(4)]
    cols = [(255,100,50),(100,255,50),(50,100,255),(255,255,50),(255,50,255)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for bx, by, phase in bursts:
            bt = (t+phase) % 1.0
            if bt < 0.5:
                prog = bt/0.5; r = int(prog*35); a = int(255*(1-prog))
                col = rng.choice(cols)
                for ang in range(0, 360, 30):
                    rad = math.radians(ang)
                    ex, ey = int(bx+r*math.cos(rad)), int(by+r*math.sin(rad))
                    draw.line([(bx,by),(ex,ey)], fill=col+(a,), width=1)
                    draw.ellipse([ex-2,ey-2,ex+2,ey+2], fill=col+(a,))

def ovl_rain(frames, seed=41):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    drops = [(rng.randint(0,W), rng.uniform(0,1), rng.randint(8,20)) for _ in range(50)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for rx, phase, length in drops:
            ry = int(((t+phase)%1.0) * H)
            draw.line([(rx,ry),(rx-2,ry+length)], fill=(150,200,255,rng.randint(80,150)), width=1)

def ovl_bubbles(frames, seed=51):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    bubs = [(rng.randint(10,W-10), rng.uniform(0,1), rng.randint(5,18)) for _ in range(15)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for bx, phase, r in bubs:
            by = H - int(((t+phase)%1.0) * H*1.1)
            if 0 <= by < H:
                a = rng.randint(80,160)
                draw.ellipse([bx-r, by-r, bx+r, by+r], outline=(150,210,255,a), width=1)
                draw.ellipse([bx-r//3, by-r//2, bx, by-r//4], fill=(255,255,255,a//2))

def ovl_smoke(frames, seed=61):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    puffs = [(rng.randint(W//4, 3*W//4), rng.uniform(0,1), rng.randint(8,20)) for _ in range(12)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for px, phase, r in puffs:
            prog = ((t+phase) % 1.0)
            py = H - int(prog * H*1.2)
            drift = int(prog * 20 * math.sin(phase*10))
            if 0 <= py < H:
                cr = int(r*(0.5+prog)); a = int(60*(1-prog))
                draw.ellipse([px+drift-cr, py-cr, px+drift+cr, py+cr], fill=(180,180,180,a))

def ovl_arrows(frames, seed=71):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    fnt = get_font(22)
    arrow_chars = ["↑","↗","→","↘","↓","↙","←","↖"]
    positions = [(rng.randint(W//4, 3*W//4), rng.randint(H//4, 3*H//4)) for _ in range(3)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for k, (ax, ay) in enumerate(positions):
            idx = int((t*8 + k*2.67)) % 8
            a = int(180 + 75*abs(math.sin(t*math.pi*2 + k)))
            draw.text((ax, ay), arrow_chars[idx], fill=(255,220,0,a), font=fnt)

def ovl_crosshair(frames, seed=81):
    n = len(frames); W, H = frames[0].size
    cx, cy = W//2, H//2
    fnt = get_font(11)
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        r = int(20 + 5*math.sin(t*math.pi*2))
        a = int(200 + 55*math.sin(t*math.pi*4))
        col = (0, 255, 100, a)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=col, width=1)
        gap = r + 5
        draw.line([(cx-r-15, cy), (cx-gap, cy)], fill=col, width=1)
        draw.line([(cx+gap,  cy), (cx+r+15, cy)], fill=col, width=1)
        draw.line([(cx, cy-r-15), (cx, cy-gap)],  fill=col, width=1)
        draw.line([(cx, cy+gap),  (cx, cy+r+15)], fill=col, width=1)
        if int(t*6) % 2 == 0:
            draw.text((cx+r+6, cy-6), "◎", fill=col, font=fnt)

def ovl_emoji_party(frames, seed=91):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    fnt = get_font(22)
    symbols = ["✨","⭐","💥","💫","🌟","⚡","★","✦","✵","❋"]
    positions = [(rng.randint(5, max(5, W-30)), rng.randint(5, max(5, H-30)), rng.uniform(0,1))
                 for _ in range(8)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for ex, ey, phase in positions:
            if abs(math.sin((t+phase)*math.pi*3)) > 0.5:
                draw.text((ex, ey), rng.choice(symbols), font=fnt)

def ovl_neon_border(frames, seed=0):
    n = len(frames); W, H = frames[0].size
    for i, f in enumerate(frames):
        t = i/n; ph = t*2*math.pi
        r = int(255*(0.5+0.5*math.sin(ph)))
        g = int(255*(0.5+0.5*math.sin(ph+2.094)))
        b = int(255*(0.5+0.5*math.sin(ph+4.189)))
        draw = ImageDraw.Draw(f)
        for w in range(4):
            draw.rectangle([w, w, W-w-1, H-w-1], outline=(r, g, b, 200-w*40))

def ovl_node_pulse(frames, seed=55):
    rng = random.Random(seed); n = len(frames)
    W, H = frames[0].size
    nodes = [(int(W*(0.15+0.175*c_)), int(H*(0.2+0.2*r_)))
             for r_ in range(3) for c_ in range(4)]
    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        for k, (nx, ny) in enumerate(nodes):
            phase = (t + k/len(nodes)) % 1.0
            radius = int(6 + 6*abs(math.sin(phase*math.pi*2)))
            a = int(180 + 75*math.sin(phase*math.pi*2))
            draw.ellipse([nx-radius, ny-radius, nx+radius, ny+radius],
                         outline=(0, 200+rng.randint(0,55), 255, a), width=2)

# Registry
OVERLAYS: dict[str, tuple[str, callable]] = {
    "sparkles":      ("✨  Sparkles",           ovl_sparkles),
    "stars":         ("⭐  Twinkling Stars",    ovl_stars),
    "fire_particles":("🔥  Fire Particles",     ovl_fire_particles),
    "snow":          ("❄   Snow",               ovl_snow),
    "confetti":      ("🎊  Confetti",            ovl_confetti),
    "lightning":     ("⚡  Lightning Flash",    ovl_lightning_flash),
    "coins":         ("💰  Bouncing Coins",      ovl_coins),
    "hearts":        ("❤   Floating Hearts",    ovl_hearts),
    "fireworks":     ("🎆  Fireworks Burst",     ovl_fireworks),
    "rain":          ("🌧   Rain",               ovl_rain),
    "bubbles":       ("🫧   Bubbles",            ovl_bubbles),
    "smoke":         ("💨  Smoke",               ovl_smoke),
    "arrows":        ("↑   Animated Arrows",    ovl_arrows),
    "crosshair":     ("🎯  Crosshair / HUD",    ovl_crosshair),
    "emoji_party":   ("🎉  Emoji Party",         ovl_emoji_party),
    "neon_border":   ("🟣  Neon Border",         ovl_neon_border),
    "node_pulse":    ("◎   Node Pulse",          ovl_node_pulse),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TEXT ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════

TEXT_STYLES = ["typewriter", "neon_glow", "glitch_text", "bounce", "scroll", "fade"]

def apply_text(frames, text: str, style="neon_glow",
               color=(255,255,100), size=22, position="bottom"):
    if not text:
        return
    n = len(frames); W, H = frames[0].size
    fnt = get_font(size)
    # measure
    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    try:
        bbox = tmp_draw.textbbox((0, 0), text, font=fnt)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = len(text) * size // 2, size

    def base_pos():
        bx = W//2 - tw//2
        if position == "bottom":  by = H - th - 14
        elif position == "top":   by = 10
        else:                     by = H//2 - th//2
        return bx, by

    for i, f in enumerate(frames):
        t = i/n; draw = ImageDraw.Draw(f)
        bx, by = base_pos()

        if style == "typewriter":
            chars = max(1, int(t * len(text) * 1.5) % len(text) + 1)
            draw.text((bx, by), text[:chars], fill=color+(220,), font=fnt)

        elif style == "neon_glow":
            a = int(180 + 75*math.sin(t*math.pi*4))
            for ox, oy in [(1,1),(-1,-1),(1,-1),(-1,1),(0,2),(0,-2),(2,0),(-2,0)]:
                draw.text((bx+ox, by+oy), text, fill=(255,255,255,a//4), font=fnt)
            draw.text((bx, by), text, fill=color+(a,), font=fnt)

        elif style == "glitch_text":
            shift = int(math.sin(t*math.pi*8)*5)
            draw.text((bx+shift, by), text, fill=(255,0,0,100),   font=fnt)
            draw.text((bx-shift, by), text, fill=(0,255,255,100), font=fnt)
            draw.text((bx, by),        text, fill=color+(220,),   font=fnt)

        elif style == "bounce":
            by2 = by - int(abs(math.sin(t*math.pi*2))*18)
            draw.text((bx, by2), text, fill=color+(220,), font=fnt)

        elif style == "scroll":
            sx = W - int(t * (W + tw*2))
            draw.text((sx, by), text, fill=color+(200,), font=fnt)

        elif style == "fade":
            a = int(255*abs(math.sin(t*math.pi*2)))
            draw.text((bx, by), text, fill=color+(a,), font=fnt)

# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate(cfg: dict) -> tuple[str, int]:
    src = cfg["source_path"]
    print(f"\n  Loading {c(os.path.basename(src), CYN)}...")
    img = load_image(src)
    img = resize_fit(img, cfg.get("max_side", 480))
    n   = cfg.get("n_frames", 20)

    efx_key = cfg.get("effect", "none") or "none"
    print(f"  Applying effect: {c(efx_key, YEL)}  ({n} frames)...")
    efx_fn = EFFECTS.get(efx_key, EFFECTS["none"])[1]
    frames = efx_fn(img, n)

    for ovl_key in cfg.get("overlays", []):
        if ovl_key in OVERLAYS:
            print(f"  Adding overlay: {c(ovl_key, YEL)}...")
            OVERLAYS[ovl_key][1](frames)

    if cfg.get("text"):
        print(f"  Adding text: {c(repr(cfg['text']), YEL)}  [{cfg.get('text_style','neon_glow')}]...")
        apply_text(
            frames, cfg["text"],
            style=cfg.get("text_style", "neon_glow"),
            color=cfg.get("text_color", (255,255,100)),
            size=cfg.get("text_size", 22),
            position=cfg.get("text_position", "bottom"),
        )

    out = cfg.get("output_path") or _auto_output(cfg)
    print(f"  Optimizing → {c(out, GRN)} ...")
    sz, side, nf, cols = save_optimized(frames, out, dur=cfg.get("duration", 80))
    print(f"  {c('✓', GRN)} {sz//1024}KB  |  {nf} frames  |  {side}px  |  {cols} colors")
    return out, sz

def _auto_output(cfg: dict) -> str:
    base = os.path.splitext(cfg["source_path"])[0]
    efx  = cfg.get("effect", "animated") or "animated"
    ovls = "-".join(cfg.get("overlays", []))
    suffix = f"-{efx}" + (f"-{ovls[:20]}" if ovls else "")
    return base + suffix + ".gif"

# ══════════════════════════════════════════════════════════════════════════════
#  PRESET / LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def load_presets() -> dict:
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}

def save_preset(name: str, cfg: dict):
    presets = load_presets()
    safe = {k: v for k, v in cfg.items() if k not in ("source_path",)}
    presets[name] = safe
    os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
    with open(PRESETS_FILE, "w") as fh:
        json.dump(presets, fh, indent=2)
    print(f"  {c('✓', GRN)} Preset '{name}' saved to {PRESETS_FILE}")

def list_presets():
    presets = load_presets()
    if not presets:
        print(f"  {c('(no saved presets)', DIM)}")
        return
    for k, v in presets.items():
        efx  = v.get("effect","none")
        ovls = ", ".join(v.get("overlays",[]))
        txt  = v.get("text","")
        print(f"  {c(k, YEL)} — effect:{efx}  overlays:{ovls or '-'}  text:{txt or '-'}")

# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════════════════════

def banner():
    print(c(BLD, BLD))
    print(c("╔══════════════════════════════════════════════════════════╗", CYN))
    print(c(f"║       🎨  GIF MAKER  v{VERSION}  —  Acrolite  Tools            ║", CYN))
    print(c("║    Turn still images into animated GIFs, from the CLI.   ║", CYN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYN))
    print(RST)

def show_status(cfg: dict):
    src  = cfg.get("source_path")
    W_, H_ = "?", "?"
    if src and os.path.exists(src):
        try: im=Image.open(src); W_,H_=im.size
        except Exception: pass
    src_label  = c(os.path.basename(src), GRN) if src else c("(not set)", RED)
    efx_label  = c(cfg.get("effect","none") or "none", CYN)
    ovl_label  = c(", ".join(cfg.get("overlays",[])) or "–", CYN)
    txt_label  = c(f"'{cfg['text']}' [{cfg.get('text_style','neon_glow')}]", CYN) if cfg.get("text") else c("–", DIM)
    out_label  = c(os.path.basename(cfg.get("output_path","(auto)")), GRN)
    print(hr())
    print(f"  {c('Source',BLD)}   {src_label}  {c(f'({W_}×{H_})', DIM)}")
    print(f"  {c('Effect',BLD)}   {efx_label}")
    print(f"  {c('Overlays',BLD)} {ovl_label}")
    print(f"  {c('Text',BLD)}     {txt_label}")
    print(f"  {c('Settings',BLD)} {cfg.get('max_side',480)}px · "
          f"{cfg.get('n_frames',20)} frames · "
          f"{cfg.get('duration',80)}ms/frame")
    print(f"  {c('Output',BLD)}   {out_label}")
    print(hr())

def show_effects_menu():
    print(f"\n{BLD}─── EFFECTS ({'─'*(BOX_W-12)}){RST}")
    for i, (k,(label,_)) in enumerate(EFFECTS.items()):
        print(f"  {c(f'{i:>2}',YEL)}  {label:<35}  {c(k,DIM)}")
    print()

def show_overlays_menu(active: list):
    print(f"\n{BLD}─── OVERLAYS ({'─'*(BOX_W-13)}){RST}")
    keys = list(OVERLAYS.keys())
    for i, (k,(label,_)) in enumerate(OVERLAYS.items()):
        letter = chr(65+i)
        tick   = c("✓", GRN) if k in active else " "
        print(f"  {c(letter,YEL)} {tick}  {label:<38}  {c(k,DIM)}")
    print()

def show_main_menu(cfg: dict):
    clr(); banner(); show_status(cfg)
    print(f"\n  {BLD}MAIN MENU{RST}")
    items = [
        ("1", "Set source image"),
        ("2", "Choose effect"),
        ("3", "Add / remove overlays"),
        ("4", "Set animated text"),
        ("5", "Settings  (size · frames · speed)"),
        ("6", "Set output filename"),
        ("7", "Save as preset"),
        ("8", "Load preset"),
        ("L", "List all effects & overlays"),
        ("G", c("Generate GIF!", GRN+BLD)),
        ("Q", "Quit"),
    ]
    for key, label in items:
        print(f"  {c(key,YEL)}  {label}")
    print()

def prompt(msg: str, default=None) -> str:
    suffix = f" {c(f'[{default}]',DIM)}" if default is not None else ""
    try:
        val = input(f"  {c('▶',CYN)} {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default or ""
    return val if val else (default or "")

def pause():
    try: input(f"  {c('Press Enter to continue…',DIM)}")
    except (EOFError, KeyboardInterrupt): pass

# ── handlers ──────────────────────────────────────────────────────────────────

_PRESET_ALLOWED_KEYS = {
    "effect", "overlays", "text", "text_style", "text_color",
    "text_size", "text_position", "max_side", "n_frames", "duration",
    "output_path",
}

def _prompt_int(msg: str, default: int, min_val: int = 1, max_val: int = 10_000) -> int:
    """Prompt for an integer, retrying on invalid input or out-of-range values."""
    while True:
        raw = prompt(msg, default=str(default))
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  {c(f'Must be between {min_val} and {max_val}.', RED)}")
        except ValueError:
            print(f"  {c('Please enter a whole number.', RED)}")

def handle_source(cfg: dict):
    p = prompt("Image path").strip("'\"")
    resolved = os.path.realpath(p)
    if os.path.isfile(resolved):
        cfg["source_path"] = resolved
        base = os.path.splitext(resolved)[0]
        cfg.setdefault("output_path", base + "-animated.gif")
        print(f"  {c('✓',GRN)} Loaded.")
    else:
        print(f"  {c('✗ File not found',RED)}: {p}")
    pause()

def handle_effect(cfg: dict):
    show_effects_menu()
    choice = prompt("Effect number", default="0")
    try:
        key = list(EFFECTS.keys())[int(choice)]
        cfg["effect"] = key
        print(f"  {c('✓',GRN)} Effect: {EFFECTS[key][0]}")
    except Exception:
        print(f"  {c('Invalid choice',RED)}")
    pause()

def handle_overlays(cfg: dict):
    while True:
        clr(); banner()
        active = cfg.setdefault("overlays", [])
        show_overlays_menu(active)
        print(f"  Active: {c(', '.join(active) or '(none)', CYN)}")
        print(f"\n  Enter {c('letter',YEL)} to toggle  |  {c('D',YEL)} clear all  |  {c('Enter',YEL)} back")
        ch = prompt("").upper()
        if ch == "":
            break
        if ch == "D":
            cfg["overlays"] = []
            continue
        idx = ord(ch) - 65
        keys = list(OVERLAYS.keys())
        if 0 <= idx < len(keys):
            k = keys[idx]
            if k in active: active.remove(k);  print(f"  Removed: {k}")
            else:            active.append(k);  print(f"  Added:   {k}")
        else:
            print(f"  {c('Invalid',RED)}")

def handle_text(cfg: dict):
    txt = prompt("Animated text (blank to remove)")
    if not txt:
        cfg["text"] = None; print(f"  {c('Text cleared.',DIM)}"); pause(); return
    cfg["text"] = txt
    print(f"\n  Styles:")
    for i, s in enumerate(TEXT_STYLES):
        print(f"    {c(i,YEL)}  {s}")
    style_idx = _prompt_int("Style number", 1, 0, len(TEXT_STYLES) - 1)
    cfg["text_style"] = TEXT_STYLES[style_idx]
    cfg["text_size"] = _prompt_int("Font size px", 22, 8, 120)
    pos = prompt("Position (bottom/top/center)", default="bottom")
    cfg["text_position"] = pos if pos in ("bottom","top","center") else "bottom"
    print(f"  {c('✓',GRN)} Text configured.")
    pause()

def handle_settings(cfg: dict):
    cfg["max_side"] = _prompt_int("Max side px", cfg.get("max_side", 480), 32, 4096)
    cfg["n_frames"] = _prompt_int("Frame count", cfg.get("n_frames", 20), 1, 120)
    cfg["duration"] = _prompt_int("ms per frame", cfg.get("duration", 80), 10, 5000)
    print(f"  {c('✓',GRN)} Settings updated.")
    pause()

def handle_output(cfg: dict):
    p = prompt("Output path", default=cfg.get("output_path","output.gif"))
    if p: cfg["output_path"] = p
    pause()

def handle_generate(cfg: dict):
    if not cfg.get("source_path") or not os.path.exists(cfg.get("source_path","")):
        print(f"  {c('✗ No valid source image.',RED)}"); pause(); return
    try:
        out, sz = generate(cfg)
        print(f"\n  {c('🎉 GIF saved: ' + out, GRN+BLD)}  ({sz//1024}KB)")
    except Exception as e:
        print(f"\n  {c('✗ Error: ' + str(e), RED)}")
        import traceback; traceback.print_exc()
    pause()

def handle_save_preset(cfg: dict):
    name = prompt("Preset name")
    if name: save_preset(name, cfg)
    pause()

def handle_load_preset(cfg: dict):
    presets = load_presets()
    if not presets:
        print(f"  {c('No saved presets.',YEL)}"); pause(); return
    list_presets()
    names = list(presets.keys())
    ch = prompt("Load preset (name or number)")
    try:
        p = presets[names[int(ch)]] if ch.isdigit() else presets[ch]
        for k, v in p.items():
            if k in _PRESET_ALLOWED_KEYS:
                cfg[k] = v
        print(f"  {c('✓',GRN)} Preset loaded.")
    except Exception:
        print(f"  {c('Not found.',RED)}")
    pause()

def handle_list(cfg: dict):
    clr(); banner()
    show_effects_menu()
    show_overlays_menu(cfg.get("overlays",[]))
    pause()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def default_cfg() -> dict:
    return {
        "source_path": None, "effect": "none", "overlays": [],
        "text": None, "text_style": "neon_glow", "text_color": (255,255,100),
        "text_size": 22, "text_position": "bottom",
        "max_side": 480, "n_frames": 20, "duration": 80,
        "output_path": None,
    }

def main():
    ap = argparse.ArgumentParser(
        description=f"GIF Maker v{VERSION} — Acrolite Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("image",      nargs="?", help="Source image path")
    ap.add_argument("-e","--effect",   default="none",  choices=list(EFFECTS.keys()))
    ap.add_argument("-o","--overlay",  action="append", dest="overlays", default=[],
                    metavar="OVERLAY", choices=list(OVERLAYS.keys()))
    ap.add_argument("-t","--text",     default=None)
    ap.add_argument("-s","--text-style",default="neon_glow", choices=TEXT_STYLES)
    ap.add_argument("-p","--text-pos", default="bottom", choices=["bottom","top","center"])
    ap.add_argument("--frames",        type=int, default=20)
    ap.add_argument("--size",          type=int, default=480)
    ap.add_argument("--duration",      type=int, default=80)
    ap.add_argument("-O","--output",   default=None)
    ap.add_argument("-n","--no-interactive", action="store_true",
                    help="Non-interactive: generate immediately and exit")
    ap.add_argument("--list-effects",  action="store_true")
    ap.add_argument("--list-overlays", action="store_true")
    ap.add_argument("--list-presets",  action="store_true")
    ap.add_argument("--version",       action="version", version=f"GIF Maker v{VERSION}")
    args = ap.parse_args()

    if args.list_effects:
        print("\nEffects:"); [print(f"  {k:<14} {v[0]}") for k,v in EFFECTS.items()]; return
    if args.list_overlays:
        print("\nOverlays:"); [print(f"  {k:<18} {v[0]}") for k,v in OVERLAYS.items()]; return
    if args.list_presets:
        print("\nSaved presets:"); list_presets(); return

    cfg = default_cfg()
    if args.image:         cfg["source_path"] = args.image
    cfg["effect"]        = args.effect
    cfg["overlays"]      = args.overlays
    cfg["text"]          = args.text
    cfg["text_style"]    = args.text_style
    cfg["text_position"] = args.text_pos
    cfg["n_frames"]      = args.frames
    cfg["max_side"]      = args.size
    cfg["duration"]      = args.duration
    cfg["output_path"]   = args.output

    # non-interactive
    if args.no_interactive:
        if not cfg["source_path"]:
            print("Error: provide an image path.", file=sys.stderr); sys.exit(1)
        if not cfg["output_path"]:
            cfg["output_path"] = _auto_output(cfg)
        generate(cfg); return

    # interactive loop
    while True:
        show_main_menu(cfg)
        try:
            ch = input(f"  {c('▶',CYN)} Choice: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {c('Goodbye! 🦆',YEL)}"); break
        if   ch == "1": handle_source(cfg)
        elif ch == "2": handle_effect(cfg)
        elif ch == "3": handle_overlays(cfg)
        elif ch == "4": handle_text(cfg)
        elif ch == "5": handle_settings(cfg)
        elif ch == "6": handle_output(cfg)
        elif ch == "7": handle_save_preset(cfg)
        elif ch == "8": handle_load_preset(cfg)
        elif ch == "L": handle_list(cfg)
        elif ch == "G": handle_generate(cfg)
        elif ch == "Q": print(f"\n  {c('Goodbye! 🦆',YEL)}"); break
        else:           print(f"  {c('Unknown option',RED)}"); pause()

if __name__ == "__main__":
    main()
