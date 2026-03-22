# GIF Maker — Acrolite Tools

An interactive CLI tool that turns still images into animated GIFs, built in pure Python.

## Features

### 15 Effects
| Key | Effect |
|-----|--------|
| `none` | Static (no effect) |
| `glitch` | RGB channel offset + scan-line pulse |
| `neon` | Hue-cycling neon glow |
| `kenburns` | Ken Burns slow zoom + pan |
| `matrix` | Matrix code rain overlay |
| `hud` | HUD scan-lines + corner brackets |
| `holographic` | Rainbow foil shimmer |
| `fire` | Warm glow + ember particles |
| `pixelsort` | Pixel-sort column sweep |
| `lightning` | Midpoint-displacement lightning |
| `vignette` | Dark vignette + twinkling stars |
| `colorgrade` | Cinematic teal-orange color grade |
| `zoompulse` | Rhythmic zoom pulse |
| `tvstatic` | TV noise / static bars |
| `ripple` | Liquid ripple warp (requires scipy) |

### 17 Overlays (stack as many as you like)
✨ Sparkles · ⭐ Twinkling Stars · 🔥 Fire Particles · ❄ Snow · 🎊 Confetti  
⚡ Lightning Flash · 💰 Bouncing Coins · ❤ Floating Hearts · 🎆 Fireworks  
🌧 Rain · 🫧 Bubbles · 💨 Smoke · ↑ Animated Arrows · 🎯 Crosshair/HUD  
🎉 Emoji Party · 🟣 Neon Border · ◎ Node Pulse

### 6 Animated Text Styles
`typewriter` · `neon_glow` · `glitch_text` · `bounce` · `scroll` · `fade`

### Other
- Auto-optimization to keep GIFs under **2 MB**
- Save / load **named presets** (`~/.gif-maker/presets.json`)
- Fully **non-interactive** CLI mode for scripting

---

## Requirements

```bash
pip install Pillow numpy scipy
```

Python 3.10+ required.

---

## Usage

### Interactive menu
```bash
python3 gif-maker.py
python3 gif-maker.py myimage.png   # pre-load an image
```

### Non-interactive (one-liner)
```bash
python3 gif-maker.py image.png -e glitch -o sparkles -o coins -t "ACROLITE" -n
```

### All CLI flags
```
positional:
  image                 Source image path

options:
  -e, --effect          Effect to apply (default: none)
  -o, --overlay         Overlay to add (repeatable)
  -t, --text            Animated text to overlay
  -s, --text-style      Text animation style (default: neon_glow)
  -p, --text-pos        Text position: bottom | top | center
  --frames              Number of frames (default: 20)
  --size                Max side in pixels (default: 480)
  --duration            Ms per frame (default: 80)
  -O, --output          Output file path
  -n, --no-interactive  Generate and exit immediately

  --list-effects        Print all available effects
  --list-overlays       Print all available overlays
  --list-presets        Print saved presets
```

### Examples
```bash
# Neon hue cycle + sparkles + scrolling text
python3 gif-maker.py logo.png -e neon -o sparkles -t "ACROLITE" -s scroll -n

# Trading-floor HUD with coins
python3 gif-maker.py trading.png -e hud -o coins -o crosshair -n

# Holographic foil + fireworks + neon border
python3 gif-maker.py duck.png -e holographic -o fireworks -o neon_border -n

# Matrix rain on a flow diagram
python3 gif-maker.py diagram.png -e matrix -o node_pulse -n
```

---

## Presets

In interactive mode, press **7** to save your current settings as a named preset, and **8** to reload any preset later. Presets are stored in `~/.gif-maker/presets.json`.

---

## License

MIT
