#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 HEARNOCULAR  --  Directional Hearing & Distance Perception Digital Twin
================================================================================

A 100% standalone monolith that captures, processes, visualizes and enhances
distant audio in real time using an 11-stage DSP pipeline.  Built as the
software counterpart to the "AetherEar" parabolic directional-hearing concept.

Four modes (cycle with TAB):

  1. SPECTRUM   Real-time frequency spectrum + spectrogram waterfall.
                Watch the voice band light up as sound arrives.  The
                enhancement pipeline runs live.  Input vs output
                waveforms shown side-by-side.  Features: dB scale on
                Y-axis, peak hold markers, frequency cursor (click to
                read), clipping indicator, confidence gauge bar, noise
                floor estimate (dashed line), pitch overlay (F0 +
                harmonics), spectral gain reduction overlay (red
                shading), spectrogram waterfall with frequency axis
                labels and color bar legend, voice band highlight.

  2. DIRECTION  Spectral energy map.  Voice-band frequency sub-bands are
                mapped to a 360-degree compass for visualization.
                NOTE: single microphone -- NOT true direction-of-arrival.
                Distance rings, confidence ring, pitch detection, and
                A-weighted dB readout.

  3. FILTER     The full 11-stage processing pipeline visualized stage-
                by-stage: raw -> wind -> bandpass -> gate -> de-reverb ->
                decoherence -> Wiener -> compressor -> presence -> formant
                -> super-resolution + loudness -> output.  Each stage
                shows its waveform, mini spectrum, RMS level, and
                inter-stage gain/attenuation indicator.

  4. METER      Professional audio metering: dual input/output VU/PPM
                meters with dB scale and 3-color zones, peak hold with
                slow decay, total gain reduction marker, headroom
                indicator, 3-band gain reduction meters, quality metrics
                panel (A-weighted, centroid, flatness, SII, VAD, wind,
                clipping, confidence, pitch, processing latency),
                confidence history graph, and feature toggle status bar.

  5. BLUEPRINT   3D rotatable hardware model of the physical Hearnocular
                device (100+ mesh groups). Software-rendered with perspective
                projection, Lambert shading, backface culling, depth-sorted
                alpha blending, edge rendering, and 3D label projection with
                leader lines. Drag to orbit, wheel to zoom, Shift+drag to
                pan, R to reset, L to toggle labels, E for exploded view,
                W for wireframe, A for auto-rotate, F to focus/highlight
                individual parts, T to toggle edges. Shows parabolic dish
                (48x14 paraboloid + torus rim + hinges + hub + spider vanes
                + backing ribs + magnetic locks), focal mic (suspension cage
                + grille + windscreen + XLR + strain relief), 8x MEMS array
                (PCB ring + traces + solder pads + screws), preamp/ADC (PCB
                + shield + op-amps + ADC + caps + headers + jacks + screws),
                processing unit (PCB + SoC + RAM + eMMC + heatsink + fan +
                40-pin GPIO + USB-A x2 + USB-C + HDMI + Ethernet + SD +
                LEDs + standoffs), display (bezel + LCD + touch + FPC +
                driver IC + backlight), power system (4x 18650 + BMS +
                USB-C PD + switch + LEDs + gauge), pistol grip (trigger +
                spring + dial + buttons + tripod + strap + pommel),
                enclosure (rails + panels + vents + gasket + GPS + IMU +
                temp sensor), and signal/power cables with connectors. UI
                overlays: scrollable spec sheet, signal flow diagram, cost
                tiers, dimensions, mode status indicators.

11-Stage DSP Pipeline:
  Wind HPF -> Bandpass -> Spectral Gate -> De-Reverb -> Decoherence ->
  Wiener -> Compressor -> Presence -> Formant -> Super-Resolution ->
  Loudness Normalization

Advanced Metrics:
  A-weighted dB, spectral centroid, spectral flatness, speech
  intelligibility index, voice activity detection, wind noise detection,
  clipping detection, composite confidence score, pitch detection.

Features:
  Auto environment adaptation, A/B compare, 4 environment presets,
  peak hold, frequency cursor with harmonic series readout (F0-5F0),
  SNR/dB/confidence history graphs, real-time FPS counter, processing
  latency measurement, recording with JSON metadata sidecar (includes
  all metrics + latency + SNR improvement).
  Deep Listen mode (K): max processing with boosted params for
  pulling barely-audible speech from heavy noise.
  VAD-gated recording (J): auto-records on voice detection, stops
  after 2s silence, creates per-segment WAV + JSON files.
  Noise profile learning (N): captures ambient noise spectrum for
  2s, uses learned profile for more accurate spectral subtraction.
  Whisper mode (U): 3x gain + boosted NR/enhance for very quiet
  or distant sources.
  Frequency isolator (Y): bandpass to 300-3000 Hz to isolate
  specific frequency content.
  Spectrum averaging (Z): exponential moving average of spectrum
  at 0.05 alpha for stable frequency analysis display.
  Spectrum snapshot (F2): freeze current spectrum as overlay for
  comparison with live data.
  Golden markers (F3): mark important moments during recording,
  timestamps saved in JSON metadata.
  Session timer: elapsed time in topbar.
  SNR improvement measurement: real-time processing gain in dB.
  Event log: rolling status feed in right panel.
  Automatic Gain Control (A): fast adaptive gain for varying sources.
  3-band EQ: bass/mid/treble shelving filters (-12 to +12 dB).
  Audio replay (O): replay last 6 seconds from ring buffer.
  Activity timeline: 600-entry voice/sound detection bar graph.
  Dynamic range: real-time peak vs average dB measurement.
  Spectrum CSV export (F4): export raw/clean spectrum to CSV.
  Clipboard metrics (F5): copy all metrics to system clipboard.
  Spectrogram speed (PGUP/PGDN): control waterfall scroll speed.
  Preset save/load (Shift+1-4 / 1-4): save custom presets to JSON.
  THD+N: real-time total harmonic distortion + noise measurement.
  Source classification: speech/music/tone/noise/silence detection.
  RT60: approximate reverberation time estimation from spectral decay.
  Audio quality grade: composite A-F score from SNR, SII, THD+N, confidence.
  EQ reset (0): instantly reset all EQ bands to 0 dB.
  Recording timeline: visual progress bar with golden markers in right panel.
  Spectral peak tracking: top 3 real-time peaks with frequency/dB display.
  Level distribution histogram: long-term dB level distribution over session.
  Session statistics: min/max/avg SNR, dB, confidence over entire session.
  VU meter with peak hold: top bar level indicator with color zones.
  Spectrum averaging (Z): exponential moving average for stable display.
  Clickable left panel: preset selector buttons, auto-adapt toggle,
  10 draggable sliders (noise reduction, enhance, voice band low/high,
  wind filter strength, de-reverb strength, compressor threshold,
  EQ bass/mid/treble).
  Topbar indicators: clipping warning, voice detected, wind noise,
  auto-adapt reason.
  Comprehensive hardware BOM with recommended instruments and
  3-tier cost estimates (budget $350, mid-range $800, premium $1800).

Dependencies:  numpy, pygame, scipy, pyaudio, soundfile
Run:           python Hearnocular.py

Controls are printed at startup and shown on-screen (press H for help,
press I for the full informational specification panel).
================================================================================
"""

# =============================================================================
# AUTO-INSTALL DEPENDENCIES
# =============================================================================
import os as _os
_os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
del _os

import subprocess as _sp
import sys as _sys

def _auto_install():
    _required = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pyaudio': 'pyaudio',
        'soundfile': 'soundfile',
        'pygame': 'pygame',
    }
    _missing = []
    for mod, pkg in _required.items():
        try:
            __import__(mod)
        except Exception:
            _missing.append(pkg)
    if _missing:
        print(f"Installing missing packages: {', '.join(_missing)}")
        _sp.check_call(
            [_sys.executable, '-m', 'pip', 'install', '--quiet'] + _missing)
        print("Installation complete.")
    else:
        print("All dependencies already installed.")

_auto_install()
del _auto_install, _sp, _sys

# =============================================================================
# IMPORTS
# =============================================================================
import math
import warnings

warnings.filterwarnings("ignore")
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from scipy import signal as scipy_signal
import pyaudio
import soundfile as sf
import queue
import threading
import time
import json
from collections import deque


# =============================================================================
# SECTION 1 -- CONFIGURATION
# =============================================================================

SAMPLE_RATE    = 44100
CHUNK_SIZE     = 2048
CHANNELS       = 1
BUFFER_SECONDS = 6
FFT_SIZE       = 2048

# Processing Parameters
NOISE_REDUCTION  = 0.82
VOICE_LOW        = 250
VOICE_HIGH       = 4800
ENHANCE_STRENGTH = 1.15

# Spectral Gate Parameters
GATE_THRESHOLD   = 0.5     # fraction of noise floor for gating
GATE_SOFTNESS    = 3.0     # tanh smoothness for soft gate transition

# Multi-band Compressor Parameters
COMP_XOVER_LOW   = 500.0   # Hz crossover between low and mid
COMP_XOVER_HIGH  = 4000.0  # Hz crossover between mid and high
COMP_THRESHOLD   = [-30.0, -24.0, -20.0]  # dB thresholds per band
COMP_RATIO       = [3.0, 4.0, 3.0]        # compression ratios per band
COMP_MAKEUP      = [6.0, 8.0, 6.0]        # dB makeup gain per band

# Presence Boost (consonant clarity 2-5 kHz)
PRESENCE_CENTER  = 3500.0  # Hz
PRESENCE_BOOST_DB = 5.0   # dB
PRESENCE_Q       = 1.5

# De-reverberation
DEREVERB_STRENGTH    = 0.3
DEREVERB_SMOOTHING   = 0.85  # spectral smoothing factor

# VAD Parameters
VAD_ENERGY_THRESH     = 0.01
VAD_FLATNESS_THRESH   = 0.45
VAD_ZCR_THRESH        = 0.15

# Clipping
CLIP_THRESHOLD   = 0.98

# Peak Hold
PEAK_HOLD_FRAMES = 90     # frames before decay starts
PEAK_DECAY_RATE  = 0.96

# Formant Enhancement (vowel clarity)
FORMANT_F1_CENTER  = 500.0   # Hz - F1 formant
FORMANT_F1_BOOST   = 3.0     # dB
FORMANT_F2_CENTER  = 1500.0  # Hz - F2 formant
FORMANT_F2_BOOST   = 4.0     # dB
FORMANT_Q          = 2.0

# Spectral Super-Resolution
SUPER_RES_STRENGTH = 0.4   # harmonic reconstruction strength
SUPER_RES_HARMONICS = 3    # number of harmonics to reconstruct

# Adaptive Wind Filter
WIND_HPF_FREQ      = 300.0  # Hz high-pass when wind detected
WIND_HPF_ORDER     = 4

# Loudness Normalization
LOUDNESS_TARGET    = -16.0  # dB target output level
LOUDNESS_ATTACK    = 0.02   # fast upward
LOUDNESS_RELEASE   = 0.005  # slow downward

# Auto-Adapt
AUTO_ADAPT_INTERVAL = 300   # frames between auto-adapt checks

# Presets
PRESETS = {
    'indoor':  {'noise_reduction': 0.60, 'enhance_strength': 1.00,
                'voice_low': 200, 'voice_high': 5000,
                'presence': 3.0, 'dereverb': 0.20, 'compress': True,
                'wind_filter': 0.3, 'comp_thresh': -20.0},
    'outdoor': {'noise_reduction': 0.82, 'enhance_strength': 1.15,
                'voice_low': 250, 'voice_high': 4800,
                'presence': 5.0, 'dereverb': 0.00, 'compress': True,
                'wind_filter': 0.7, 'comp_thresh': -24.0},
    'distant': {'noise_reduction': 0.90, 'enhance_strength': 1.50,
                'voice_low': 300, 'voice_high': 4000,
                'presence': 7.0, 'dereverb': 0.30, 'compress': True,
                'wind_filter': 0.5, 'comp_thresh': -30.0},
    'noisy':   {'noise_reduction': 0.95, 'enhance_strength': 1.30,
                'voice_low': 300, 'voice_high': 4000,
                'presence': 6.0, 'dereverb': 0.10, 'compress': True,
                'wind_filter': 0.8, 'comp_thresh': -28.0},
}
PRESET_NAMES = ['indoor', 'outdoor', 'distant', 'noisy']

# Direction Parameters
NUM_DIRECTIONS   = 36       # 360 / 10-degree bins
SWEEP_SPEED     = 0.8      # radians per second (visual sweep animation)

# Distance estimation parameters
VOICE_DB_AT_1M   = 65.0     # typical conversational voice level at 1 m

# Modes
MODE_SPECTRUM  = "spectrum"
MODE_DIRECTION = "direction"
MODE_FILTER    = "filter"
MODE_METER     = "meter"
MODE_BLUEPRINT = "blueprint"
MODES = [MODE_SPECTRUM, MODE_DIRECTION, MODE_FILTER, MODE_METER, MODE_BLUEPRINT]
MODE_NAMES = {
    MODE_SPECTRUM:  "SPECTRUM",
    MODE_DIRECTION: "DIRECTION",
    MODE_FILTER:    "FILTER",
    MODE_METER:     "METER",
    MODE_BLUEPRINT: "BLUEPRINT",
}


# =============================================================================
# SECTION 2 -- COLOR PALETTE  (matches GmansRun / SE style)
# =============================================================================

BG_TOP      = (14, 18, 26)
BG_BOT      = (4, 6, 10)
C_PANEL     = (18, 24, 34)
C_PANEL_HI  = (30, 40, 56)
C_BORDER    = (40, 40, 80)
C_TEXT      = (224, 230, 238)
C_TEXT_DIM  = (150, 160, 175)
C_ACCENT    = (90, 200, 255)
C_GOOD      = (90, 220, 130)
C_WARN      = (255, 200, 60)
C_BAD       = (255, 90, 90)
C_HOT       = (255, 110, 30)
C_COOL      = (60, 140, 200)
C_VOICE     = (100, 220, 180)     # voice-band highlight
C_NOISE     = (120, 80, 160)      # noise colour
C_PROC      = (180, 130, 255)     # processing colour
C_DIRECTION = (255, 180, 60)      # directional colour
C_RAW       = (100, 110, 130)     # raw waveform colour
C_CLEAN     = (90, 220, 130)      # cleaned waveform colour
C_SPECTRUM  = [
    (10, 10, 40),    # low energy
    (30, 40, 120),
    (60, 100, 200),
    (90, 200, 255),
    (180, 240, 100),
    (255, 220, 60),
    (255, 140, 40),
    (255, 60, 40),   # high energy
]


# =============================================================================
# SECTION 3 -- UI HELPERS
# =============================================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _mix(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def vgradient(surf, top, bot):
    h = surf.get_height()
    w = surf.get_width()
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = np.linspace(top[c], bot[c], h).astype(np.uint8)[:, None]
    pygame.surfarray.blit_array(surf, np.transpose(arr, (1, 0, 2)))


def panel(surf, x, y, w, h, alpha=210):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((C_PANEL[0], C_PANEL[1], C_PANEL[2], alpha))
    surf.blit(s, (x, y))
    pygame.draw.rect(surf, C_PANEL_HI, (x, y, w, h), 1, border_radius=6)


def bar(surf, font, x, y, w, h, frac, color, label, valtext):
    pygame.draw.rect(surf, C_PANEL_HI, (x, y, w, h), border_radius=4)
    frac = clamp(frac)
    pygame.draw.rect(surf, color, (x, y, int(w * frac), h), border_radius=4)
    img = font.render(label, True, C_TEXT_DIM)
    surf.blit(img, (x, y - 16))
    img2 = font.render(valtext, True, color)
    surf.blit(img2, (x + w - img2.get_width(), y - 16))


def wrap_text(font, text, maxpx):
    out, cur = [], ""
    for word in text.split(" "):
        trial = word if not cur else cur + " " + word
        if font.size(trial)[0] <= maxpx:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def spectrum_color(val):
    """Map 0..1 to a spectrum colour gradient."""
    idx = clamp(val) * (len(C_SPECTRUM) - 1)
    i0 = int(idx)
    i1 = min(i0 + 1, len(C_SPECTRUM) - 1)
    t = idx - i0
    return _mix(C_SPECTRUM[i0], C_SPECTRUM[i1], t)


# =============================================================================
# SECTION 3.5 -- 3D ENGINE FOR BLUEPRINT MODE
# =============================================================================

_BP_LIGHT = np.array([-0.344, 0.541, -0.767])
_BP_LIGHT = _BP_LIGHT / np.linalg.norm(_BP_LIGHT)

def _bp_rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def _bp_rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

class BPMesh:
    """3D mesh for blueprint rendering."""
    def __init__(self, verts, faces, color, name="", alpha=255):
        self.verts = np.asarray(verts, dtype=float)
        self.faces = faces
        self.color = color
        self.name = name
        self.alpha = alpha
        f3 = [f for f in faces if len(f) == 3]
        f4 = [f for f in faces if len(f) == 4]
        self.idx3 = np.array(f3, dtype=np.intp) if f3 else np.zeros((0, 3), dtype=np.intp)
        self.idx4 = np.array(f4, dtype=np.intp) if f4 else np.zeros((0, 4), dtype=np.intp)

class BPPart:
    """A named group of meshes."""
    def __init__(self, key, name, meshes, specs=None):
        self.key = key
        self.name = name
        self.meshes = meshes
        self.specs = specs or []

def _bp_box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2),
         (2, 6, 7, 3), (3, 7, 4, 0)]
    return v, f

def _bp_cyl(r, z0, z1, seg=24):
    seg = max(6, int(seg))
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    c0, c1 = 2 * seg, 2 * seg + 1
    verts.append((0, 0, z0))
    verts.append((0, 0, z1))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
        faces.append((c0, b, a))
        faces.append((c1, seg + a, seg + b))
    return verts, faces

def _bp_sph(r, seg_u=16, seg_v=10):
    seg_u = max(6, int(seg_u))
    seg_v = max(4, int(seg_v))
    verts, faces = [], []
    for j in range(seg_v + 1):
        va = math.pi * j / seg_v - math.pi / 2
        for i in range(seg_u):
            ha = 2 * math.pi * i / seg_u
            verts.append((r * math.cos(va) * math.cos(ha),
                          r * math.cos(va) * math.sin(ha),
                          r * math.sin(va)))
    for j in range(seg_v):
        for i in range(seg_u):
            a = j * seg_u + i
            b = j * seg_u + (i + 1) % seg_u
            c = (j + 1) * seg_u + (i + 1) % seg_u
            d = (j + 1) * seg_u + i
            faces.append((a, b, c, d))
    return verts, faces

def _bp_ring(ro, ri, z, seg=48):
    seg = max(6, int(seg))
    verts = []
    for a in np.linspace(0, 2 * math.pi, seg, endpoint=False):
        verts.append((ro * math.cos(a), ro * math.sin(a), z))
        verts.append((ri * math.cos(a), ri * math.sin(a), z))
    faces = []
    for i in range(seg):
        a, b = 2 * i, 2 * (i + 1) % (2 * seg)
        faces.append((a, b, b + 1, a + 1))
    return verts, faces

def _bp_ann(r_out, r_in, z0, z1, seg=24):
    seg = max(6, int(seg))
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r_out * math.cos(a), r_out * math.sin(a), z))
        for a in ang:
            verts.append((r_in * math.cos(a), r_in * math.sin(a), z))
    n = seg
    def oo(layer, i):
        return layer * 2 * n + i
    def ii(layer, i):
        return layer * 2 * n + n + i
    for i in range(n):
        j = (i + 1) % n
        faces.append((oo(0, i), ii(0, i), ii(0, j), oo(0, j)))
        faces.append((oo(1, i), oo(1, j), ii(1, j), ii(1, i)))
        faces.append((oo(0, i), oo(0, j), oo(1, j), oo(1, i)))
        faces.append((ii(0, i), ii(1, i), ii(1, j), ii(0, j)))
    return verts, faces

def _bp_cone(r, z0, z1, seg=20):
    seg = max(6, int(seg))
    verts = []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for a in ang:
        verts.append((r * math.cos(a), r * math.sin(a), z0))
    verts.append((0, 0, z1))
    faces = []
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg))
    return verts, faces


def _bp_cyl_x(r, x0, x1, seg=16):
    """Cylinder oriented along X axis."""
    v, f = _bp_cyl(r, x0, x1, seg)
    # Swap x and z: (x,y,z) -> (z,y,x)
    v = [(p[2], p[1], p[0]) for p in v]
    return v, f


def _bp_cyl_y(r, y0, y1, seg=16):
    """Cylinder oriented along Y axis."""
    v, f = _bp_cyl(r, y0, y1, seg)
    # Swap y and z: (x,y,z) -> (x,z,y)
    v = [(p[0], p[2], p[1]) for p in v]
    return v, f


def _bp_torus(R, r, seg_u=24, seg_v=12):
    """Torus with major radius R and minor radius r."""
    seg_u = max(8, int(seg_u))
    seg_v = max(6, int(seg_v))
    verts, faces = [], []
    for j in range(seg_v):
        va = 2 * math.pi * j / seg_v
        for i in range(seg_u):
            ha = 2 * math.pi * i / seg_u
            cx = (R + r * math.cos(va)) * math.cos(ha)
            cy = (R + r * math.cos(va)) * math.sin(ha)
            cz = r * math.sin(va)
            verts.append((cx, cy, cz))
    for j in range(seg_v):
        for i in range(seg_u):
            a = j * seg_u + i
            b = j * seg_u + (i + 1) % seg_u
            c = ((j + 1) % seg_v) * seg_u + (i + 1) % seg_u
            d = ((j + 1) % seg_v) * seg_u + i
            faces.append((a, b, c, d))
    return verts, faces


def _bp_prism(cx, cy, cz, radius, height, n_sides=6):
    """N-sided prism (hexagonal=6, octagonal=8)."""
    n_sides = max(3, int(n_sides))
    verts, faces = [], []
    hz = height / 2
    ang = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
    for z in (cz - hz, cz + hz):
        for a in ang:
            verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z))
    for i in range(n_sides):
        a, b = i, (i + 1) % n_sides
        faces.append((a, b, n_sides + b, n_sides + a))
    # Caps
    faces.append(tuple(range(n_sides)))
    faces.append(tuple(range(n_sides, 2 * n_sides)))
    return verts, faces


def _bp_screw(cx, cy, cz, r=0.02, length=0.08):
    """Screw: small cylinder head + shaft."""
    verts, faces = [], []
    # Head (flat cylinder)
    v1, f1 = _bp_cyl(r, cz + length - 0.015, cz + length, 8)
    # Shaft (thinner cylinder)
    v2, f2 = _bp_cyl(r * 0.5, cz, cz + length - 0.01, 6)
    # Offset head verts
    base = len(v2)
    v = v2 + v1
    f = f2 + [(fi[0] + base, fi[1] + base, fi[2] + base) if len(fi) == 3
              else (fi[0] + base, fi[1] + base, fi[2] + base, fi[3] + base) for fi in f1]
    v = [(p[0] + cx, p[1] + cy, p[2]) for p in v]
    return v, f


def _bp_grid(cx, cy, cz, sx, sy, nx, ny, z_thickness=0.005):
    """Grid/PCB trace pattern: nx x ny small boxes on a plane."""
    verts, faces = [], []
    dx = sx / nx
    dy = sy / ny
    for ix in range(nx):
        for iy in range(ny):
            if (ix + iy) % 2 == 0:
                px = cx - sx / 2 + dx * (ix + 0.5)
                py = cy - sy / 2 + dy * (iy + 0.5)
                v, f = _bp_box(px, py, cz, dx * 0.8, dy * 0.8, z_thickness)
                verts.extend(v)
                faces.extend([(fi[0] + len(verts) - len(v), fi[1] + len(verts) - len(v),
                               fi[2] + len(verts) - len(v)) if len(fi) == 3 else
                              (fi[0] + len(verts) - len(v), fi[1] + len(verts) - len(v),
                               fi[2] + len(verts) - len(v), fi[3] + len(verts) - len(v))
                              for fi in f])
    return verts, faces


def _bp_translate(verts, dx, dy, dz):
    """Translate all vertices by (dx, dy, dz)."""
    return [(p[0] + dx, p[1] + dy, p[2] + dz) for p in verts]


def _bp_offset_faces(faces, offset):
    """Offset all face indices by a constant."""
    if len(faces) == 0:
        return faces
    return [(tuple(idx + offset for idx in f)) for f in faces]


class BlueprintRenderer:
    """Advanced 3D software renderer for the hardware blueprint view."""
    def __init__(self, parts, az=0.5, el=0.3, dist=8.0):
        self.parts = parts
        self.az = az
        self.el = el
        self.dist = dist
        self.td = dist
        self.px = 0.0
        self.py = 0.0
        self.zf = 1.0
        self._home_dist = dist
        self.auto_rotate = False
        self.wireframe = False
        self.exploded = 0.0  # 0.0 = assembled, 1.0 = fully exploded
        self.highlight_part = -1  # index into parts, -1 = none
        self.show_edges = True
        self._explode_dirs = {}
        self._compute_explode_dirs()
        self.auto_frame()

    def _compute_explode_dirs(self):
        """Compute explosion direction for each part (away from center)."""
        all_pts = []
        for part in self.parts:
            for mesh in part.meshes:
                if len(mesh.verts):
                    all_pts.append(mesh.verts)
        if not all_pts:
            return
        center = np.concatenate(all_pts, axis=0).mean(axis=0)
        for pi, part in enumerate(self.parts):
            ppts = []
            for mesh in part.meshes:
                if len(mesh.verts):
                    ppts.append(mesh.verts)
            if ppts:
                pc = np.concatenate(ppts, axis=0).mean(axis=0)
                direction = pc - center
                dl = np.linalg.norm(direction)
                if dl > 1e-6:
                    direction = direction / dl
                else:
                    direction = np.array([0, 0, 1.0])
            else:
                direction = np.array([0, 0, 1.0])
            self._explode_dirs[pi] = direction * 2.5  # explosion magnitude

    def auto_frame(self):
        all_pts = []
        for part in self.parts:
            for mesh in part.meshes:
                if len(mesh.verts):
                    all_pts.append(mesh.verts)
        if not all_pts:
            return
        pts = np.concatenate(all_pts, axis=0)
        c = pts.mean(axis=0)
        r = float(np.sqrt(((pts - c) ** 2).sum(axis=1)).max())
        self.dist = self.td = self._home_dist = max(r * 2.5, 1.0)

    def reset(self):
        self.az = 0.5
        self.el = 0.3
        self.px = 0.0
        self.py = 0.0
        self.zf = 1.0
        self.auto_rotate = False
        self.wireframe = False
        self.exploded = 0.0
        self.highlight_part = -1
        self.auto_frame()

    def orbit(self, dx, dy):
        self.az += dx * 0.005
        self.el = clamp(self.el + dy * 0.005, -0.1, 1.4)

    def pan(self, dx, dy):
        self.px += dx * 0.003
        self.py -= dy * 0.003

    def zoom(self, factor):
        self.td = clamp(self.td * factor, 0.5, 100.0)
        self.dist = self.td

    def tick(self, dt):
        if self.auto_rotate:
            self.az += dt * 0.3

    def render(self, surf, rect, show_labels=True, font=None, font_small=None):
        cx = rect.centerx + self.px * rect.w
        cy = rect.centery + self.py * rect.h
        sc = rect.h / (self.dist * 3.0) * self.zf
        RT = (_bp_rot_x(self.el) @ _bp_rot_y(self.az)).T
        focal = self.dist * 3.0  # perspective focal distance

        all_faces = []
        all_edges = []
        for pi, part in enumerate(self.parts):
            # Exploded view offset
            explode_offset = np.zeros(3)
            if self.exploded > 0 and pi in self._explode_dirs:
                explode_offset = self._explode_dirs[pi] * self.exploded

            is_highlighted = (self.highlight_part == pi)
            is_dimmed = (self.highlight_part >= 0 and self.highlight_part != pi)

            for mesh in part.meshes:
                if len(mesh.verts) == 0:
                    continue
                rv = (mesh.verts + explode_offset) @ RT

                # Perspective projection
                z_vals = rv[:, 2] + focal
                z_safe = np.where(np.abs(z_vals) < 0.1, 0.1, z_vals)
                persp = focal / z_safe
                proj = np.column_stack((cx + rv[:, 0] * sc * persp,
                                        cy - rv[:, 1] * sc * persp))

                # Determine alpha modulation for highlight/dim
                mesh_alpha = mesh.alpha
                mesh_color = mesh.color
                if is_dimmed:
                    mesh_alpha = max(30, mesh_alpha // 4)
                if is_highlighted:
                    # Brighten highlighted part
                    mesh_color = tuple(min(255, int(c * 1.3)) for c in mesh.color)

                # Collect edges for wireframe/edge mode
                if self.wireframe or self.show_edges:
                    if len(mesh.idx3):
                        for fi in range(len(mesh.idx3)):
                            idx_f = mesh.idx3[fi]
                            for ei in range(3):
                                a, b = idx_f[ei], idx_f[(ei + 1) % 3]
                                if a < b:
                                    all_edges.append((float(rv[idx_f].mean(axis=0)[2]),
                                                      proj[a], proj[b],
                                                      mesh_alpha if not self.wireframe else 200))
                    if len(mesh.idx4):
                        for fi in range(len(mesh.idx4)):
                            idx_f = mesh.idx4[fi]
                            for ei in range(4):
                                a, b = idx_f[ei], idx_f[(ei + 1) % 4]
                                if a < b:
                                    all_edges.append((float(rv[idx_f].mean(axis=0)[2]),
                                                      proj[a], proj[b],
                                                      mesh_alpha if not self.wireframe else 200))

                if self.wireframe:
                    continue  # skip filled faces in wireframe mode

                if len(mesh.idx3):
                    idx = mesh.idx3
                    p0 = rv[idx[:, 0]]
                    p1 = rv[idx[:, 1]]
                    p2 = rv[idx[:, 2]]
                    nrm = np.cross(p1 - p0, p2 - p0)
                    nl = np.linalg.norm(nrm, axis=1)
                    safe = nl > 1e-12
                    flip = nrm[:, 2] > 0
                    nrm = np.where(flip[:, None], -nrm, nrm)
                    dd = (nrm @ _BP_LIGHT) / np.where(safe, nl, 1.0)
                    sh = np.where(safe, 0.30 + 0.70 * np.maximum(dd, 0.0), 0.6)
                    if is_highlighted:
                        sh = sh * 1.15
                    col = np.minimum(255, (np.asarray(mesh_color) * sh[:, None]).astype(np.int64))
                    for fi in range(len(idx)):
                        pts = proj[idx[fi]].tolist()
                        depth = float(rv[idx[fi]].mean(axis=0)[2])
                        all_faces.append((depth, pts, col[fi].tolist(), mesh_alpha, mesh.name))

                if len(mesh.idx4):
                    idx = mesh.idx4
                    p0 = rv[idx[:, 0]]
                    p1 = rv[idx[:, 1]]
                    p2 = rv[idx[:, 2]]
                    nrm = np.cross(p1 - p0, p2 - p0)
                    nl = np.linalg.norm(nrm, axis=1)
                    safe = nl > 1e-12
                    flip = nrm[:, 2] > 0
                    nrm = np.where(flip[:, None], -nrm, nrm)
                    dd = (nrm @ _BP_LIGHT) / np.where(safe, nl, 1.0)
                    sh = np.where(safe, 0.30 + 0.70 * np.maximum(dd, 0.0), 0.6)
                    if is_highlighted:
                        sh = sh * 1.15
                    col = np.minimum(255, (np.asarray(mesh_color) * sh[:, None]).astype(np.int64))
                    for fi in range(len(idx)):
                        pts = proj[idx[fi]].tolist()
                        depth = float(rv[idx[fi]].mean(axis=0)[2])
                        all_faces.append((depth, pts, col[fi].tolist(), mesh_alpha, mesh.name))

        all_faces.sort(key=lambda x: -x[0])

        for depth, pts, col, alpha, name in all_faces:
            n = len(pts)
            if n < 3:
                continue
            if alpha < 255:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                mnx, mxx = min(xs), max(xs)
                mny, mxy = min(ys), max(ys)
                mx, my = int(mnx), int(mny)
                w, h = int(mxx) + 1 - mx, int(mxy) + 1 - my
                if w <= 0 or h <= 0 or w > rect.w * 2 or h > rect.h * 2:
                    continue
                if int(mxx) < 0 or int(mxy) < 0 or mx >= surf.get_width() or my >= surf.get_height():
                    continue
                sf = pygame.Surface((w, h), pygame.SRCALPHA)
                pygame.draw.polygon(sf, (col[0], col[1], col[2], alpha),
                                    [(p[0] - mx, p[1] - my) for p in pts])
                surf.blit(sf, (mx, my))
            else:
                pygame.draw.polygon(surf, col, pts)

        # Draw edges
        if self.show_edges or self.wireframe:
            all_edges.sort(key=lambda x: -x[0])
            edge_col = (60, 70, 85) if not self.wireframe else (100, 140, 180)
            for depth, p1, p2, alpha in all_edges:
                if rect.collidepoint(int(p1[0]), int(p1[1])) or rect.collidepoint(int(p2[0]), int(p2[1])):
                    if self.wireframe:
                        pygame.draw.aaline(surf, edge_col, p1, p2)
                    else:
                        pygame.draw.line(surf, edge_col,
                                         (int(p1[0]), int(p1[1])),
                                         (int(p2[0]), int(p2[1])), 1)

        if show_labels and font and font_small:
            self._draw_labels(surf, rect, font, font_small, RT, sc, cx, cy, focal)

    def _draw_labels(self, surf, rect, font, font_small, RT, sc, cx, cy, focal):
        seen_pos = []
        for pi, part in enumerate(self.parts):
            explode_offset = np.zeros(3)
            if self.exploded > 0 and pi in self._explode_dirs:
                explode_offset = self._explode_dirs[pi] * self.exploded
            is_dimmed = (self.highlight_part >= 0 and self.highlight_part != pi)
            if is_dimmed:
                continue
            for mesh in part.meshes:
                if not mesh.name or len(mesh.verts) == 0:
                    continue
                c = (mesh.verts + explode_offset).mean(axis=0)
                rc = c @ RT
                z_val = rc[2] + focal
                z_safe = max(0.1, abs(z_val))
                persp = focal / z_safe
                x = cx + rc[0] * sc * persp
                y = cy - rc[1] * sc * persp
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                px, py = int(x), int(y)
                if not rect.collidepoint(px, py):
                    continue
                skip = False
                for sx, sy in seen_pos:
                    if abs(px - sx) < 22 and abs(py - sy) < 14:
                        skip = True
                        break
                if skip:
                    continue
                seen_pos.append((px, py))
                lbl = font_small.render(mesh.name, True, C_TEXT)
                bg = pygame.Surface((lbl.get_width() + 8, lbl.get_height() + 4), pygame.SRCALPHA)
                bg.fill((10, 14, 22, 200))
                # Leader line from label to point
                lbl_x = px + 8
                lbl_y = py - 10
                pygame.draw.line(surf, (80, 100, 130), (px, py), (lbl_x, lbl_y + 5), 1)
                surf.blit(bg, (lbl_x, lbl_y))
                surf.blit(lbl, (lbl_x + 4, lbl_y + 2))


def build_hearnocular_model():
    """Build highly detailed 3D mesh model of the Hearnocular hardware device.
    10 major parts, 100+ mesh groups with screws, gaskets, PCB traces, connectors."""
    parts = []
    dish_r = 2.8
    dish_depth = 0.5

    # ===================== PARABOLIC DISH =====================
    dish_meshes = []
    seg_u, seg_v = 48, 14
    front_verts, back_verts = [], []
    for j in range(seg_v + 1):
        frac_j = j / seg_v
        r = dish_r * frac_j
        z = dish_depth * frac_j * frac_j
        for i in range(seg_u):
            a = 2 * math.pi * i / seg_u
            front_verts.append((r * math.cos(a), r * math.sin(a), z))
            back_verts.append((r * math.cos(a), r * math.sin(a), z - 0.10))
    all_dish_verts = front_verts + back_verts
    dish_faces = []
    for j in range(seg_v):
        for i in range(seg_u):
            a = j * seg_u + i
            b = j * seg_u + (i + 1) % seg_u
            c = (j + 1) * seg_u + (i + 1) % seg_u
            d = (j + 1) * seg_u + i
            dish_faces.append((a, b, c, d))
            a2, b2 = a + len(front_verts), b + len(front_verts)
            c2, d2 = c + len(front_verts), d + len(front_verts)
            dish_faces.append((a2, d2, c2, b2))
    dish_meshes.append(BPMesh(all_dish_verts, dish_faces, C_COOL, "Parabolic Dish", alpha=215))

    # Reinforced rim torus
    v, f = _bp_torus(dish_r, 0.04, 32, 10)
    verts = [(p[0], p[1], p[2] + dish_depth) for p in v]
    dish_meshes.append(BPMesh(verts, f, C_PANEL_HI, "Rim"))

    # 6 panel ridges with hinges
    for seg_i in range(6):
        a = 2 * math.pi * seg_i / 6
        ca, sa = math.cos(a), math.sin(a)
        v, f = _bp_box(dish_r * 0.5 * ca, dish_r * 0.5 * sa,
                       dish_depth * 0.25, 0.03, dish_r * 0.95, 0.03)
        dish_meshes.append(BPMesh(v, f, C_PANEL_HI, "", alpha=120))
        # Hinge at outer edge
        v, f = _bp_cyl(0.04, dish_r * ca - 0.05 * ca, dish_r * sa - 0.05 * sa, 6)
        verts = [(p[0], p[1], p[2] + dish_depth - 0.02) for p in v]
        dish_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "Hinge"))

    # Center hub + hex cap
    v, f = _bp_cyl(0.14, dish_depth - 0.06, dish_depth + 0.06, 16)
    dish_meshes.append(BPMesh(v, f, C_PANEL_HI, "Dish Hub"))
    v, f = _bp_prism(0, 0, dish_depth + 0.08, 0.06, 0.04, 6)
    dish_meshes.append(BPMesh(v, f, C_ACCENT, "Hub Cap"))

    # 4 spider vanes with screws
    for si in range(4):
        a = math.pi / 4 + si * math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        v, f = _bp_box(0.6 * ca, 0.6 * sa, dish_depth + 0.15, 1.2, 0.025, 0.025)
        dish_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Spider Vane"))
        v, f = _bp_screw(0.04 * ca, 0.04 * sa, dish_depth + 0.02, 0.015, 0.06)
        dish_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))

    # 4 backing ribs
    for ring_r in [dish_r * 0.3, dish_r * 0.5, dish_r * 0.7, dish_r * 0.9]:
        v, f = _bp_ring(ring_r, ring_r - 0.025, -0.08, 28)
        dish_meshes.append(BPMesh(v, f, C_PANEL_HI, "", alpha=100))

    # 6 magnetic locks
    for si in range(6):
        a = 2 * math.pi * si / 6
        v, f = _bp_cyl(0.03, dish_depth - 0.05, dish_depth + 0.03, 8)
        verts = [(p[0] + dish_r * 0.5 * math.cos(a),
                  p[1] + dish_r * 0.5 * math.sin(a), p[2]) for p in v]
        dish_meshes.append(BPMesh(verts, f, C_WARN, "Mag Lock"))

    # Focus mark ring
    v, f = _bp_ring(0.20, 0.16, dish_depth + 0.005, 24)
    dish_meshes.append(BPMesh(v, f, C_ACCENT, "Focus Mark", alpha=150))

    parts.append(BPPart("dish", "PARABOLIC DISH", dish_meshes, [
        "22-24 inch carbon fiber composite",
        "6-panel deployable, magnetic locks",
        "Surface accuracy: <0.5mm RMS",
        "4 spider vanes + center hub + hinges",
        "Reinforced rim + backing ribs",
        "Weight target: <3 lbs",
    ]))

    # ===================== FOCAL MICROPHONE =====================
    mic_meshes = []
    # Suspension cage torus
    v, f = _bp_torus(0.16, 0.015, 20, 8)
    verts = [(p[0], p[1], p[2] + dish_depth + 0.30) for p in v]
    mic_meshes.append(BPMesh(verts, f, C_PANEL_HI, "Suspension Cage"))
    # Shock mount cross supports (4)
    for si in range(4):
        a = si * math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        v, f = _bp_box(0.16 * ca, 0.16 * sa, dish_depth + 0.30, 0.32, 0.012, 0.012)
        mic_meshes.append(BPMesh(v, f, C_TEXT_DIM, ""))
    # Mic capsule sphere (high-res)
    v, f = _bp_sph(0.08, 16, 12)
    mic_meshes.append(BPMesh(v, f, C_RAW, "Focal Mic"))
    # Mic grille cap
    v, f = _bp_sph(0.075, 10, 6)
    verts = [(p[0], p[1], p[2] + 0.04) for p in v]
    mic_meshes.append(BPMesh(verts, f, C_PANEL_HI, "Grille", alpha=100))
    # Windscreen foam (translucent)
    v, f = _bp_sph(0.12, 14, 10)
    mic_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Windscreen", alpha=50))
    # XLR connector housing
    v, f = _bp_cyl(0.06, dish_depth + 0.35, dish_depth + 0.48, 12)
    mic_meshes.append(BPMesh(v, f, C_PANEL, "XLR Connector"))
    # XLR pins (3)
    for pi in range(3):
        pa = 2 * math.pi * pi / 3 + 0.5
        v, f = _bp_cyl(0.008, dish_depth + 0.46, dish_depth + 0.50, 5)
        verts = [(p[0] + 0.025 * math.cos(pa), p[1] + 0.025 * math.sin(pa), p[2]) for p in v]
        mic_meshes.append(BPMesh(verts, f, C_WARN, ""))
    # Cable strain relief
    v, f = _bp_cone(0.05, dish_depth + 0.48, dish_depth + 0.55, 10)
    mic_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Strain Relief"))
    # Suspension mounting screws (4)
    for si in range(4):
        a = si * math.pi / 2 + math.pi / 4
        v, f = _bp_screw(0.17 * math.cos(a), 0.17 * math.sin(a),
                         dish_depth + 0.28, 0.012, 0.04)
        mic_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    parts.append(BPPart("focal_mic", "FOCAL MICROPHONE", mic_meshes, [
        "Primo EM-272Z1 omnidirectional",
        "or DPA 4060/4061",
        "Self-noise: <14 dB-A",
        "Flat response to 20 kHz",
        "Suspension cage + grille + windscreen",
        "XLR + strain relief + screws",
    ]))

    # ===================== MEMS ARRAY =====================
    mems_meshes = []
    mems_r = 1.0
    # PCB mounting ring (annulus)
    v, f = _bp_ann(mems_r + 0.06, mems_r - 0.06, -0.04, 0.02, 36)
    mems_meshes.append(BPMesh(v, f, C_PANEL, "MEMS PCB Ring", alpha=210))
    # PCB trace grid
    v, f = _bp_grid(0, 0, 0.025, 2.0, 2.0, 8, 8, 0.003)
    mems_meshes.append(BPMesh(v, f, C_ACCENT, "PCB Traces", alpha=80))
    # 8 MEMS mic capsules with details
    for i in range(8):
        a = 2 * math.pi * i / 8 + math.pi / 8
        mx = mems_r * math.cos(a)
        my = mems_r * math.sin(a)
        # Mic body
        v, f = _bp_cyl(0.055, -0.03, 0.05, 12)
        verts = [(p[0] + mx, p[1] + my, p[2]) for p in v]
        mems_meshes.append(BPMesh(verts, f, C_ACCENT, "MEMS %d" % (i + 1), alpha=225))
        # Port hole
        v2, f2 = _bp_cyl(0.028, 0.05, 0.06, 8)
        verts2 = [(p[0] + mx, p[1] + my, p[2]) for p in v2]
        mems_meshes.append(BPMesh(verts2, f2, C_TEXT_DIM, ""))
        # Solder pads (2)
        for si in range(2):
            sa = a + math.pi / 2 + (si - 0.5) * 0.3
            v3, f3 = _bp_cyl(0.015, -0.03, -0.02, 6)
            verts3 = [(p[0] + mx + 0.06 * math.cos(sa),
                       p[1] + my + 0.06 * math.sin(sa), p[2]) for p in v3]
            mems_meshes.append(BPMesh(verts3, f3, C_WARN, ""))
    # Ribbon cable
    v, f = _bp_box(2.0, 0, -0.01, 1.5, 0.05, 0.025)
    mems_meshes.append(BPMesh(v, f, C_RAW, "Array Cable", alpha=130))
    # Mounting screws (4)
    for si in range(4):
        a = si * math.pi / 2 + math.pi / 4
        v, f = _bp_screw(mems_r * 1.1 * math.cos(a), mems_r * 1.1 * math.sin(a),
                         -0.02, 0.015, 0.05)
        mems_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    parts.append(BPPart("mems_array", "MEMS ARRAY (8x)", mems_meshes, [
        "InvenSense ICS-43434",
        "85mm diameter circle",
        "24-bit ADC per channel",
        "Half-wavelength at 2 kHz",
        "PCB ring + traces + solder pads",
        "Ribbon cable + mounting screws",
    ]))

    # ===================== PREAMP + ADC =====================
    pa_x = 3.8
    preamp_meshes = []
    # PCB base with trace grid
    v, f = _bp_box(pa_x, 0, -0.02, 0.9, 0.55, 0.04)
    preamp_meshes.append(BPMesh(v, f, C_PANEL, "Preamp PCB"))
    v, f = _bp_grid(pa_x, 0, 0.01, 0.8, 0.45, 6, 4, 0.003)
    preamp_meshes.append(BPMesh(v, f, C_ACCENT, "Traces", alpha=80))
    # Shielding can
    v, f = _bp_box(pa_x, 0, 0.08, 0.6, 0.4, 0.15)
    preamp_meshes.append(BPMesh(v, f, C_COOL, "Shield Can", alpha=210))
    # Shield can mounting tabs (2)
    for tx in (-0.25, 0.25):
        v, f = _bp_box(pa_x + tx, 0, 0.06, 0.08, 0.35, 0.02)
        preamp_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # 3 op-amp chips with pin 1 dots
    for i in range(3):
        cx = pa_x - 0.25 + i * 0.25
        v, f = _bp_box(cx, 0.08, 0.24, 0.12, 0.08, 0.04)
        preamp_meshes.append(BPMesh(v, f, C_PANEL_HI, "OPA1612"))
        v, f = _bp_sph(0.008, 5, 4)
        verts = [(p[0] + cx - 0.04, p[1] + 0.08 - 0.03, p[2] + 0.27) for p in v]
        preamp_meshes.append(BPMesh(verts, f, C_WARN, ""))
    # ADC chip
    v, f = _bp_box(pa_x + 0.25, 0, 0.24, 0.20, 0.20, 0.05)
    preamp_meshes.append(BPMesh(v, f, C_PANEL_HI, "CS5368 ADC"))
    # 4 electrolytic capacitors with stripes
    for i in range(4):
        cx = pa_x - 0.30 + i * 0.20
        v, f = _bp_cyl(0.04, 0.02, 0.13, 10)
        verts = [(p[0] + cx, p[1] - 0.15, p[2]) for p in v]
        preamp_meshes.append(BPMesh(verts, f, C_WARN, "Cap"))
        v, f = _bp_cyl(0.041, 0.10, 0.12, 10)
        verts = [(p[0] + cx, p[1] - 0.15, p[2]) for p in v]
        preamp_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "", alpha=150))
    # Pin headers (2 rows)
    for row in (-0.22, 0.22):
        v, f = _bp_box(pa_x, row, 0.05, 0.5, 0.04, 0.06)
        preamp_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Header"))
    # Input connectors with center pins
    for ci in range(2):
        cx = pa_x - 0.35 + ci * 0.25
        v, f = _bp_cyl(0.045, -0.28, -0.20, 10)
        verts = [(p[0] + cx, p[1], p[2]) for p in v]
        preamp_meshes.append(BPMesh(verts, f, C_PANEL_HI, "Input Jack"))
        v, f = _bp_cyl(0.015, -0.22, -0.16, 6)
        verts = [(p[0] + cx, p[1], p[2]) for p in v]
        preamp_meshes.append(BPMesh(verts, f, C_WARN, ""))
    # PCB mounting screws (4 corners)
    for cx, cy in [(-0.40, -0.22), (0.40, -0.22), (-0.40, 0.22), (0.40, 0.22)]:
        v, f = _bp_screw(pa_x + cx, cy, -0.04, 0.015, 0.05)
        preamp_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    parts.append(BPPart("preamp", "PREAMP + ADC", preamp_meshes, [
        "TI OPA1612 op-amp stages",
        "Gain: 20-60 dB programmable",
        "THD < 0.0009%",
        "CS5368 8-ch 24-bit/216kHz",
        "Shielded analog + pin headers",
        "PCB traces + caps + screws",
    ]))

    # ===================== PROCESSING UNIT =====================
    pu_x = 5.2
    proc_meshes = []
    # PCB base with traces
    v, f = _bp_box(pu_x, 0, -0.02, 1.1, 0.75, 0.04)
    proc_meshes.append(BPMesh(v, f, C_PANEL, "CPU PCB"))
    v, f = _bp_grid(pu_x, 0, 0.01, 1.0, 0.65, 8, 6, 0.003)
    proc_meshes.append(BPMesh(v, f, C_ACCENT, "Traces", alpha=80))
    # Main SoC chip with pin 1 dot
    v, f = _bp_box(pu_x, 0, 0.06, 0.35, 0.35, 0.06)
    proc_meshes.append(BPMesh(v, f, C_GOOD, "SoC"))
    v, f = _bp_sph(0.01, 5, 4)
    verts = [(p[0] + pu_x - 0.15, p[1] - 0.15, p[2] + 0.10) for p in v]
    proc_meshes.append(BPMesh(verts, f, C_WARN, ""))
    # RAM chip
    v, f = _bp_box(pu_x + 0.30, 0.10, 0.04, 0.18, 0.12, 0.03)
    proc_meshes.append(BPMesh(v, f, C_PANEL_HI, "RAM"))
    # eMMC storage chip
    v, f = _bp_box(pu_x - 0.30, 0.15, 0.04, 0.14, 0.10, 0.03)
    proc_meshes.append(BPMesh(v, f, C_PANEL_HI, "eMMC"))
    # Ethernet magnetics
    v, f = _bp_box(pu_x - 0.30, -0.15, 0.04, 0.12, 0.08, 0.03)
    proc_meshes.append(BPMesh(v, f, C_COOL, "Ethernet Mag"))
    # Heatsink (7 fins)
    for i in range(7):
        v, f = _bp_box(pu_x - 0.30 + i * 0.10, 0, 0.14, 0.04, 0.45, 0.22)
        proc_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Fan housing + blades + hub
    v, f = _bp_cyl(0.14, 0.36, 0.42, 16)
    verts = [(p[0] + pu_x, p[1], p[2]) for p in v]
    proc_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "Fan"))
    for bi in range(4):
        ba = bi * math.pi / 2
        v, f = _bp_box(0, 0, 0.40, 0.10, 0.22, 0.02)
        verts = [(p[0] * math.cos(ba) - p[1] * math.sin(ba) + pu_x,
                  p[0] * math.sin(ba) + p[1] * math.cos(ba), p[2]) for p in v]
        proc_meshes.append(BPMesh(verts, f, C_PANEL, "", alpha=150))
    v, f = _bp_cyl(0.03, 0.42, 0.45, 8)
    verts = [(p[0] + pu_x, p[1], p[2]) for p in v]
    proc_meshes.append(BPMesh(verts, f, C_PANEL_HI, ""))
    # GPIO header + 40 pins
    v, f = _bp_box(pu_x + 0.35, -0.25, 0.04, 0.25, 0.06, 0.08)
    proc_meshes.append(BPMesh(v, f, C_TEXT_DIM, "GPIO"))
    for row in range(2):
        for col in range(20):
            px = pu_x + 0.22 + col * 0.012
            py = -0.27 + row * 0.04
            v, f = _bp_cyl(0.004, 0.12, 0.14, 4)
            verts = [(p[0] + px, p[1] + py, p[2]) for p in v]
            proc_meshes.append(BPMesh(verts, f, C_WARN, ""))
    # USB-A ports (2x) with inner inserts
    for ui in range(2):
        uy = -0.15 + ui * 0.20
        v, f = _bp_box(pu_x + 0.48, uy, 0.04, 0.12, 0.10, 0.08)
        proc_meshes.append(BPMesh(v, f, C_COOL, "USB-A"))
        v, f = _bp_box(pu_x + 0.50, uy, 0.06, 0.08, 0.06, 0.04)
        proc_meshes.append(BPMesh(v, f, C_TEXT_DIM, ""))
    # USB-C port
    v, f = _bp_box(pu_x + 0.48, 0.28, 0.04, 0.10, 0.08, 0.06)
    proc_meshes.append(BPMesh(v, f, C_WARN, "USB-C"))
    # HDMI port
    v, f = _bp_box(pu_x - 0.48, 0.20, 0.04, 0.14, 0.10, 0.08)
    proc_meshes.append(BPMesh(v, f, C_ACCENT, "HDMI"))
    # SD card slot with card
    v, f = _bp_box(pu_x - 0.48, -0.25, 0.02, 0.18, 0.04, 0.03)
    proc_meshes.append(BPMesh(v, f, C_PANEL_HI, "SD Slot"))
    v, f = _bp_box(pu_x - 0.52, -0.25, 0.03, 0.08, 0.03, 0.02)
    proc_meshes.append(BPMesh(v, f, C_WARN, "SD Card"))
    # Ethernet jack (RJ45)
    v, f = _bp_box(pu_x - 0.48, -0.05, 0.04, 0.14, 0.12, 0.10)
    proc_meshes.append(BPMesh(v, f, C_COOL, "Ethernet"))
    # Status LED + Power LED
    v, f = _bp_sph(0.04, 8, 6)
    verts = [(p[0] + pu_x + 0.40, p[1] - 0.30, p[2] + 0.06) for p in v]
    proc_meshes.append(BPMesh(verts, f, C_GOOD, "Status LED"))
    v, f = _bp_sph(0.03, 6, 5)
    verts = [(p[0] + pu_x + 0.40, p[1] - 0.35, p[2] + 0.06) for p in v]
    proc_meshes.append(BPMesh(verts, f, C_WARN, "PWR LED"))
    # Mounting standoffs (4 corners)
    for cx, cy in [(-0.45, -0.30), (0.45, -0.30), (-0.45, 0.30), (0.45, 0.30)]:
        v, f = _bp_cyl(0.035, -0.05, 0.01, 8)
        verts = [(p[0] + pu_x + cx, p[1] + cy, p[2]) for p in v]
        proc_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "Standoff"))
    # PCB screws (4)
    for cx, cy in [(-0.48, -0.32), (0.48, -0.32), (-0.48, 0.32), (0.48, 0.32)]:
        v, f = _bp_screw(pu_x + cx, cy, -0.04, 0.015, 0.05)
        proc_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    parts.append(BPPart("processor", "PROCESSING UNIT", proc_meshes, [
        "Raspberry Pi 5 (8GB) or Jetson Orin Nano",
        "Pi 5: 2.4 GHz quad-core, <8W",
        "Jetson: 1024 CUDA cores, GPU FFT",
        "HiFiBerry DAC+ ADC Pro",
        "Heatsink + fan + GPIO (40-pin)",
        "USB-A x2 + USB-C + HDMI + Ethernet",
        "SD card + eMMC + status LEDs",
    ]))

    # ===================== DISPLAY =====================
    disp_x = 6.9
    disp_meshes = []
    # Backing panel
    v, f = _bp_box(disp_x, 0, 0.05, 0.95, 0.65, 0.06)
    disp_meshes.append(BPMesh(v, f, C_PANEL, "Display Back"))
    # Bezel frame
    v, f = _bp_box(disp_x, 0, 0.12, 0.90, 0.60, 0.04)
    disp_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Bezel"))
    # Screen face (glowing)
    v, f = _bp_box(disp_x, 0, 0.15, 0.72, 0.48, 0.02)
    disp_meshes.append(BPMesh(v, f, C_ACCENT, "LCD Panel", alpha=180))
    # Touch panel overlay
    v, f = _bp_box(disp_x, 0, 0.165, 0.74, 0.50, 0.005)
    disp_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Touch Layer", alpha=60))
    # FPC ribbon cable
    v, f = _bp_box(disp_x - 0.45, 0, 0.08, 0.30, 0.08, 0.02)
    disp_meshes.append(BPMesh(v, f, C_RAW, "FPC Cable", alpha=150))
    # Mount bracket
    v, f = _bp_box(disp_x - 0.50, 0, -0.02, 0.15, 0.30, 0.08)
    disp_meshes.append(BPMesh(v, f, C_PANEL_HI, "Mount Bracket"))
    # Mount screws (2)
    for sy in (-0.10, 0.10):
        v, f = _bp_screw(disp_x - 0.50, sy, -0.04, 0.015, 0.05)
        disp_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Display driver IC
    v, f = _bp_box(disp_x + 0.30, 0.20, 0.08, 0.10, 0.06, 0.03)
    disp_meshes.append(BPMesh(v, f, C_PANEL_HI, "Driver IC"))
    # Backlight strip
    v, f = _bp_box(disp_x, -0.28, 0.10, 0.80, 0.03, 0.03)
    disp_meshes.append(BPMesh(v, f, C_WARN, "Backlight", alpha=150))
    parts.append(BPPart("display", "DISPLAY", disp_meshes, [
        "5-inch HDMI LCD 800x480",
        "Capacitive touch overlay",
        "Side-mounted or headless",
        "Bluetooth phone app option",
        "Bezel + FPC + mount + driver IC",
        "Backlight LED strip",
    ]))

    # ===================== POWER SYSTEM =====================
    pwr_meshes = []
    # Battery pack enclosure
    v, f = _bp_box(0, 0, -1.2, 1.7, 0.85, 0.45)
    pwr_meshes.append(BPMesh(v, f, C_PANEL, "Battery Enclosure", alpha=225))
    # 4x 18650 cells with terminals + wrapper lines
    for i in range(4):
        cx = -0.55 + i * 0.37
        # Cell body
        v, f = _bp_cyl(0.075, -1.40, -1.00, 12)
        verts = [(p[0] + cx, p[1] - 0.15, p[2]) for p in v]
        pwr_meshes.append(BPMesh(verts, f, C_WARN, "18650-%d" % (i + 1)))
        # Positive terminal
        v2, f2 = _bp_cyl(0.028, -1.00, -0.96, 8)
        verts2 = [(p[0] + cx, p[1] - 0.15, p[2]) for p in v2]
        pwr_meshes.append(BPMesh(verts2, f2, C_PANEL_HI, "+"))
        # Negative terminal
        v3, f3 = _bp_cyl(0.05, -1.42, -1.40, 8)
        verts3 = [(p[0] + cx, p[1] - 0.15, p[2]) for p in v3]
        pwr_meshes.append(BPMesh(verts3, f3, C_TEXT_DIM, "-"))
        # Wrapper line
        v4, f4 = _bp_ring(0.076, 0.072, -1.20, 16)
        verts4 = [(p[0] + cx, p[1] - 0.15, p[2]) for p in v4]
        pwr_meshes.append(BPMesh(verts4, f4, C_TEXT_DIM, "", alpha=100))
    # BMS protection board with traces
    v, f = _bp_box(0, 0.25, -1.15, 0.8, 0.25, 0.06)
    pwr_meshes.append(BPMesh(v, f, C_GOOD, "BMS Board"))
    v, f = _bp_grid(0, 0.25, -1.10, 0.7, 0.20, 6, 3, 0.003)
    pwr_meshes.append(BPMesh(v, f, C_ACCENT, "BMS Traces", alpha=80))
    # BMS ICs (2x)
    for i in range(2):
        v, f = _bp_box(-0.15 + i * 0.20, 0.25, -1.08, 0.08, 0.06, 0.03)
        pwr_meshes.append(BPMesh(v, f, C_PANEL_HI, "BMS IC"))
    # BMS capacitors (3x)
    for i in range(3):
        v, f = _bp_cyl(0.02, -1.10, -1.04, 6)
        verts = [(p[0] - 0.25 + i * 0.20, p[1] + 0.30, p[2]) for p in v]
        pwr_meshes.append(BPMesh(verts, f, C_WARN, ""))
    # USB-C PD charge port
    v, f = _bp_box(0.75, 0.25, -1.0, 0.10, 0.08, 0.06)
    pwr_meshes.append(BPMesh(v, f, C_WARN, "USB-C PD"))
    # Power switch (rocker style)
    v, f = _bp_box(-0.75, 0.25, -1.0, 0.10, 0.10, 0.06)
    pwr_meshes.append(BPMesh(v, f, C_BAD, "Power Switch"))
    v, f = _bp_box(-0.75, 0.25, -0.96, 0.06, 0.06, 0.02)
    pwr_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Battery level LEDs (4)
    for i in range(4):
        v, f = _bp_sph(0.028, 6, 5)
        verts = [(p[0] - 0.30 + i * 0.12, p[1] + 0.38, p[2] - 1.10) for p in v]
        col = C_GOOD if i < 2 else C_WARN if i < 3 else C_BAD
        pwr_meshes.append(BPMesh(verts, f, col, "LED-%d" % (i + 1)))
    # Battery holder clips (2x)
    for cy in (-0.30, 0.0):
        v, f = _bp_box(0, cy, -1.42, 1.5, 0.04, 0.04)
        pwr_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Enclosure screws (4)
    for cx, cy in [(-0.75, -0.35), (0.75, -0.35), (-0.75, 0.35), (0.75, 0.35)]:
        v, f = _bp_screw(cx, cy, -1.42, 0.015, 0.05)
        pwr_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Battery gauge display
    v, f = _bp_box(0, -0.35, -0.95, 0.30, 0.08, 0.03)
    pwr_meshes.append(BPMesh(v, f, C_PANEL, "Gauge Display"))
    parts.append(BPPart("power", "POWER SYSTEM", pwr_meshes, [
        "4x 18650 Li-ion (3400 mAh each)",
        "14.8V nominal, ~13.6 Ah",
        "4-8 hour runtime",
        "USB-C PD 30W fast charge",
        "BMS protection + traces + ICs",
        "Level LEDs + gauge + switch + screws",
    ]))

    # ===================== PISTOL GRIP =====================
    grip_meshes = []
    # Grip body upper (tapered)
    v, f = _bp_box(0, -0.3, -2.5, 0.45, 0.35, 1.3)
    grip_meshes.append(BPMesh(v, f, C_PANEL, "Pistol Grip"))
    # Grip body lower (narrower)
    v, f = _bp_box(0, -0.3, -3.5, 0.35, 0.28, 0.7)
    grip_meshes.append(BPMesh(v, f, C_PANEL, ""))
    # Grip pommel cap
    v, f = _bp_cyl(0.18, -4.0, -3.9, 10)
    verts = [(p[0], p[1] - 0.3, p[2]) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "Pommel"))
    # Trigger guard
    v, f = _bp_box(0.12, -0.42, -2.7, 0.04, 0.20, 0.30)
    grip_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Trigger Guard"))
    # REC trigger button
    v, f = _bp_sph(0.06, 10, 8)
    verts = [(p[0] + 0.10, p[1] - 0.38, p[2] - 2.85) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_BAD, "REC Trigger"))
    # Trigger spring (torus)
    v, f = _bp_torus(0.03, 0.008, 12, 6)
    verts = [(p[0] + 0.10, p[1] - 0.38, p[2] - 2.92) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_PANEL_HI, ""))
    # Mode selector dial + notch
    v, f = _bp_cyl(0.08, -1.8, -1.72, 14)
    verts = [(p[0] - 0.15, p[1] - 0.10, p[2]) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_ACCENT, "Mode Dial"))
    v, f = _bp_box(-0.15, -0.02, -1.70, 0.02, 0.08, 0.02)
    grip_meshes.append(BPMesh(v, f, C_WARN, ""))
    # Side panel buttons (3x)
    for i in range(3):
        v, f = _bp_sph(0.035, 6, 5)
        verts = [(p[0] + 0.22, p[1] - 0.10 + i * 0.12, p[2] - 2.2) for p in v]
        grip_meshes.append(BPMesh(verts, f, C_PANEL_HI, "Btn %d" % (i + 1)))
    # 1/4-20 tripod mount + thread
    v, f = _bp_cyl(0.065, -3.8, -3.65, 10)
    verts = [(p[0], p[1] - 0.3, p[2]) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "Tripod Mount"))
    v, f = _bp_ring(0.045, 0.035, -3.72, 8)
    verts = [(p[0], p[1] - 0.3, p[2]) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_PANEL_HI, ""))
    # Shoulder strap loop + torus
    v, f = _bp_box(-0.15, -0.48, -1.85, 0.08, 0.12, 0.10)
    grip_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Strap Loop"))
    v, f = _bp_torus(0.04, 0.012, 10, 6)
    verts = [(p[0] - 0.15, p[1] - 0.48, p[2] - 1.85) for p in v]
    grip_meshes.append(BPMesh(verts, f, C_TEXT_DIM, ""))
    # Grip texture ridges (6)
    for i in range(6):
        v, f = _bp_box(0.22, -0.46, -2.2 - i * 0.18, 0.02, 0.02, 0.16)
        grip_meshes.append(BPMesh(v, f, C_PANEL_HI, "", alpha=100))
    # Grip assembly screws (3)
    for sz in (-2.3, -3.0, -3.7):
        v, f = _bp_screw(-0.18, -0.46, sz, 0.012, 0.04)
        grip_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Wrist strap anchor
    v, f = _bp_box(0, -0.3, -4.1, 0.15, 0.10, 0.05)
    grip_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Wrist Anchor"))
    parts.append(BPPart("grip", "PISTOL GRIP", grip_meshes, [
        "3D-printed PETG/ABS housing",
        "Weatherproof IP54",
        "Trigger-style record button + spring",
        "Mode dial + 3 side buttons",
        "1/4-20 tripod + shoulder + wrist strap",
        "Pommel + texture ridges + screws",
    ]))

    # ===================== ENCLOSURE / FRAME =====================
    enc_meshes = []
    # Support rail
    v, f = _bp_box(2.0, 0, -0.5, 2.5, 0.08, 0.06)
    enc_meshes.append(BPMesh(v, f, C_PANEL, "Support Rail"))
    # Side panels (2x translucent)
    for sy in (-0.45, 0.45):
        v, f = _bp_box(3.0, sy, -0.3, 4.0, 0.04, 0.8)
        enc_meshes.append(BPMesh(v, f, C_PANEL, "Side Panel", alpha=90))
    # Ventilation slots (5x)
    for i in range(5):
        v, f = _bp_box(4.3 + i * 0.22, 0, 0.35, 0.12, 0.30, 0.02)
        enc_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Vent", alpha=70))
    # Weather seal gasket (torus)
    v, f = _bp_torus(0.22, 0.015, 20, 8)
    verts = [(p[0], p[1], p[2] - 0.45) for p in v]
    enc_meshes.append(BPMesh(verts, f, C_TEXT_DIM, "Seal Gasket", alpha=120))
    # Enclosure screws (6x on frame)
    for sx in (1.0, 3.0, 5.0):
        for sy in (-0.42, 0.42):
            v, f = _bp_screw(sx, sy, -0.48, 0.012, 0.04)
            enc_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Cable pass-through grommet
    v, f = _bp_torus(0.06, 0.02, 10, 6)
    verts = [(p[0] + 3.0, p[1] + 0.42, p[2] - 0.10) for p in v]
    enc_meshes.append(BPMesh(verts, f, C_WARN, "Cable Grommet"))
    # GPS antenna pad
    v, f = _bp_cyl(0.08, 0.30, 0.36, 12)
    verts = [(p[0] + 3.5, p[1], p[2]) for p in v]
    enc_meshes.append(BPMesh(verts, f, C_ACCENT, "GPS Antenna"))
    # IMU module
    v, f = _bp_box(3.8, -0.35, 0.30, 0.12, 0.08, 0.04)
    enc_meshes.append(BPMesh(v, f, C_GOOD, "IMU"))
    # Temperature sensor
    v, f = _bp_sph(0.025, 6, 5)
    verts = [(p[0] + 4.2, p[1] + 0.35, p[2] + 0.32) for p in v]
    enc_meshes.append(BPMesh(verts, f, C_WARN, "Temp Sensor"))
    parts.append(BPPart("enclosure", "ENCLOSURE / FRAME", enc_meshes, [
        "3D-printed PETG/ABS chassis",
        "Weatherproof IP54 gasket seals",
        "5 ventilation slots for thermal",
        "Side panels + support rail",
        "GPS antenna + IMU + temp sensor",
        "Cable grommet + 6 frame screws",
    ]))

    # ===================== CABLES & CONNECTORS =====================
    cable_meshes = []
    # Mic to preamp (shielded audio cable)
    v, f = _bp_cyl(0.028, 0.45, 3.35, 10)
    cable_meshes.append(BPMesh(v, f, C_RAW, "Mic Cable", alpha=150))
    # XLR connector at mic end
    v, f = _bp_cyl(0.04, 0.45, 0.52, 10)
    cable_meshes.append(BPMesh(v, f, C_PANEL_HI, "XLR Plug"))
    # Cable connector at preamp end
    v, f = _bp_cyl(0.04, 3.28, 3.42, 10)
    cable_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Preamp to processor (I2S ribbon)
    v, f = _bp_box(4.7, 0, 0.05, 0.55, 0.06, 0.03)
    cable_meshes.append(BPMesh(v, f, C_COOL, "I2S Ribbon", alpha=150))
    # I2S connectors (2x)
    for cx in (4.35, 5.05):
        v, f = _bp_box(cx, 0, 0.04, 0.08, 0.08, 0.05)
        cable_meshes.append(BPMesh(v, f, C_PANEL_HI, ""))
    # Processor to display (HDMI/MIPI)
    v, f = _bp_box(5.8, 0, 0.08, 0.55, 0.05, 0.03)
    cable_meshes.append(BPMesh(v, f, C_GOOD, "Display Cable", alpha=150))
    # HDMI connector
    v, f = _bp_box(5.72, 0, 0.07, 0.10, 0.08, 0.05)
    cable_meshes.append(BPMesh(v, f, C_PANEL_HI, "HDMI Plug"))
    # Power bus (battery to processor)
    v, f = _bp_cyl(0.035, -0.8, 4.7, 10)
    cable_meshes.append(BPMesh(v, f, C_WARN, "Power Bus", alpha=130))
    # Power connector at battery
    v, f = _bp_cyl(0.045, -0.85, -0.75, 10)
    cable_meshes.append(BPMesh(v, f, C_PANEL_HI, "PWR Plug"))
    # Power to preamp
    v, f = _bp_cyl(0.022, -0.8, 3.35, 8)
    cable_meshes.append(BPMesh(v, f, C_WARN, "", alpha=100))
    # Ground strap
    v, f = _bp_box(2.0, -0.35, -0.45, 2.0, 0.02, 0.02)
    cable_meshes.append(BPMesh(v, f, C_TEXT_DIM, "Ground", alpha=80))
    # GPS antenna cable
    v, f = _bp_cyl(0.015, 3.5, 5.0, 6)
    verts = [(p[0], p[1], p[2] + 0.35) for p in v]
    cable_meshes.append(BPMesh(verts, f, C_ACCENT, "GPS Cable", alpha=120))
    parts.append(BPPart("cables", "SIGNAL & POWER CABLES", cable_meshes, [
        "Shielded mic cable + XLR plugs",
        "I2S ribbon (preamp -> CPU) + connectors",
        "HDMI/MIPI (CPU -> display) + plug",
        "Power bus + PWR plug + preamp tap",
        "Ground strap + GPS antenna cable",
    ]))

    return parts


# =============================================================================
# SECTION 4 -- WIENER FILTER SPECTRAL ENHANCEMENT
# =============================================================================

# Noise profile learned from incoming audio (exponential moving average)
_noise_profile = None
_profile_frames = 0
_PROFILE_LEARN_FRAMES = 10  # frames to learn noise profile before enhancing
_voice_detected_flag = False  # set by wiener_enhance each call


def _update_noise_profile(magnitude_spectrum, is_voice):
    """Update noise floor estimate using exponential moving average.
    When voice is not detected, we accumulate noise statistics."""
    global _noise_profile, _profile_frames
    if _noise_profile is None:
        _noise_profile = magnitude_spectrum.copy()
        _profile_frames = 1
        return
    alpha = 0.05 if not is_voice else 0.01
    _noise_profile = (1 - alpha) * _noise_profile + alpha * magnitude_spectrum
    _profile_frames += 1


def wiener_enhance(audio, sr=SAMPLE_RATE, gain=ENHANCE_STRENGTH):
    """Real spectral enhancement using a Wiener filter.
    Estimates noise floor from non-voice frames, then applies
    H(f) = max(SNR / (SNR + 1), floor) per frequency bin.
    This is the same approach used in spectral subtraction and
    commercial noise reduction systems."""
    if len(audio) < 256:
        return audio * gain
    # Windowed FFT
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    # Detect voice: check if voice-band energy exceeds noise floor
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    voice_mask = (freqs >= VOICE_LOW) & (freqs <= VOICE_HIGH)
    voice_energy = np.mean(mag[voice_mask]) if np.any(voice_mask) else 0.0
    noise_energy = np.mean(mag[~voice_mask]) if np.any(~voice_mask) else 0.0
    is_voice = voice_energy > noise_energy * 1.5
    global _voice_detected_flag
    _voice_detected_flag = is_voice
    # Update noise profile
    _update_noise_profile(mag, is_voice)
    # If we haven't learned enough noise profile yet, pass through
    if _profile_frames < _PROFILE_LEARN_FRAMES:
        return audio * gain
    # Wiener filter: H = SNR / (SNR + 1)
    noise = _noise_profile + 1e-10
    snr = (mag ** 2) / (noise ** 2)
    wiener_gain = snr / (snr + 1.0)
    # Apply gain floor to avoid musical noise artifacts
    wiener_gain = np.maximum(wiener_gain, 0.1)
    # Boost voice band slightly
    wiener_gain[voice_mask] *= 1.2
    # Apply filter
    enhanced_mag = mag * wiener_gain * gain
    enhanced_spectrum = enhanced_mag * np.exp(1j * phase)
    # Inverse FFT with overlap-add simplification (just zero-pad inverse)
    enhanced = np.fft.irfft(enhanced_spectrum, n=len(audio))
    # Compensate for windowing energy loss
    window_energy = np.mean(window ** 2) + 1e-10
    enhanced = enhanced / np.sqrt(window_energy)
    return enhanced.astype(np.float32)


# =============================================================================
# SECTION 5 -- AUDIO PROCESSING CORE
# =============================================================================

# Cached filter coefficients (rebuilt when voice band changes)
_butter_cache = {}

def _get_butter_sos(low, high, sr=SAMPLE_RATE):
    key = (int(low), int(high), int(sr))
    if key not in _butter_cache:
        _butter_cache[key] = scipy_signal.butter(
            8, [low, high], btype='band', fs=sr, output='sos')
    return _butter_cache[key]

def voice_bandpass(audio, sr=SAMPLE_RATE, low=VOICE_LOW, high=VOICE_HIGH):
    sos = _get_butter_sos(low, high, sr)
    return scipy_signal.sosfilt(sos, audio)


def spectral_decoherence(audio, strength=NOISE_REDUCTION):
    f = np.fft.rfft(audio)
    mag = np.abs(f)
    phase = np.angle(f)
    noise_floor = np.percentile(mag, 12) * strength
    mag_clean = np.maximum(mag - noise_floor, mag * 0.15)
    cleaned = mag_clean * np.exp(1j * phase)
    return np.fft.irfft(cleaned, n=len(audio))


def wiener_enhance_audio(audio_chunk, gain=ENHANCE_STRENGTH):
    """Enhance audio using Wiener filter spectral subtraction."""
    return wiener_enhance(audio_chunk, gain=gain)


def compute_spectrum(audio, fft_size=FFT_SIZE):
    """Single-frame magnitude spectrum."""
    if len(audio) < fft_size:
        audio = np.pad(audio, (0, fft_size - len(audio)))
    windowed = audio[:fft_size] * np.hanning(fft_size)
    return np.abs(np.fft.rfft(windowed))


def estimate_distance(audio, sr=SAMPLE_RATE):
    """Estimate distance to source based on received signal level.

    NOTE: This uses digital dBFS (decibels relative to full scale).
    True acoustic distance estimation requires calibrated SPL measurement.
    The VOICE_DB_AT_1M reference is an approximation for conversational
    voice -- actual accuracy depends on microphone sensitivity, preamp
    gain, and source loudness. Treat results as a rough order-of-magnitude
    estimate, not precise distance.
    """
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    db = 20 * math.log10(rms)
    if db < -60:
        return 999.0, -60.0
    # Inverse-square law: each 6 dB drop ~ doubles distance
    # NOTE: VOICE_DB_AT_1M is an approximation. Real calibration needed.
    db_drop = VOICE_DB_AT_1M - db
    if db_drop < 0:
        return 1.0, db
    distance = 2 ** (db_drop / 6.0)
    return distance, db


def compute_snr(audio, sr=SAMPLE_RATE, low=VOICE_LOW, high=VOICE_HIGH):
    """Estimate signal-to-noise ratio in dB.

    Uses voice-band energy vs out-of-band energy as a proxy.
    This is an approximation -- with a single microphone, true SNR
    requires a reference noise recording. Values are relative and
    should be compared trend-wise, not as absolute measurements.
    """
    if len(audio) < 256:
        return 0.0
    # Voice-band energy vs out-of-band energy
    voice = voice_bandpass(audio, sr, low=low, high=high)
    voice_energy = np.mean(voice ** 2) + 1e-10
    # Noise estimate: difference between original and voice-band
    noise = audio - voice
    noise_energy = np.mean(noise ** 2) + 1e-10
    return 10 * math.log10(voice_energy / noise_energy)


def spectral_direction_analysis(audio, num_dirs=NUM_DIRECTIONS, sr=SAMPLE_RATE,
                                low=VOICE_LOW, high=VOICE_HIGH):
    """Frequency-domain energy distribution analysis.

    NOTE: With a SINGLE microphone, true direction-of-arrival estimation
    is NOT possible. This function computes a spectral energy profile
    across frequency sub-bands and maps them to directional bins for
    visualization. This is NOT real beamforming -- it shows which
    frequency bands are active, mapped around a compass for display.

    Real direction estimation requires a microphone array (2+ mics).
    """
    if len(audio) < FFT_SIZE:
        audio = np.pad(audio, (0, FFT_SIZE - len(audio)))
    spectrum = np.fft.rfft(audio[:FFT_SIZE] * np.hanning(FFT_SIZE))
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    voice_mask = (freqs >= low) & (freqs <= high)
    voice_mag = magnitudes * voice_mask
    # Split voice band into sub-bands and map to directional bins
    n_voice_bins = np.sum(voice_mask)
    if n_voice_bins == 0:
        return np.zeros(num_dirs)
    voice_freqs = freqs[voice_mask]
    voice_amps = magnitudes[voice_mask]
    # Assign each frequency bin to a direction bin (vectorized)
    bin_indices = (np.arange(n_voice_bins) * num_dirs // n_voice_bins) % num_dirs
    direction_bins = np.bincount(bin_indices, weights=voice_amps, minlength=num_dirs)
    # Normalize
    mx = np.max(direction_bins) + 1e-10
    return direction_bins / mx


# =============================================================================
# SECTION 5B -- ADVANCED DSP: SPECTRAL GATE, COMPRESSOR, ENHANCEMENT
# =============================================================================

def spectral_gate(audio, sr=SAMPLE_RATE, threshold=GATE_THRESHOLD,
                  strength=NOISE_REDUCTION, softness=GATE_SOFTNESS):
    """Spectral gate: attenuates frequency bins below a noise-derived threshold.
    Uses a soft tanh transition to avoid musical noise artifacts."""
    if len(audio) < 256:
        return audio
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    noise_floor = np.percentile(mag, 15) * threshold * strength
    gate = np.tanh(np.maximum(mag - noise_floor, 0) * softness / (noise_floor + 1e-10))
    gated_mag = mag * gate
    out = gated_mag * np.exp(1j * phase)
    return np.fft.irfft(out, n=len(audio)).astype(np.float32)


class MultibandCompressor:
    """3-band compressor for speech intelligibility enhancement.
    Splits audio into low/mid/high bands, applies per-band compression
    with makeup gain. Dramatically improves distant/quiet speech clarity."""

    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        nyq = sr / 2
        self.sos_low = scipy_signal.butter(
            4, COMP_XOVER_LOW / nyq, btype='low', output='sos')
        self.sos_mid = scipy_signal.butter(
            4, [COMP_XOVER_LOW / nyq, COMP_XOVER_HIGH / nyq],
            btype='band', output='sos')
        self.sos_high = scipy_signal.butter(
            4, COMP_XOVER_HIGH / nyq, btype='high', output='sos')
        self.env = [-60.0, -60.0, -60.0]
        self.gr = [0.0, 0.0, 0.0]
        self.enabled = True

    def process(self, audio):
        if not self.enabled or len(audio) < 64:
            return audio
        low = scipy_signal.sosfilt(self.sos_low, audio)
        mid = scipy_signal.sosfilt(self.sos_mid, audio)
        high = scipy_signal.sosfilt(self.sos_high, audio)
        bands = [low, mid, high]
        output = np.zeros_like(audio)
        for i, band in enumerate(bands):
            rms = np.sqrt(np.mean(band ** 2)) + 1e-10
            db = 20 * math.log10(rms)
            self.env[i] = 0.85 * self.env[i] + 0.15 * db
            if self.env[i] > COMP_THRESHOLD[i]:
                over = self.env[i] - COMP_THRESHOLD[i]
                gr_db = -over * (1 - 1.0 / COMP_RATIO[i])
            else:
                gr_db = 0.0
            self.gr[i] = gr_db
            gain = 10 ** ((gr_db + COMP_MAKEUP[i]) / 20)
            output += band * gain
        return np.clip(output, -1.0, 1.0).astype(np.float32)

    def reset(self):
        self.env = [-60.0, -60.0, -60.0]
        self.gr = [0.0, 0.0, 0.0]


def presence_boost(audio, sr=SAMPLE_RATE, boost_db=PRESENCE_BOOST_DB,
                   center=PRESENCE_CENTER, q=PRESENCE_Q):
    """Boost the 2-5 kHz presence band where speech consonants live.
    Improves intelligibility of distant/quiet speech by enhancing
    consonant sounds (s, t, k, p, etc.)."""
    if boost_db <= 0 or len(audio) < 64:
        return audio
    w0 = center / (sr / 2)
    bw = w0 / q
    low = max(1e-4, w0 - bw / 2)
    high = min(0.999, w0 + bw / 2)
    if low >= high:
        return audio
    sos = scipy_signal.butter(2, [low, high], btype='band', output='sos')
    boosted = scipy_signal.sosfilt(sos, audio)
    gain = 10 ** (boost_db / 20)
    return np.clip(audio + boosted * (gain - 1), -1.0, 1.0).astype(np.float32)


def dereverberate(audio, sr=SAMPLE_RATE, strength=DEREVERB_STRENGTH,
                  smoothing=DEREVERB_SMOOTHING):
    """Simple spectral de-reverberation via temporal magnitude smoothing.
    Reduces smearing/echo by smoothing spectral magnitudes across frames,
    suppressing late reflections that manifest as spectral tails."""
    if strength <= 0 or len(audio) < 256:
        return audio
    global _prev_dereverb_mag
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    if _prev_dereverb_mag is None or len(_prev_dereverb_mag) != len(mag):
        _prev_dereverb_mag = mag.copy()
    smoothed = smoothing * _prev_dereverb_mag + (1 - smoothing) * mag
    diff = mag - smoothed
    reduced = mag - diff * strength
    reduced = np.maximum(reduced, 0)
    _prev_dereverb_mag = smoothed
    out = reduced * np.exp(1j * phase)
    return np.fft.irfft(out, n=len(audio)).astype(np.float32)


_prev_dereverb_mag = None


def voice_activity_detect(audio, sr=SAMPLE_RATE, low=VOICE_LOW, high=VOICE_HIGH):
    """Proper Voice Activity Detection using three features:
    1. Energy in voice band
    2. Spectral flatness (noise is flat, speech is peaky)
    3. Zero-crossing rate (voiced speech has lower ZCR)
    Returns (is_voice, confidence 0-1)."""
    if len(audio) < 256:
        return False, 0.0
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    energy_score = min(rms / VAD_ENERGY_THRESH, 1.0)
    spectrum = np.fft.rfft(audio[:min(len(audio), FFT_SIZE)] *
                           np.hanning(min(len(audio), FFT_SIZE)))
    mag = np.abs(spectrum) + 1e-10
    flatness = np.exp(np.mean(np.log(mag))) / np.mean(mag)
    flatness_score = max(0, 1 - flatness / VAD_FLATNESS_THRESH)
    zcr = np.mean(np.abs(np.diff(np.sign(audio))) > 0)
    zcr_score = max(0, 1 - zcr / VAD_ZCR_THRESH)
    freqs = np.fft.rfftfreq(min(len(audio), FFT_SIZE), 1.0 / sr)
    voice_mask = (freqs >= low) & (freqs <= high)
    voice_energy = np.sum(mag[voice_mask] ** 2) if np.any(voice_mask) else 0
    total_energy = np.sum(mag ** 2) + 1e-10
    band_ratio = voice_energy / total_energy
    band_score = min(band_ratio * 3, 1.0)
    confidence = (0.3 * energy_score + 0.25 * flatness_score +
                  0.2 * zcr_score + 0.25 * band_score)
    is_voice = confidence > 0.5 and rms > VAD_ENERGY_THRESH
    return is_voice, float(confidence)


def a_weighted_db(audio, sr=SAMPLE_RATE):
    """A-weighted perceptual loudness in dB.
    More accurate representation of human hearing sensitivity than raw dBFS."""
    if len(audio) < 64:
        return -60.0
    # A-weighting IIR filter coefficients (simplified)
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.147
    a4 = (2 * math.pi * f4) ** 2
    a3 = (2 * math.pi * f3) ** 2
    a2 = (2 * math.pi * f2) ** 2
    a1 = (2 * math.pi * f1) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    w = 2 * math.pi * freqs
    num = (w ** 2) * a4
    den = ((w ** 2 + a1) * (w ** 2 + a2) *
           np.sqrt((w ** 2 + a3) * (w ** 2 + a4)) *
           ((w ** 2 + a4)))
    den[den == 0] = 1e-10
    weight = num / den
    weight = weight / np.max(weight + 1e-10)
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    mag = np.abs(spectrum) * weight
    rms_weighted = np.sqrt(np.mean(mag ** 2)) + 1e-10
    return 20 * math.log10(rms_weighted)


def spectral_centroid(audio, sr=SAMPLE_RATE):
    """Spectral centroid: weighted average frequency.
    Higher = brighter/sharper sound, lower = duller/muffled.
    Useful for characterizing voice vs noise vs wind."""
    if len(audio) < 256:
        return 0.0
    n = min(len(audio), FFT_SIZE)
    windowed = audio[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(windowed)) + 1e-10
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return float(np.sum(freqs * mag) / np.sum(mag))


def spectral_flatness(audio, sr=SAMPLE_RATE):
    """Spectral flatness: 0 = pure tone, 1 = white noise.
    Speech typically 0.1-0.4, noise 0.6-0.9."""
    if len(audio) < 256:
        return 0.0
    n = min(len(audio), FFT_SIZE)
    windowed = audio[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(windowed)) + 1e-10
    return float(np.exp(np.mean(np.log(mag))) / np.mean(mag))


def detect_wind_noise(audio, sr=SAMPLE_RATE):
    """Detect wind noise: characterized by high low-frequency energy
    (< 200 Hz) relative to total. Returns (is_wind, strength 0-1)."""
    if len(audio) < 256:
        return False, 0.0
    n = min(len(audio), FFT_SIZE)
    windowed = audio[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    low_mask = freqs < 200
    low_energy = np.sum(mag[low_mask] ** 2) if np.any(low_mask) else 0
    total_energy = np.sum(mag ** 2) + 1e-10
    ratio = low_energy / total_energy
    is_wind = ratio > 0.35
    return is_wind, float(min(ratio * 2, 1.0))


def speech_intelligibility_index(audio, sr=SAMPLE_RATE,
                                 low=VOICE_LOW, high=VOICE_HIGH):
    """Approximate Speech Intelligibility Index (SII).
    Based on speech-band SNR and band energy distribution.
    Returns 0-1 (higher = more intelligible)."""
    if len(audio) < 256:
        return 0.0
    snr_db = compute_snr(audio, sr, low, high)
    snr_linear = 10 ** (snr_db / 10)
    sii = snr_linear / (snr_linear + 1.0)
    n = min(len(audio), FFT_SIZE)
    windowed = audio[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    voice_mask = (freqs >= low) & (freqs <= high)
    if np.any(voice_mask):
        formant_score = 1.0 - spectral_flatness(audio, sr)
        sii = sii * 0.7 + formant_score * 0.3
    return float(clamp(sii, 0, 1))


def detect_clipping(audio, threshold=CLIP_THRESHOLD):
    """Detect if audio samples are clipping."""
    return float(np.max(np.abs(audio))) >= threshold


def formant_enhance(audio, sr=SAMPLE_RATE,
                    f1_center=FORMANT_F1_CENTER, f1_boost=FORMANT_F1_BOOST,
                    f2_center=FORMANT_F2_CENTER, f2_boost=FORMANT_F2_BOOST,
                    q=FORMANT_Q):
    """Enhance vowel formants F1 (~500 Hz) and F2 (~1500 Hz).
    Formants are the resonant frequencies of the vocal tract that define
    vowels. Boosting them improves vowel clarity and speech intelligibility,
    especially for distant/quiet speech where formants are weakened."""
    if len(audio) < 64:
        return audio
    nyq = sr / 2
    output = audio.copy()
    for center, boost_db in [(f1_center, f1_boost), (f2_center, f2_boost)]:
        if boost_db <= 0:
            continue
        w0 = center / nyq
        bw = w0 / q
        low = max(1e-4, w0 - bw / 2)
        high = min(0.999, w0 + bw / 2)
        if low >= high:
            continue
        sos = scipy_signal.butter(2, [low, high], btype='band', output='sos')
        boosted = scipy_signal.sosfilt(sos, audio)
        gain = 10 ** (boost_db / 20)
        output = output + boosted * (gain - 1)
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def spectral_super_resolution(audio, sr=SAMPLE_RATE,
                              strength=SUPER_RES_STRENGTH,
                              n_harmonics=SUPER_RES_HARMONICS):
    """Spectral super-resolution: reconstruct weakened high-frequency
    harmonics by detecting fundamental peaks and regenerating their
    harmonic series. Helps recover detail in distant/low-SNR speech
    where high frequencies are attenuated by air absorption."""
    if strength <= 0 or len(audio) < 256:
        return audio
    n = len(audio)
    window = np.hanning(n)
    spectrum = np.fft.rfft(audio * window)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    enhanced_mag = mag.copy()
    voice_mask = (freqs >= 80) & (freqs <= 2000)
    # Precompute percentile threshold once
    voice_mags = mag[voice_mask]
    if len(voice_mags) < 4:
        return audio
    threshold = np.percentile(voice_mags, 70)
    # Vectorized peak detection: mag[i] > mag[i-1] and mag[i] > mag[i+1]
    is_peak = (mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])
    # Apply voice mask and threshold
    peak_mask = np.zeros(len(mag), dtype=bool)
    peak_mask[1:-1] = is_peak & voice_mask[1:-1] & (mag[1:-1] > threshold)
    peak_indices = np.where(peak_mask)[0]
    for idx in peak_indices:
        f0 = freqs[idx]
        for h in range(2, n_harmonics + 2):
            harmonic_freq = f0 * h
            if harmonic_freq > sr / 2:
                break
            h_idx = int(harmonic_freq / (sr / 2) * len(mag))
            if 0 < h_idx < len(enhanced_mag):
                expected = mag[idx] / (h ** 1.5)
                if enhanced_mag[h_idx] < expected:
                    enhanced_mag[h_idx] = (enhanced_mag[h_idx] * (1 - strength) +
                                           expected * strength)
    out = enhanced_mag * np.exp(1j * phase)
    return np.fft.irfft(out, n=n).astype(np.float32)


def adaptive_wind_filter(audio, sr=SAMPLE_RATE, wind_active=False,
                         wind_strength=0.0):
    """Adaptive high-pass filter that activates when wind noise is detected.
    Cutoff frequency scales with wind strength for adaptive filtering."""
    if not wind_active or len(audio) < 64:
        return audio
    cutoff = WIND_HPF_FREQ + wind_strength * 200.0
    nyq = sr / 2
    wn = min(cutoff / nyq, 0.95)
    sos = scipy_signal.butter(WIND_HPF_ORDER, wn, btype='high', output='sos')
    return scipy_signal.sosfilt(sos, audio).astype(np.float32)


_loudness_state = -16.0

def loudness_normalize(audio, target_db=LOUDNESS_TARGET):
    """Loudness normalization with fast attack / slow release.
    Maintains consistent output level regardless of input distance.
    Uses RMS measurement with smoothed gain control."""
    global _loudness_state
    if len(audio) < 64:
        return audio
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    current_db = 20 * math.log10(rms)
    error = target_db - current_db
    if error > 0:
        _loudness_state += error * LOUDNESS_ATTACK
    else:
        _loudness_state += error * LOUDNESS_RELEASE
    _loudness_state = clamp(_loudness_state, -20, 30)
    gain = 10 ** (_loudness_state / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def composite_confidence_score(sii, vad_conf, snr_db, clipping=False):
    """Composite speech quality confidence score (0-1).
    Combines SII, VAD confidence, and SNR into a single metric.
    Penalizes clipping heavily."""
    if clipping:
        return 0.0
    snr_norm = clamp((snr_db + 10) / 40)
    score = 0.4 * sii + 0.3 * vad_conf + 0.3 * snr_norm
    return float(clamp(score, 0, 1))


def detect_pitch(audio, sr=SAMPLE_RATE, fmin=70, fmax=400):
    """Detect fundamental frequency using normalized autocorrelation.
    Returns (pitch_hz, confidence) or (0, 0) if no clear pitch.
    Uses FFT-based autocorrelation for O(n log n) performance."""
    if len(audio) < 256:
        return 0.0, 0.0
    n = len(audio)
    audio_dc = audio - np.mean(audio)
    rms = np.sqrt(np.mean(audio_dc ** 2))
    if rms < 1e-5:
        return 0.0, 0.0
    lag_min = int(sr / fmax)
    lag_max = min(int(sr / fmin), n // 2)
    if lag_max <= lag_min:
        return 0.0, 0.0
    # FFT-based autocorrelation (much faster than loop)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_audio = np.fft.rfft(audio_dc, fft_size)
    auto_corr = np.fft.irfft(fft_audio * np.conj(fft_audio))[:lag_max + 1]
    auto_corr /= (rms * rms * n + 1e-10)
    corr = auto_corr[lag_min:lag_max + 1]
    peak_idx = int(np.argmax(corr))
    if corr[peak_idx] > 0.3:
        pitch = sr / (lag_min + peak_idx)
        return float(pitch), float(corr[peak_idx])
    return 0.0, 0.0


def auto_detect_environment(audio, sr=SAMPLE_RATE, snr=0.0,
                            wind=False, wind_str=0.0, centroid=0.0,
                            flatness=0.0, distance=0.0):
    """Analyze audio characteristics and suggest the best preset.
    Returns (preset_name, reason_string)."""
    if wind and wind_str > 0.4:
        return 'noisy', 'wind noise detected (%.0f%%)' % (wind_str * 100)
    if distance > 150:
        return 'distant', 'far source (%.0f m)' % distance
    if snr < 5:
        return 'noisy', 'low SNR (%.1f dB)' % snr
    if flatness > 0.5:
        return 'noisy', 'high spectral flatness (%.2f)' % flatness
    if flatness < 0.3 and snr > 15:
        return 'indoor', 'clean signal (SNR %.1f dB, flatness %.2f)' % (snr, flatness)
    return 'outdoor', 'default outdoor profile'


# =============================================================================
# SECTION 6 -- AUDIO CAPTURE THREAD
# =============================================================================

class AudioCapture:
    """Threaded real-time audio capture and output using PyAudio."""

    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=30)
        self.running = False
        self.buffer = deque(maxlen=int(SAMPLE_RATE * BUFFER_SECONDS))
        self._pa = None
        self._stream = None
        self._out_stream = None
        self._thread = None
        self.available = False
        self.error_msg = ""
        self.device_index = None
        self.device_name = ""
        self.output_enabled = False
        self.output_volume = 0.5
        self._try_init()

    def _find_input_device(self, pa):
        """Enumerate audio devices and return the best input device index."""
        count = pa.get_device_count()
        candidates = []
        for i in range(count):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if info["maxInputChannels"] >= CHANNELS:
                candidates.append((i, info))
        if not candidates:
            return None, None
        # Prefer devices with "microphone" in the name (case-insensitive)
        for idx, info in candidates:
            name = info["name"].lower()
            if "microphone" in name or "mic" in name:
                return idx, info
        # Prefer "line" devices next
        for idx, info in candidates:
            name = info["name"].lower()
            if "line" in name:
                return idx, info
        # Fallback: first available input device
        return candidates[0][0], candidates[0][1]

    def _try_init(self):
        try:
            self._pa = pyaudio.PyAudio()
            # Find a real input device (don't rely on OS default)
            idx, info = self._find_input_device(self._pa)
            if idx is None:
                self.error_msg = "No input devices found (no microphone connected)"
                self.available = False
                return
            self.device_index = idx
            self.device_name = info["name"]
            # Use the device's native sample rate if it doesn't support 44100
            dev_sr = int(info["defaultSampleRate"])
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=idx,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._callback)
            # Also open an output stream for real-time monitoring
            try:
                self._out_stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=self._out_callback)
            except Exception:
                self._out_stream = None
            self.available = True
        except Exception as e:
            self.error_msg = str(e)
            self.available = False

    def _callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data.copy())
        return (in_data, pyaudio.paContinue)

    def _out_callback(self, in_data, frame_count, time_info, status):
        if self.output_enabled and not self.output_queue.empty():
            chunk = self.output_queue.get()
            chunk = chunk * self.output_volume
            chunk = np.clip(chunk, -1.0, 1.0)
            return (chunk.tobytes(), pyaudio.paContinue)
        return (np.zeros(frame_count, dtype=np.float32).tobytes(),
                pyaudio.paContinue)

    def start(self):
        if not self.available:
            return
        self.running = True
        if self._stream:
            self._stream.start_stream()
        if self._out_stream:
            self._out_stream.start_stream()

    def stop(self):
        self.running = False
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
        if self._out_stream:
            try:
                self._out_stream.stop_stream()
            except Exception:
                pass

    def put_output(self, chunk):
        """Queue enhanced audio for output playback."""
        if self._out_stream and self.output_enabled:
            try:
                self.output_queue.put_nowait(chunk.astype(np.float32))
            except queue.Full:
                pass

    def get_chunk(self):
        """Get latest audio chunk, draining any queued chunks.
        Returns None if queue empty or stream stopped."""
        # Check if input stream died (device disconnect)
        if self._stream and not self._stream.is_active():
            self.available = False
            self.error_msg = "Audio stream stopped (device may have disconnected)"
            return None
        chunk = None
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            self.buffer.extend(chunk)
        return chunk

    def get_buffer_array(self, n=None):
        """Return the ring buffer as a numpy array."""
        if n is not None:
            data = list(self.buffer)[-n:]
            return np.array(data) if data else np.zeros(0, dtype=np.float32)
        return np.array(self.buffer) if len(self.buffer) > 0 else np.zeros(0, dtype=np.float32)

    def close(self):
        self.stop()
        if self._stream:
            self._stream.close()
        if self._out_stream:
            self._out_stream.close()
        if self._pa:
            self._pa.terminate()


# =============================================================================
# SECTION 7 -- MAIN APPLICATION
# =============================================================================

class App:
    TOP_BAR_H     = 34
    LEFT_PANEL_W  = 280
    RIGHT_PANEL_W = 300
    BOTTOM_BAR_H  = 80

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(
            "Hearnocular -- Directional Hearing Digital Twin")
        self.W, self.H = 1480, 900
        self.screen = pygame.display.set_mode((self.W, self.H),
                                               pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        mono = "consolas,dejavusansmono,monospace"
        self.font    = pygame.font.SysFont(mono, 14)
        self.fs      = pygame.font.SysFont(mono, 12)
        self.fb      = pygame.font.SysFont(mono, 20, bold=True)
        self.fbig    = pygame.font.SysFont(mono, 30, bold=True)
        self.fhuge   = pygame.font.SysFont(mono, 48, bold=True)
        self.fsmall  = pygame.font.SysFont(mono, 11)
        self.fmicro  = pygame.font.SysFont(mono, 10)

        self.mode = MODE_SPECTRUM
        self.running = True
        self.show_help = False
        self.show_info = False
        self.info_scroll = 0
        self.help_scroll = 0
        self.paused = False
        # 3D blueprint renderer (lazy init)
        self._bp_renderer = None
        self._bp_drag = None        # None or (last_mouse_x, last_mouse_y)
        self._bp_show_labels = True

        # Audio capture
        self.capture = AudioCapture()
        if self.capture.available:
            self.capture.start()
        else:
            print("WARNING: Audio input unavailable -- %s" % self.capture.error_msg)
            print("No microphone detected.  All data panels will show NO INPUT.")

        # Processing state
        self.raw_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.filtered_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.gated_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.dereverbed_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.decohered_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.enhanced_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.compressed_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.presence_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.raw_spectrum = np.zeros(FFT_SIZE // 2 + 1)
        self.clean_spectrum = np.zeros(FFT_SIZE // 2 + 1)
        self.spectrogram_history = deque(maxlen=200)
        self.direction_bins = np.zeros(NUM_DIRECTIONS)
        self.distance_est = 0.0
        self.db_level = -60.0
        self.snr_est = 0.0
        self.peak_level = 0.0
        self.voice_detected = False
        self._frame = 0
        self.has_real_audio = False
        self._no_input_dismissed = False

        # Advanced metrics
        self.aweighted_db = -60.0
        self.centroid_hz = 0.0
        self.flatness = 0.0
        self.sii_score = 0.0
        self.wind_detected = False
        self.wind_strength = 0.0
        self.clipping = False
        self.vad_confidence = 0.0
        self.compressor = MultibandCompressor()
        self.gr_low = 0.0
        self.gr_mid = 0.0
        self.gr_high = 0.0

        # Feature toggles
        self.ab_compare = False       # B: toggle raw vs processed output
        self.dereverb_enabled = True  # D: toggle de-reverb
        self.compress_enabled = True  # C: toggle compressor
        self.gate_enabled = True      # G: toggle spectral gate
        self.presence_enabled = True  # V: toggle presence boost
        self.formant_enabled = True   # T: toggle formant enhancement
        self.superres_enabled = True  # X: toggle super-resolution
        self.wind_filter_enabled = True  # auto wind HPF
        self.loudness_enabled = True  # L: toggle loudness normalization
        self.auto_adapt_enabled = False  # E: toggle auto environment adaptation
        self.current_preset = None    # 1-4: load preset
        self.auto_adapt_reason = ""
        self._auto_adapt_counter = 0
        self.deep_listen = False      # K: max processing mode
        self.vad_record = False       # J: VAD-gated recording
        self._vad_record_active = False  # internal: currently recording within VAD mode
        self.snr_improvement = 0.0    # measured SNR improvement (dB)
        self._raw_snr_est = 0.0       # SNR of raw input before processing
        self.event_log = deque(maxlen=8)  # recent status events
        self.noise_profile = None       # learned noise spectrum for spectral subtraction
        self.noise_profile_frames = 0   # number of frames averaged into noise profile
        self.noise_profile_learning = False  # N key: capture ambient noise
        self.whisper_mode = False       # U key: max gain for very quiet sources
        self.freq_isolator_enabled = False  # Y key: isolate specific freq band
        self.freq_isolator_low = 300.0  # isolator low cutoff
        self.freq_isolator_high = 3000.0  # isolator high cutoff
        self.spectrum_snapshot = None   # frozen spectrum for analysis
        self.spectrum_snapshot_time = None
        self.session_start_time = time.time()
        self.golden_markers = []        # timestamps of marked moments during recording
        self.agc_enabled = False        # A key: automatic gain control
        self._agc_target_rms = 0.05     # AGC target RMS level
        self._agc_current_gain = 1.0    # AGC current gain factor
        self.activity_timeline = deque(maxlen=600)  # voice/sound activity over session
        self.dynamic_range_db = 0.0     # peak vs average dB difference
        self._peak_db_hold = -60.0      # peak dB hold for dynamic range
        self._avg_db_smooth = -60.0     # smoothed average dB
        self.eq_bass = 0.0             # 3-band EQ: bass gain (dB)
        self.eq_mid = 0.0              # 3-band EQ: mid gain (dB)
        self.eq_treble = 0.0           # 3-band EQ: treble gain (dB)
        self.replay_buffer = None       # cached audio for replay
        self.replay_playing = False     # O key: replay from ring buffer
        self._spectrogram_speed = 1.0   # PGUP/PGDN: spectrogram scroll speed

        # THD+N measurement
        self.thd_n_percent = 0.0        # total harmonic distortion + noise %
        self._thd_smooth = 0.0          # smoothed THD+N value

        # Audio source classification
        self.source_class = "unknown"   # speech / music / noise / tone / silence
        self._class_history = deque(maxlen=30)  # rolling classification votes

        # RT60 reverberation time estimation
        self.rt60_ms = 0.0             # estimated reverberation time in ms
        self._rt60_smooth = 0.0        # smoothed RT60 value

        # Audio quality grade (A-F composite)
        self.quality_grade = "--"      # letter grade A/B/C/D/F
        self.quality_score = 0.0       # 0-100 numeric score

        # Spectrum peak tracking (top 3 moving peaks)
        self.spectral_peaks = []       # list of (freq_hz, db) for top 3 peaks
        self._peak_trackers = []       # smoothed peak trackers [(freq, amp), ...]

        # Long-term dB histogram (level distribution)
        self._db_histogram = [0] * 30  # 30 bins from -60 to 0 dB
        self._db_hist_count = 0        # total samples collected

        # Session statistics (min/max/avg over session)
        self._session_snr_sum = 0.0
        self._session_snr_count = 0
        self._session_snr_min = 999.0
        self._session_snr_max = -999.0
        self._session_db_sum = 0.0
        self._session_db_count = 0
        self._session_db_min = 999.0
        self._session_db_max = -999.0
        self._session_conf_sum = 0.0
        self._session_conf_count = 0
        self._session_start_time = time.time()

        # VU meter with peak hold
        self._vu_level = 0.0          # smoothed VU level (0-1)
        self._vu_peak = 0.0           # peak hold level (0-1)
        self._vu_peak_decay = 0.0     # peak hold decay timer

        # Spectrum averaging mode
        self.spectrum_avg_enabled = False  # toggle with Z key
        self._spectrum_avg = None     # averaged spectrum buffer
        self._spectrum_avg_count = 0  # number of frames averaged

        # SNR history for graph
        self.snr_history = deque(maxlen=200)
        self.db_history = deque(maxlen=200)
        self.confidence_history = deque(maxlen=200)

        # New pipeline chunk buffers
        self.wind_filtered_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.formant_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.superres_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.loudness_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

        # Peak hold for spectrum display
        self._peak_hold = np.zeros(FFT_SIZE // 2 + 1)
        self._peak_hold_frames = np.zeros(FFT_SIZE // 2 + 1, dtype=int)

        # Frequency cursor (click on spectrum to read frequency)
        self._cursor_freq = None
        self._hover_freq = None   # mouse hover frequency for readout
        self._hover_db = None     # mouse hover dB value

        # Composite confidence and pitch
        self.confidence_score = 0.0
        self.pitch_hz = 0.0
        self.pitch_conf = 0.0
        self._noise_floor_est = 0.0
        self._proc_latency_ms = 0.0
        self._vu_peak_hold = 0.0
        self._vu_peak_hold_frames = 0
        self._vu_peak_hold_in = 0.0
        self._vu_peak_hold_in_frames = 0
        self._fps = 0.0
        self._fps_frames = 0
        self._fps_timer = time.time()

        # Temporal smoothing (exponential moving average)
        self._smoothing_alpha = 0.15
        self._smooth_distance = 0.0
        self._smooth_db = -60.0
        self._smooth_snr = 0.0
        self._smooth_peak = 0.0
        self._smooth_direction = np.zeros(NUM_DIRECTIONS)

        # Input gain (auto-normalization)
        self.input_gain = 1.0
        self._gain_alpha = 0.02
        self._target_rms = 0.15

        # Recording state
        self.recording = False
        self.record_buffer = []
        self.record_start_time = 0.0
        self.record_file = None

        # Focus lock
        self.focus_lock_enabled = False
        self.focus_direction = 0  # bin index
        self.focus_angle = 0.0    # degrees

        # Latency tracking
        self._latency_ms = 0.0
        self._update_start_time = 0.0

        # Cached mini-spectra for filter mode
        self._mini_specs = {}

        # Reset global noise profile for Wiener filter
        global _noise_profile, _profile_frames, _voice_detected_flag, _prev_dereverb_mag
        _noise_profile = None
        _profile_frames = 0
        _voice_detected_flag = False
        _prev_dereverb_mag = None

        # Directional sweep state
        self.sweep_angle = 0.0
        self.direction_history = deque(maxlen=60)

        # Sliders / controls
        self.noise_reduction = NOISE_REDUCTION
        self.enhance_strength = ENHANCE_STRENGTH
        self.voice_low = VOICE_LOW
        self.voice_high = VOICE_HIGH
        self.wind_filter_strength = 0.5       # 0.0-1.0 scales wind HPF cutoff
        self.dereverb_strength = DEREVERB_STRENGTH  # 0.0-0.6
        self.compressor_threshold = -24.0     # -40 to -10 dB, applied to mid band
        self._drag_slider = None

        # Mode hitboxes
        self._mode_hitboxes = {}
        self._slider_hitboxes = {}

        self.bg = None
        self._rebuild_bg()

    def _rebuild_bg(self):
        self.bg = pygame.Surface((self.W, self.H))
        vgradient(self.bg, BG_TOP, BG_BOT)

    def view_rect(self):
        return pygame.Rect(
            self.LEFT_PANEL_W + 4, self.TOP_BAR_H + 4,
            self.W - self.LEFT_PANEL_W - self.RIGHT_PANEL_W - 8,
            self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 8)

    # ---- events ----------------------------------------------------------

    def handle_events(self, dt):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.W, self.H = max(1120, e.w), max(700, e.h)
                self.screen = pygame.display.set_mode(
                    (self.W, self.H), pygame.RESIZABLE)
                self._rebuild_bg()
            elif e.type == pygame.KEYDOWN:
                self._key(e)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    if self._handle_mode_click(e.pos):
                        continue
                    if self._handle_left_btn_click(e.pos):
                        continue
                    if self._handle_slider_click(e.pos):
                        continue
                    # Dismiss no-input overlay on click in view area
                    if not self.has_real_audio and not self._no_input_dismissed:
                        rect = self.view_rect()
                        if rect.collidepoint(e.pos):
                            self._no_input_dismissed = True
                        continue
                    # Blueprint 3D orbit/pan drag
                    if self.mode == MODE_BLUEPRINT:
                        rect = self.view_rect()
                        if rect.collidepoint(e.pos):
                            mods = pygame.key.get_mods()
                            self._bp_drag = (e.pos[0], e.pos[1],
                                             bool(mods & pygame.KMOD_SHIFT))
                        continue
                    # Frequency cursor in spectrum mode
                    if self.mode == MODE_SPECTRUM and self.has_real_audio:
                        rect = self.view_rect()
                        bar_area = pygame.Rect(rect.x + 8, rect.y + 36,
                                               rect.w - 16, rect.h // 2 - 40)
                        if bar_area.collidepoint(e.pos):
                            n_bins = 256
                            frac = (e.pos[0] - bar_area.x - 4) / bar_area.w
                            self._cursor_freq = clamp(frac, 0, 1) * (SAMPLE_RATE / 2)
                        else:
                            self._cursor_freq = None
                elif e.button == 4 and self.mode == MODE_BLUEPRINT:
                    if self._bp_renderer:
                        self._bp_renderer.zoom(0.85)
                elif e.button == 5 and self.mode == MODE_BLUEPRINT:
                    if self._bp_renderer:
                        self._bp_renderer.zoom(1.0 / 0.85)
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self._drag_slider = None
                    self._bp_drag = None
            elif e.type == pygame.MOUSEMOTION:
                if self._drag_slider is not None:
                    self._handle_slider_drag(e.pos)
                # Blueprint 3D orbit/pan
                if self.mode == MODE_BLUEPRINT and self._bp_drag is not None:
                    dx = e.pos[0] - self._bp_drag[0]
                    dy = e.pos[1] - self._bp_drag[1]
                    if self._bp_drag[2]:  # shift = pan
                        if self._bp_renderer:
                            self._bp_renderer.pan(dx, dy)
                    else:  # normal = orbit
                        if self._bp_renderer:
                            self._bp_renderer.orbit(dx, dy)
                    self._bp_drag = (e.pos[0], e.pos[1], self._bp_drag[2])
                # Hover readout on spectrum plot
                if self.mode == MODE_SPECTRUM and self.has_real_audio:
                    rect = self.view_rect()
                    bar_area = pygame.Rect(rect.x + 8, rect.y + 36,
                                           rect.w - 16, rect.h // 2 - 40)
                    if bar_area.collidepoint(e.pos):
                        n_bins = 256
                        frac = clamp((e.pos[0] - bar_area.x - 4) / bar_area.w, 0, 1)
                        self._hover_freq = frac * (SAMPLE_RATE / 2)
                        hover_bin = int(frac * n_bins)
                        if 0 <= hover_bin < len(self.raw_spectrum):
                            self._hover_db = 20 * math.log10(self.raw_spectrum[hover_bin] + 1e-10)
                        else:
                            self._hover_db = None
                    else:
                        self._hover_freq = None
                        self._hover_db = None
                else:
                    self._hover_freq = None
                    self._hover_db = None
            elif e.type == pygame.MOUSEWHEEL:
                if self.show_info:
                    self.info_scroll = max(0, self.info_scroll - e.y * 30)
                elif self.show_help:
                    self.help_scroll = max(0, self.help_scroll - e.y * 30)
                elif self.mode == MODE_BLUEPRINT and self._bp_renderer:
                    self._bp_renderer.zoom(0.85 if e.y > 0 else 1.0 / 0.85)

    def _key(self, e):
        k = e.key
        # Blueprint mode specific keys (take priority)
        if self.mode == MODE_BLUEPRINT:
            if k == pygame.K_r and self._bp_renderer:
                self._bp_renderer.reset()
                self.event_log.appendleft("BLUEPRINT view reset")
                return
            elif k == pygame.K_l:
                self._bp_show_labels = not self._bp_show_labels
                self.event_log.appendleft("BLUEPRINT labels %s" % (
                    "on" if self._bp_show_labels else "off"))
                return
            elif k == pygame.K_e and self._bp_renderer:
                self._bp_renderer.exploded = 0.0 if self._bp_renderer.exploded > 0.5 else 1.0
                self.event_log.appendleft("BLUEPRINT exploded %s" % (
                    "on" if self._bp_renderer.exploded > 0.5 else "off"))
                return
            elif k == pygame.K_w and self._bp_renderer:
                self._bp_renderer.wireframe = not self._bp_renderer.wireframe
                self.event_log.appendleft("BLUEPRINT wireframe %s" % (
                    "on" if self._bp_renderer.wireframe else "off"))
                return
            elif k == pygame.K_a and self._bp_renderer:
                self._bp_renderer.auto_rotate = not self._bp_renderer.auto_rotate
                self.event_log.appendleft("BLUEPRINT auto-rotate %s" % (
                    "on" if self._bp_renderer.auto_rotate else "off"))
                return
            elif k == pygame.K_f and self._bp_renderer:
                self._bp_renderer.highlight_part = (
                    self._bp_renderer.highlight_part + 1) % (len(self._bp_renderer.parts) + 1)
                if self._bp_renderer.highlight_part >= len(self._bp_renderer.parts):
                    self._bp_renderer.highlight_part = -1
                if self._bp_renderer.highlight_part >= 0:
                    pname = self._bp_renderer.parts[self._bp_renderer.highlight_part].name
                    self.event_log.appendleft("BLUEPRINT focus: %s" % pname)
                else:
                    self.event_log.appendleft("BLUEPRINT focus: none")
                return
            elif k == pygame.K_t and self._bp_renderer:
                self._bp_renderer.show_edges = not self._bp_renderer.show_edges
                self.event_log.appendleft("BLUEPRINT edges %s" % (
                    "on" if self._bp_renderer.show_edges else "off"))
                return
        if k == pygame.K_ESCAPE:
            if self.show_help or self.show_info:
                self.show_help = False
                self.show_info = False
            elif not self.has_real_audio and not self._no_input_dismissed:
                self._no_input_dismissed = True
            else:
                self.running = False
        elif k == pygame.K_TAB:
            i = MODES.index(self.mode)
            self.mode = MODES[(i + 1) % len(MODES)]
        elif k == pygame.K_h:
            self.show_help = not self.show_help
            self.help_scroll = 0
        elif k == pygame.K_i:
            self.show_info = not self.show_info
            self.info_scroll = 0
        elif k == pygame.K_p:
            self.paused = not self.paused
        elif k == pygame.K_r:
            self.noise_reduction = NOISE_REDUCTION
            self.enhance_strength = ENHANCE_STRENGTH
            self.voice_low = VOICE_LOW
            self.voice_high = VOICE_HIGH
            self.wind_filter_strength = 0.5
            self.dereverb_strength = DEREVERB_STRENGTH
            self.compressor_threshold = -24.0
            # Reset noise profile and history
            global _noise_profile, _profile_frames, _voice_detected_flag
            _noise_profile = None
            _profile_frames = 0
            _voice_detected_flag = False
            self.spectrogram_history.clear()
            self.direction_history.clear()
            self.input_gain = 1.0
            print("Reset: sliders, noise profile, and history cleared")
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_m:
            self.capture.output_enabled = not self.capture.output_enabled
            state = "ON" if self.capture.output_enabled else "OFF"
            print("Audio monitor: %s" % state)
        elif k == pygame.K_UP:
            if self.show_info:
                self.info_scroll = max(0, self.info_scroll - 30)
            else:
                self.capture.output_volume = clamp(
                    self.capture.output_volume + 0.1, 0.0, 2.0)
        elif k == pygame.K_DOWN:
            if self.show_info:
                self.info_scroll += 30
            else:
                self.capture.output_volume = clamp(
                    self.capture.output_volume - 0.1, 0.0, 2.0)
        elif k == pygame.K_s:
            self._toggle_recording()
        elif k == pygame.K_f:
            self._toggle_focus_lock()
        elif k == pygame.K_b:
            self.ab_compare = not self.ab_compare
            state = "RAW" if self.ab_compare else "PROCESSED"
            print("A/B compare: %s" % state)
        elif k == pygame.K_d:
            self.dereverb_enabled = not self.dereverb_enabled
            print("De-reverb: %s" % ("ON" if self.dereverb_enabled else "OFF"))
        elif k == pygame.K_c:
            self.compress_enabled = not self.compress_enabled
            print("Compressor: %s" % ("ON" if self.compress_enabled else "OFF"))
        elif k == pygame.K_g:
            self.gate_enabled = not self.gate_enabled
            print("Spectral gate: %s" % ("ON" if self.gate_enabled else "OFF"))
        elif k == pygame.K_v:
            self.presence_enabled = not self.presence_enabled
            print("Presence boost: %s" % ("ON" if self.presence_enabled else "OFF"))
        elif k == pygame.K_1:
            if e.mod & pygame.KMOD_SHIFT:
                self._save_preset(0)
            else:
                self._load_custom_preset(0)
        elif k == pygame.K_2:
            if e.mod & pygame.KMOD_SHIFT:
                self._save_preset(1)
            else:
                self._load_custom_preset(1)
        elif k == pygame.K_3:
            if e.mod & pygame.KMOD_SHIFT:
                self._save_preset(2)
            else:
                self._load_custom_preset(2)
        elif k == pygame.K_4:
            if e.mod & pygame.KMOD_SHIFT:
                self._save_preset(3)
            else:
                self._load_custom_preset(3)
        elif k == pygame.K_t:
            self.formant_enabled = not self.formant_enabled
            print("Formant enhancement: %s" % ("ON" if self.formant_enabled else "OFF"))
        elif k == pygame.K_x:
            self.superres_enabled = not self.superres_enabled
            print("Spectral super-resolution: %s" % ("ON" if self.superres_enabled else "OFF"))
        elif k == pygame.K_l:
            self.loudness_enabled = not self.loudness_enabled
            print("Loudness normalization: %s" % ("ON" if self.loudness_enabled else "OFF"))
        elif k == pygame.K_e:
            self.auto_adapt_enabled = not self.auto_adapt_enabled
            print("Auto-adapt: %s" % ("ON" if self.auto_adapt_enabled else "OFF"))
        elif k == pygame.K_w:
            self.wind_filter_enabled = not self.wind_filter_enabled
            print("Wind filter: %s" % ("ON" if self.wind_filter_enabled else "OFF"))
        elif k == pygame.K_k:
            self.deep_listen = not self.deep_listen
            if self.deep_listen:
                self.noise_reduction = min(self.noise_reduction + 0.1, 0.95)
                self.enhance_strength = min(self.enhance_strength + 0.3, 2.5)
                self.dereverb_strength = min(self.dereverb_strength + 0.1, 0.6)
                self.presence_enabled = True
                self.formant_enabled = True
                self.superres_enabled = True
                self.gate_enabled = True
                self.dereverb_enabled = True
                self.compress_enabled = True
                self.loudness_enabled = True
                self.wind_filter_enabled = True
                self.event_log.appendleft("DEEP LISTEN engaged")
                print("Deep Listen: ON (all stages boosted)")
            else:
                self.event_log.appendleft("DEEP LISTEN disengaged")
                print("Deep Listen: OFF")
        elif k == pygame.K_j:
            self.vad_record = not self.vad_record
            if self.vad_record:
                self.event_log.appendleft("VAD-REC armed")
                print("VAD-gated recording: ARMED (will record when voice detected)")
            else:
                if self._vad_record_active:
                    self._toggle_recording()
                    self._vad_record_active = False
                self.event_log.appendleft("VAD-REC disarmed")
                print("VAD-gated recording: DISARMED")
        elif k == pygame.K_n:
            if not self.noise_profile_learning:
                self.noise_profile_learning = True
                self.noise_profile = None
                self.noise_profile_frames = 0
                self.event_log.appendleft("NOISE LEARNING started")
                print("Noise profile learning: CAPTURING (stay silent for 2-3s)")
            else:
                self.noise_profile_learning = False
                if self.noise_profile is not None:
                    self.event_log.appendleft("NOISE PROFILE saved (%d frames)" % self.noise_profile_frames)
                    print("Noise profile saved: %d frames averaged" % self.noise_profile_frames)
                else:
                    self.event_log.appendleft("NOISE LEARNING cancelled")
                    print("Noise profile learning: CANCELLED (no audio)")
        elif k == pygame.K_u:
            self.whisper_mode = not self.whisper_mode
            if self.whisper_mode:
                self.input_gain = min(self.input_gain * 3.0, 10.0)
                self.noise_reduction = min(self.noise_reduction + 0.15, 0.95)
                self.enhance_strength = min(self.enhance_strength + 0.5, 3.0)
                self.compress_enabled = True
                self.loudness_enabled = True
                self.superres_enabled = True
                self.event_log.appendleft("WHISPER MODE engaged")
                print("Whisper mode: ON (3x gain, boosted NR + enhance)")
            else:
                self.input_gain = max(self.input_gain / 3.0, 0.1)
                self.event_log.appendleft("WHISPER MODE disengaged")
                print("Whisper mode: OFF")
        elif k == pygame.K_y:
            self.freq_isolator_enabled = not self.freq_isolator_enabled
            if self.freq_isolator_enabled:
                self.event_log.appendleft("FREQ ISOLATOR on")
                print("Frequency isolator: ON (300-3000 Hz band only)")
            else:
                self.event_log.appendleft("FREQ ISOLATOR off")
                print("Frequency isolator: OFF")
        elif k == pygame.K_z:
            self.spectrum_avg_enabled = not self.spectrum_avg_enabled
            self._spectrum_avg = None
            self._spectrum_avg_count = 0
            if self.spectrum_avg_enabled:
                self.event_log.appendleft("SPECTRUM AVG on")
                print("Spectrum averaging: ON")
            else:
                self.event_log.appendleft("SPECTRUM AVG off")
                print("Spectrum averaging: OFF")
        elif k == pygame.K_F2:
            if self.spectrum_snapshot is not None:
                self.spectrum_snapshot = None
                self.spectrum_snapshot_time = None
                self.event_log.appendleft("Snapshot cleared")
                print("Spectrum snapshot: CLEARED")
            else:
                self.spectrum_snapshot = self.raw_spectrum.copy()
                self.spectrum_snapshot_time = time.time()
                self.event_log.appendleft("Snapshot captured")
                print("Spectrum snapshot: CAPTURED (F2 to clear)")
        elif k == pygame.K_F3:
            if self.recording:
                mark_time = time.time() - self.record_start_time
                self.golden_markers.append(mark_time)
                self.event_log.appendleft("GOLDEN MARK @ %.1fs" % mark_time)
                print("Golden moment marked at %.1fs" % mark_time)
        elif k == pygame.K_F4:
            self._export_spectrum_csv()
        elif k == pygame.K_F5:
            self._copy_metrics_clipboard()
        elif k == pygame.K_PAGEUP:
            self._spectrogram_speed = max(0.2, getattr(self, '_spectrogram_speed', 1.0) - 0.2)
            print("Spectrogram speed: %.1fx" % self._spectrogram_speed)
        elif k == pygame.K_PAGEDOWN:
            self._spectrogram_speed = min(3.0, getattr(self, '_spectrogram_speed', 1.0) + 0.2)
            print("Spectrogram speed: %.1fx" % self._spectrogram_speed)
        elif k == pygame.K_a:
            self.agc_enabled = not self.agc_enabled
            if self.agc_enabled:
                self.event_log.appendleft("AGC engaged")
                print("Automatic Gain Control: ON (target RMS=%.3f)" % self._agc_target_rms)
            else:
                self.event_log.appendleft("AGC disengaged")
                print("Automatic Gain Control: OFF")
        elif k == pygame.K_o:
            if not self.replay_playing and self.has_real_audio:
                buf = self.capture.get_buffer_array(SAMPLE_RATE * 6)
                if len(buf) > 0:
                    self.replay_buffer = buf.copy()
                    self.replay_playing = True
                    self.event_log.appendleft("REPLAY 6s buffer")
                    print("Replaying last 6 seconds of processed audio")
                    if self.capture.output_enabled:
                        self.capture.put_output(self.replay_buffer)
                    self.replay_playing = False
        elif k == pygame.K_LEFTBRACKET:
            self.eq_bass = max(-12.0, self.eq_bass - 1.0)
            print("EQ Bass: %.0f dB" % self.eq_bass)
        elif k == pygame.K_RIGHTBRACKET:
            self.eq_bass = min(12.0, self.eq_bass + 1.0)
            print("EQ Bass: %.0f dB" % self.eq_bass)
        elif k == pygame.K_SEMICOLON:
            self.eq_mid = max(-12.0, self.eq_mid - 1.0)
            print("EQ Mid: %.0f dB" % self.eq_mid)
        elif k == pygame.K_QUOTE:
            self.eq_mid = min(12.0, self.eq_mid + 1.0)
            print("EQ Mid: %.0f dB" % self.eq_mid)
        elif k == pygame.K_COMMA:
            self.eq_treble = max(-12.0, self.eq_treble - 1.0)
            print("EQ Treble: %.0f dB" % self.eq_treble)
        elif k == pygame.K_PERIOD:
            self.eq_treble = min(12.0, self.eq_treble + 1.0)
            print("EQ Treble: %.0f dB" % self.eq_treble)
        elif k == pygame.K_0:
            self.eq_bass = 0.0
            self.eq_mid = 0.0
            self.eq_treble = 0.0
            print("EQ reset to 0 dB")

    def _toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_buffer = []
            self.record_start_time = time.time()
            ts = time.strftime("%Y%m%d_%H%M%S")
            rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
            os.makedirs(rec_dir, exist_ok=True)
            self.record_file = os.path.join(rec_dir, "hearnocular_%s.wav" % ts)
            print("Recording started -> %s" % self.record_file)
        else:
            self.recording = False
            if self.record_buffer:
                audio_data = np.concatenate(self.record_buffer)
                sf.write(self.record_file, audio_data, SAMPLE_RATE)
                dur = time.time() - self.record_start_time
                # Save metadata sidecar
                meta_file = self.record_file.replace('.wav', '.json')
                meta = {
                    'filename': self.record_file,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'duration_sec': round(dur, 2),
                    'samples': len(audio_data),
                    'sample_rate': SAMPLE_RATE,
                    'device': self.capture.device_name,
                    'device_index': self.capture.device_index,
                    'pipeline': '11-stage DSP',
                    'processing_params': {
                        'noise_reduction': round(self.noise_reduction, 2),
                        'enhance_strength': round(self.enhance_strength, 2),
                        'voice_band': [self.voice_low, self.voice_high],
                        'wind_filter_strength': round(self.wind_filter_strength, 2),
                        'dereverb_strength': round(self.dereverb_strength, 2),
                        'compressor_threshold': round(self.compressor_threshold, 1),
                        'gate_enabled': self.gate_enabled,
                        'dereverb_enabled': self.dereverb_enabled,
                        'compress_enabled': self.compress_enabled,
                        'presence_enabled': self.presence_enabled,
                        'formant_enabled': self.formant_enabled,
                        'superres_enabled': self.superres_enabled,
                        'loudness_enabled': self.loudness_enabled,
                        'wind_filter_enabled': self.wind_filter_enabled,
                        'auto_adapt_enabled': self.auto_adapt_enabled,
                        'deep_listen': self.deep_listen,
                        'vad_record': self.vad_record,
                        'whisper_mode': self.whisper_mode,
                        'noise_profile_set': self.noise_profile is not None,
                        'freq_isolator_enabled': self.freq_isolator_enabled,
                        'agc_enabled': self.agc_enabled,
                        'eq_bass_db': round(self.eq_bass, 1),
                        'eq_mid_db': round(self.eq_mid, 1),
                        'eq_treble_db': round(self.eq_treble, 1),
                        'spectrogram_speed': round(self._spectrogram_speed, 2),
                    },
                    'preset': self.current_preset,
                    'final_metrics': {
                        'distance_m': round(self.distance_est, 1),
                        'snr_db': round(self.snr_est, 1),
                        'signal_db': round(self.db_level, 1),
                        'a_weighted_db': round(self.aweighted_db, 1),
                        'centroid_hz': round(self.centroid_hz, 1),
                        'flatness': round(self.flatness, 3),
                        'sii_score': round(self.sii_score, 2),
                        'vad_confidence': round(self.vad_confidence, 2),
                        'confidence_score': round(self.confidence_score, 2),
                        'pitch_hz': round(self.pitch_hz, 1),
                        'wind_detected': self.wind_detected,
                        'wind_strength': round(self.wind_strength, 2),
                        'clipping': self.clipping,
                        'voice_detected': self.voice_detected,
                        'proc_latency_ms': round(self._proc_latency_ms, 2),
                        'snr_improvement_db': round(self.snr_improvement, 2),
                        'dynamic_range_db': round(self.dynamic_range_db, 2),
                        'thd_n_percent': round(self.thd_n_percent, 2),
                        'source_class': self.source_class,
                        'rt60_ms': round(self.rt60_ms, 1),
                        'quality_grade': self.quality_grade,
                        'quality_score': round(self.quality_score, 1),
                        'spectral_peaks': self.spectral_peaks,
                        'db_histogram': self._db_histogram,
                        'db_histogram_samples': self._db_hist_count,
                        'session_stats': {
                            'duration_s': round(time.time() - self._session_start_time, 1),
                            'snr_avg': round(self._session_snr_sum / max(self._session_snr_count, 1), 2),
                            'snr_min': round(self._session_snr_min if self._session_snr_min < 999 else 0, 2),
                            'snr_max': round(self._session_snr_max if self._session_snr_max > -999 else 0, 2),
                            'db_avg': round(self._session_db_sum / max(self._session_db_count, 1), 2),
                            'db_min': round(self._session_db_min if self._session_db_min < 999 else 0, 2),
                            'db_max': round(self._session_db_max if self._session_db_max > -999 else 0, 2),
                            'conf_avg': round(self._session_conf_sum / max(self._session_conf_count, 1), 3),
                        },
                    },
                    'focus_lock': self.focus_lock_enabled,
                    'focus_angle_deg': round(self.focus_angle, 1) if self.focus_lock_enabled else None,
                    'golden_markers': [round(m, 2) for m in self.golden_markers],
                    'noise_profile_set': self.noise_profile is not None,
                    'whisper_mode': self.whisper_mode,
                    'freq_isolator': self.freq_isolator_enabled,
                    'freq_isolator_range': [self.freq_isolator_low, self.freq_isolator_high] if self.freq_isolator_enabled else None,
                    'spectrum_avg': self.spectrum_avg_enabled,
                    'algorithm': '11-stage: wind->bandpass->gate->dereverb->decoherence->wiener->compress->presence->formant->superres->loudness',
                }
                with open(meta_file, 'w') as f:
                    json.dump(meta, f, indent=2)
                print("Recording saved: %s (%.1f s)" % (self.record_file, dur))
                print("Metadata saved: %s" % meta_file)
            else:
                print("Recording cancelled (no data)")
            self.record_buffer = []
            self.golden_markers = []

    def _toggle_focus_lock(self):
        if not self.focus_lock_enabled:
            self.focus_lock_enabled = True
            self.focus_direction = int(np.argmax(self.direction_bins))
            self.focus_angle = self.focus_direction * 360 / NUM_DIRECTIONS
            print("Focus lock ON: direction %d deg (%s)" % (
                self.focus_angle, self._compass_dir(self.focus_angle)))
        else:
            self.focus_lock_enabled = False
            print("Focus lock OFF")

    def _apply_preset(self, idx):
        if idx < 0 or idx >= len(PRESET_NAMES):
            return
        name = PRESET_NAMES[idx]
        p = PRESETS[name]
        self.noise_reduction = p['noise_reduction']
        self.enhance_strength = p['enhance_strength']
        self.voice_low = p['voice_low']
        self.voice_high = p['voice_high']
        self.dereverb_enabled = p['dereverb'] > 0
        self.dereverb_strength = p['dereverb']
        self.compress_enabled = p['compress']
        self.wind_filter_strength = p.get('wind_filter', 0.5)
        self.compressor_threshold = p.get('comp_thresh', -24.0)
        self.current_preset = name
        global _noise_profile, _profile_frames, _prev_dereverb_mag
        _noise_profile = None
        _profile_frames = 0
        _prev_dereverb_mag = None
        self.compressor.reset()
        self.spectrogram_history.clear()
        self._peak_hold[:] = 0
        self._peak_hold_frames[:] = 0
        self.snr_history.clear()
        self.db_history.clear()
        self.confidence_history.clear()
        global _loudness_state
        _loudness_state = -16.0
        print("Preset: %s (NR=%.2f  gain=%.2fx  band=%d-%d  dereverb=%s  comp=%s)" % (
            name, p['noise_reduction'], p['enhance_strength'],
            p['voice_low'], p['voice_high'],
            "ON" if p['dereverb'] > 0 else "OFF",
            "ON" if p['compress'] else "OFF"))

    def _save_preset(self, idx):
        """Save current settings to a JSON preset file (Shift+1-4)."""
        name = PRESET_NAMES[idx] if idx < len(PRESET_NAMES) else "custom_%d" % idx
        preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
        os.makedirs(preset_dir, exist_ok=True)
        preset_file = os.path.join(preset_dir, "preset_%s.json" % name)
        settings = {
            'noise_reduction': round(self.noise_reduction, 3),
            'enhance_strength': round(self.enhance_strength, 3),
            'voice_low': round(self.voice_low, 1),
            'voice_high': round(self.voice_high, 1),
            'dereverb': round(self.dereverb_strength, 3) if self.dereverb_enabled else 0,
            'compress': self.compress_enabled,
            'wind_filter': round(self.wind_filter_strength, 3),
            'comp_thresh': round(self.compressor_threshold, 2),
            'gate_enabled': self.gate_enabled,
            'dereverb_enabled': self.dereverb_enabled,
            'presence_enabled': self.presence_enabled,
            'formant_enabled': self.formant_enabled,
            'superres_enabled': self.superres_enabled,
            'loudness_enabled': self.loudness_enabled,
            'wind_filter_enabled': self.wind_filter_enabled,
            'eq_bass': round(self.eq_bass, 1),
            'eq_mid': round(self.eq_mid, 1),
            'eq_treble': round(self.eq_treble, 1),
            'agc_enabled': self.agc_enabled,
            'freq_isolator_enabled': self.freq_isolator_enabled,
            'freq_isolator_low': round(self.freq_isolator_low, 1),
            'freq_isolator_high': round(self.freq_isolator_high, 1),
        }
        with open(preset_file, 'w') as f:
            json.dump(settings, f, indent=2)
        self.event_log.appendleft("PRESET SAVED: %s" % name)
        print("Preset saved: %s -> %s" % (name, preset_file))

    def _load_custom_preset(self, idx):
        """Load custom preset from JSON file if it exists, else fall back to built-in."""
        name = PRESET_NAMES[idx] if idx < len(PRESET_NAMES) else "custom_%d" % idx
        preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
        preset_file = os.path.join(preset_dir, "preset_%s.json" % name)
        if os.path.exists(preset_file):
            with open(preset_file, 'r') as f:
                p = json.load(f)
            self.noise_reduction = p['noise_reduction']
            self.enhance_strength = p['enhance_strength']
            self.voice_low = p['voice_low']
            self.voice_high = p['voice_high']
            self.dereverb_enabled = p.get('dereverb_enabled', p['dereverb'] > 0)
            self.dereverb_strength = p['dereverb']
            self.compress_enabled = p['compress']
            self.wind_filter_strength = p.get('wind_filter', 0.5)
            self.compressor_threshold = p.get('comp_thresh', -24.0)
            self.gate_enabled = p.get('gate_enabled', True)
            self.presence_enabled = p.get('presence_enabled', True)
            self.formant_enabled = p.get('formant_enabled', True)
            self.superres_enabled = p.get('superres_enabled', True)
            self.loudness_enabled = p.get('loudness_enabled', True)
            self.wind_filter_enabled = p.get('wind_filter_enabled', True)
            self.eq_bass = p.get('eq_bass', 0.0)
            self.eq_mid = p.get('eq_mid', 0.0)
            self.eq_treble = p.get('eq_treble', 0.0)
            self.agc_enabled = p.get('agc_enabled', False)
            self.freq_isolator_enabled = p.get('freq_isolator_enabled', False)
            self.freq_isolator_low = p.get('freq_isolator_low', 300.0)
            self.freq_isolator_high = p.get('freq_isolator_high', 3000.0)
            self.current_preset = name + " (custom)"
            global _loudness_state
            _loudness_state = -16.0
            self.compressor.reset()
            self.snr_history.clear()
            self.db_history.clear()
            self.confidence_history.clear()
            self.event_log.appendleft("PRESET LOADED: %s" % name)
            print("Custom preset loaded: %s <- %s" % (name, preset_file))
        else:
            self._apply_preset(idx)

    def _export_spectrum_csv(self):
        """Export current spectrum data to CSV file (F4)."""
        if not self.has_real_audio:
            print("No audio data to export")
            return
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
        os.makedirs(export_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(export_dir, "spectrum_%s.csv" % ts)
        freq_step = (SAMPLE_RATE / 2) / len(self.raw_spectrum)
        lines = ["frequency_hz,raw_db,clean_db,gain_db"]
        for i in range(len(self.raw_spectrum)):
            freq = i * freq_step
            raw_db = 20 * math.log10(self.raw_spectrum[i] + 1e-10)
            clean_db = 20 * math.log10(self.clean_spectrum[i] + 1e-10)
            gain_db = clean_db - raw_db
            lines.append("%.1f,%.2f,%.2f,%.2f" % (freq, raw_db, clean_db, gain_db))
        with open(csv_file, 'w') as f:
            f.write("\n".join(lines))
        self.event_log.appendleft("CSV exported")
        print("Spectrum CSV exported: %s (%d bins)" % (csv_file, len(lines) - 1))

    def _copy_metrics_clipboard(self):
        """Copy current metrics summary to clipboard (F5)."""
        import subprocess
        metrics = (
            "Hearnocular Metrics @ %s\n"
            "  Distance: %.1f m  SNR: %.1f dB  Signal: %.1f dB\n"
            "  SII: %.2f  Confidence: %.2f  Pitch: %.1f Hz\n"
            "  Centroid: %.1f Hz  Flatness: %.3f  Dyn Range: %.1f dB\n"
            "  SNR Improvement: %.1f dB  Latency: %.1f ms\n"
            "  THD+N: %.1f%%  Source: %s  RT60: %.0f ms\n"
            "  Quality: %s (%.0f/100)\n"
            "  Peaks: %s  dB Hist: %d samples\n"
            "  Session: SNR %.1f/%.1f/%.1f  dB %.1f/%.1f/%.1f  Conf %.0f%%  SpecAvg: %s\n"
            "  Preset: %s  Voice: %s  Wind: %s (%.0f%%)\n"
            "  AGC: %s  EQ: B%+.0f M%+.0f T%+.0f  Whisper: %s\n"
            "  Gate:%s DR:%s Comp:%s Pres:%s Form:%s SR:%s Loud:%s Wind:%s"
            % (
                time.strftime("%H:%M:%S"),
                self.distance_est, self.snr_est, self.db_level,
                self.sii_score, self.confidence_score, self.pitch_hz,
                self.centroid_hz, self.flatness, self.dynamic_range_db,
                self.snr_improvement, self._proc_latency_ms,
                self.thd_n_percent, self.source_class, self.rt60_ms,
                self.quality_grade, self.quality_score,
                "; ".join("P%d:%.0fHz/%.0fdB" % (i+1, f, d) for i, (f, d) in enumerate(self.spectral_peaks)) or "none",
                self._db_hist_count,
                self._session_snr_sum / max(self._session_snr_count, 1),
                self._session_snr_min if self._session_snr_min < 999 else 0,
                self._session_snr_max if self._session_snr_max > -999 else 0,
                self._session_db_sum / max(self._session_db_count, 1),
                self._session_db_min if self._session_db_min < 999 else 0,
                self._session_db_max if self._session_db_max > -999 else 0,
                self._session_conf_sum / max(self._session_conf_count, 1) * 100,
                "ON" if self.spectrum_avg_enabled else "off",
                self.current_preset or "custom",
                "YES" if self.voice_detected else "no",
                "YES" if self.wind_detected else "no", self.wind_strength * 100,
                "ON" if self.agc_enabled else "off",
                self.eq_bass, self.eq_mid, self.eq_treble,
                "ON" if self.whisper_mode else "off",
                "Y" if self.gate_enabled else "N",
                "Y" if self.dereverb_enabled else "N",
                "Y" if self.compress_enabled else "N",
                "Y" if self.presence_enabled else "N",
                "Y" if self.formant_enabled else "N",
                "Y" if self.superres_enabled else "N",
                "Y" if self.loudness_enabled else "N",
                "Y" if self.wind_filter_enabled else "N",
            )
        )
        try:
            subprocess.run(['clip'], input=metrics.encode('utf-8'),
                           shell=True, timeout=2)
            self.event_log.appendleft("Metrics copied to clipboard")
            print("Metrics copied to clipboard (F5)")
        except Exception as ex:
            print("Clipboard copy failed: %s" % ex)
            print(metrics)

    def _handle_mode_click(self, pos):
        for mode, rect in self._mode_hitboxes.items():
            if rect.collidepoint(pos):
                self.mode = mode
                return True
        return False

    def _handle_left_btn_click(self, pos):
        for key, rect in self._left_btn_hitboxes.items():
            if rect.collidepoint(pos):
                if key.startswith("preset_"):
                    idx = int(key.split("_")[1])
                    self._apply_preset(idx)
                elif key == "auto_adapt":
                    self.auto_adapt_enabled = not self.auto_adapt_enabled
                    print("Auto-adapt: %s" % (
                        "ON" if self.auto_adapt_enabled else "OFF"))
                return True
        return False

    def _handle_slider_click(self, pos):
        for key, rect in self._slider_hitboxes.items():
            if rect.inflate(16, 20).collidepoint(pos):
                self._drag_slider = key
                self._set_slider_from_x(key, pos[0])
                return True
        return False

    def _handle_slider_drag(self, pos):
        if self._drag_slider:
            self._set_slider_from_x(self._drag_slider, pos[0])

    def _set_slider_from_x(self, key, mx):
        rect = self._slider_hitboxes[key]
        frac = clamp((mx - rect.x) / rect.w)
        if key == "noise_reduction":
            self.noise_reduction = 0.3 + frac * 0.6
        elif key == "enhance_strength":
            self.enhance_strength = 0.5 + frac * 2.0
        elif key == "voice_low":
            self.voice_low = int(50 + frac * 1000)
            # Ensure minimum gap of 200 Hz above voice_low
            if self.voice_low >= self.voice_high - 200:
                self.voice_high = self.voice_low + 200
        elif key == "voice_high":
            self.voice_high = int(1000 + frac * 10000)
            # Ensure minimum gap of 200 Hz below voice_high
            if self.voice_high <= self.voice_low + 200:
                self.voice_low = self.voice_high - 200
        elif key == "wind_filter_strength":
            self.wind_filter_strength = frac
        elif key == "dereverb_strength":
            self.dereverb_strength = frac * 0.6
        elif key == "compressor_threshold":
            self.compressor_threshold = -40.0 + frac * 30.0  # -40 to -10 dB
        elif key == "eq_bass":
            self.eq_bass = -12.0 + frac * 24.0  # -12 to +12 dB
        elif key == "eq_mid":
            self.eq_mid = -12.0 + frac * 24.0
        elif key == "eq_treble":
            self.eq_treble = -12.0 + frac * 24.0

    # ---- update ----------------------------------------------------------

    def update(self, dt):
        self._frame += 1
        if self._bp_renderer and self.mode == MODE_BLUEPRINT:
            self._bp_renderer.tick(dt)
        if self.paused:
            return

        self._update_start_time = time.time()

        # Get real audio chunk from microphone
        chunk = self.capture.get_chunk()
        if chunk is not None:
            self.raw_chunk = chunk
            self.has_real_audio = True
            self._no_input_dismissed = False
        else:
            # No audio input available -- do NOT generate fake data.
            # Reset all metrics to zero/defaults to prevent stale display
            self.has_real_audio = False
            self._smooth_distance = 0.0
            self._smooth_db = -60.0
            self._smooth_snr = 0.0
            self._smooth_peak = 0.0
            self._smooth_direction[:] = 0.0
            self.direction_bins[:] = 0.0
            self.distance_est = 0.0
            self.db_level = -60.0
            self.snr_est = 0.0
            self.peak_level = 0.0
            self.voice_detected = False
            self.raw_spectrum[:] = 0.0
            self.clean_spectrum[:] = 0.0
            self._mini_specs = {}
            return

        # Input gain auto-normalization (slow AGC)
        rms = np.sqrt(np.mean(self.raw_chunk ** 2)) + 1e-10
        if rms > 1e-6:
            desired_gain = self._target_rms / rms
            self.input_gain = (1 - self._gain_alpha) * self.input_gain + \
                self._gain_alpha * desired_gain
            self.input_gain = clamp(self.input_gain, 0.1, 20.0)
        # AGC mode: faster adaptive gain on top of base input gain
        agc_gain = 1.0
        if self.agc_enabled and rms > 1e-6:
            agc_gain = self._agc_target_rms / rms
            agc_gain = clamp(agc_gain, 0.1, 20.0)
            self._agc_current_gain = 0.9 * self._agc_current_gain + 0.1 * agc_gain
            agc_gain = self._agc_current_gain
        audio = np.clip(self.raw_chunk * self.input_gain * agc_gain, -1.0, 1.0)

        # Activity timeline: record voice/sound detection state
        self.activity_timeline.append({
            'voice': self.voice_detected,
            'level': self.db_level,
            'time': time.time() - self.session_start_time
        })

        # Dynamic range measurement (peak vs average dB)
        if self.has_real_audio and self.db_level > -60:
            if self.db_level > self._peak_db_hold:
                self._peak_db_hold = self.db_level
            else:
                self._peak_db_hold = max(-60.0, self._peak_db_hold - 0.1)
            self._avg_db_smooth = 0.95 * self._avg_db_smooth + 0.05 * self.db_level
            self.dynamic_range_db = self._peak_db_hold - self._avg_db_smooth

        # Measure processing latency
        _proc_t0 = time.perf_counter()

        # Run 11-stage processing pipeline on REAL audio only
        # Stage 0: Adaptive wind filter
        if self.wind_filter_enabled and self.has_real_audio:
            is_wind, wind_str = detect_wind_noise(audio)
            if is_wind:
                self.wind_filtered_chunk = adaptive_wind_filter(
                    audio, wind_active=True,
                    wind_strength=wind_str * self.wind_filter_strength)
            else:
                self.wind_filtered_chunk = audio.copy()
        else:
            self.wind_filtered_chunk = audio.copy()
        # Stage 1: Bandpass filter
        self.filtered_chunk = voice_bandpass(
            self.wind_filtered_chunk, low=self.voice_low, high=self.voice_high)
        # Stage 2: Spectral gate
        if self.gate_enabled:
            self.gated_chunk = spectral_gate(
                self.filtered_chunk, strength=self.noise_reduction)
        else:
            self.gated_chunk = self.filtered_chunk.copy()
        # Stage 3: De-reverberation
        if self.dereverb_enabled:
            self.dereverbed_chunk = dereverberate(
                self.gated_chunk, strength=self.dereverb_strength)
        else:
            self.dereverbed_chunk = self.gated_chunk.copy()
        # Stage 4: Spectral decoherence (with learned noise profile if available)
        if self.noise_profile is not None and not self.noise_profile_learning:
            # Use learned noise profile for more accurate subtraction
            f = np.fft.rfft(self.dereverbed_chunk)
            mag = np.abs(f)
            phase = np.angle(f)
            # Subtract learned noise profile scaled by noise_reduction strength
            cleaned_mag = mag - self.noise_profile * self.noise_reduction
            cleaned_mag = np.maximum(cleaned_mag, mag * (1 - self.noise_reduction))
            self.decohered_chunk = np.fft.irfft(
                cleaned_mag * np.exp(1j * phase), n=len(self.dereverbed_chunk)
            ).astype(np.float32)
        else:
            self.decohered_chunk = spectral_decoherence(
                self.dereverbed_chunk, strength=self.noise_reduction)
        # Noise profile learning: accumulate spectrum during silence
        if self.noise_profile_learning and self.has_real_audio:
            raw_spec = np.abs(np.fft.rfft(self.dereverbed_chunk))
            if self.noise_profile is None:
                self.noise_profile = raw_spec.copy()
                self.noise_profile_frames = 1
            else:
                a = 0.1  # learning rate
                self.noise_profile = (1 - a) * self.noise_profile + a * raw_spec
                self.noise_profile_frames += 1
            # Auto-stop after ~120 frames (~2 seconds at 60fps)
            if self.noise_profile_frames >= 120:
                self.noise_profile_learning = False
                self.event_log.appendleft(
                    "NOISE PROFILE saved (%d frames)" % self.noise_profile_frames)
                print("Noise profile saved: %d frames averaged" % self.noise_profile_frames)
        # Stage 5: Wiener enhancement
        self.enhanced_chunk = wiener_enhance_audio(
            self.decohered_chunk, gain=self.enhance_strength)
        # Stage 6: Multi-band compression
        self.compressor.enabled = self.compress_enabled
        # Apply user-adjustable threshold to mid band (offset from default)
        _orig_mid_thresh = COMP_THRESHOLD[1]
        COMP_THRESHOLD[1] = self.compressor_threshold
        self.compressed_chunk = self.compressor.process(self.enhanced_chunk)
        COMP_THRESHOLD[1] = _orig_mid_thresh
        self.gr_low = self.compressor.gr[0]
        self.gr_mid = self.compressor.gr[1]
        self.gr_high = self.compressor.gr[2]
        # Stage 7: Presence boost
        if self.presence_enabled:
            self.presence_chunk = presence_boost(self.compressed_chunk)
        else:
            self.presence_chunk = self.compressed_chunk.copy()
        # Stage 8: Formant enhancement
        if self.formant_enabled:
            self.formant_chunk = formant_enhance(self.presence_chunk)
        else:
            self.formant_chunk = self.presence_chunk.copy()
        # Stage 9: Spectral super-resolution
        if self.superres_enabled:
            self.superres_chunk = spectral_super_resolution(self.formant_chunk)
        else:
            self.superres_chunk = self.formant_chunk.copy()
        # Stage 10: Loudness normalization
        if self.loudness_enabled:
            self.loudness_chunk = loudness_normalize(self.superres_chunk)
        else:
            self.loudness_chunk = self.superres_chunk.copy()
        # Frequency isolator: bandpass to isolate specific frequency range
        if self.freq_isolator_enabled:
            self.loudness_chunk = bandpass_filter(
                self.loudness_chunk, self.freq_isolator_low, self.freq_isolator_high)
        # 3-band EQ: bass (low shelf), mid (peak), treble (high shelf)
        if self.eq_bass != 0 or self.eq_mid != 0 or self.eq_treble != 0:
            eq_audio = self.loudness_chunk
            # Bass: low shelf at 200 Hz
            if self.eq_bass != 0:
                bass_gain = 10 ** (self.eq_bass / 20.0)
                eq_audio = bandpass_filter(eq_audio, 20, 200) * bass_gain + \
                    bandpass_filter(eq_audio, 200, SAMPLE_RATE // 2)
            # Mid: peak at 1 kHz
            if self.eq_mid != 0:
                mid_gain = 10 ** (self.eq_mid / 20.0)
                eq_audio = bandpass_filter(eq_audio, 200, 4000) * mid_gain + \
                    np.concatenate([bandpass_filter(eq_audio, 20, 200),
                                   bandpass_filter(eq_audio, 4000, SAMPLE_RATE // 2)])
            # Treble: high shelf at 4 kHz
            if self.eq_treble != 0:
                treble_gain = 10 ** (self.eq_treble / 20.0)
                eq_audio = bandpass_filter(eq_audio, 4000, SAMPLE_RATE // 2) * treble_gain + \
                    bandpass_filter(eq_audio, 20, 4000)
            self.loudness_chunk = np.clip(eq_audio, -1.0, 1.0).astype(np.float32)
        # Final output
        self.enhanced_chunk = np.clip(self.loudness_chunk, -1.0, 1.0)
        self._proc_latency_ms = (time.perf_counter() - _proc_t0) * 1000

        # Focus lock: suppress non-locked directions by applying a
        # directional gain based on spectral bin similarity
        if self.focus_lock_enabled and self.has_real_audio:
            buf = self.capture.get_buffer_array(FFT_SIZE * 2)
            if len(buf) >= FFT_SIZE:
                raw_dir = spectral_direction_analysis(
                    buf, low=self.voice_low, high=self.voice_high)
                # Compute similarity to locked direction
                focus_bin = self.focus_direction
                # Weight = closeness to focus direction (gaussian)
                angles = np.arange(NUM_DIRECTIONS) * 360 / NUM_DIRECTIONS
                focus_ang = self.focus_angle
                ang_diff = np.minimum(
                    np.abs(angles - focus_ang),
                    360 - np.abs(angles - focus_ang))
                direction_weight = np.exp(-(ang_diff ** 2) / (2 * 30 ** 2))
                # Apply weighted gain based on current direction energy
                current_dir = np.argmax(raw_dir)
                current_weight = direction_weight[current_dir]
                self.enhanced_chunk = self.enhanced_chunk * current_weight

        # A/B compare: send raw or processed audio to output
        if self.ab_compare:
            self.capture.put_output(audio)
            if self.recording:
                self.record_buffer.append(audio.copy())
        else:
            self.capture.put_output(self.enhanced_chunk)
            if self.recording:
                self.record_buffer.append(self.enhanced_chunk.copy())

        # VAD-gated recording: auto-start/stop recording based on voice detection
        if self.vad_record and self.has_real_audio:
            if self.voice_detected and not self._vad_record_active:
                self._toggle_recording()
                self._vad_record_active = True
                self.event_log.appendleft("VAD-REC: voice detected, recording")
            elif not self.voice_detected and self._vad_record_active:
                # Stop after 2 seconds of silence
                if not hasattr(self, '_vad_silence_start'):
                    self._vad_silence_start = time.time()
                elif time.time() - self._vad_silence_start > 2.0:
                    self._toggle_recording()
                    self._vad_record_active = False
                    self._vad_silence_start = None
                    self.event_log.appendleft("VAD-REC: silence, stopped")
            else:
                self._vad_silence_start = None

        # Spectra
        self.raw_spectrum = compute_spectrum(audio)
        self.clean_spectrum = compute_spectrum(self.enhanced_chunk)
        # Noise floor estimate (10th percentile of spectrum, smoothed)
        raw_nf = np.percentile(self.raw_spectrum[:256], 10)
        self._noise_floor_est = (1 - 0.05) * self._noise_floor_est + 0.05 * raw_nf

        # SNR improvement measurement (raw vs processed)
        if self.has_real_audio:
            raw_spec = self.raw_spectrum[:256]
            vb_lo = int(self.voice_low / (SAMPLE_RATE / 2) * 256)
            vb_hi = int(self.voice_high / (SAMPLE_RATE / 2) * 256)
            vb_lo = max(1, min(vb_lo, 255))
            vb_hi = max(vb_lo + 1, min(vb_hi, 256))
            voice_bins = raw_spec[vb_lo:vb_hi]
            noise_bins = np.concatenate([raw_spec[:vb_lo], raw_spec[vb_hi:]])
            if len(voice_bins) > 0 and len(noise_bins) > 0:
                voice_energy = np.mean(voice_bins ** 2) + 1e-10
                noise_energy = np.mean(noise_bins ** 2) + 1e-10
                self._raw_snr_est = 10 * math.log10(voice_energy / noise_energy)
            a = 0.05
            self.snr_improvement = (1 - a) * self.snr_improvement + a * (
                self.snr_est - self._raw_snr_est)
        # Cached mini-spectra for filter mode (avoid recompute in draw)
        self._mini_specs = {
            'raw': compute_spectrum(self.raw_chunk),
            'filtered': compute_spectrum(self.filtered_chunk),
            'gated': compute_spectrum(self.gated_chunk),
            'dereverbed': compute_spectrum(self.dereverbed_chunk),
            'decohered': compute_spectrum(self.decohered_chunk),
            'enhanced': compute_spectrum(self.enhanced_chunk),
            'compressed': compute_spectrum(self.compressed_chunk),
            'presence': compute_spectrum(self.presence_chunk),
            'formant': compute_spectrum(self.formant_chunk),
            'superres': compute_spectrum(self.superres_chunk),
            'loudness': self.clean_spectrum.copy(),
        }

        # Spectrogram history (speed-controlled: skip/duplicate frames)
        n_spectro_frames = max(1, int(round(self._spectrogram_speed)))
        for _ in range(n_spectro_frames):
            self.spectrogram_history.append(self.raw_spectrum.copy())

        # Directional analysis
        buf = self.capture.get_buffer_array(FFT_SIZE * 2)
        if len(buf) < FFT_SIZE:
            buf = audio
        raw_direction = spectral_direction_analysis(
            buf, low=self.voice_low, high=self.voice_high)

        # Distance + SNR + levels (raw values)
        raw_distance, raw_db = estimate_distance(buf)
        raw_snr = compute_snr(buf, low=self.voice_low, high=self.voice_high)
        raw_peak = float(np.max(np.abs(self.enhanced_chunk)))

        # Advanced metrics
        raw_aweighted = a_weighted_db(buf)
        raw_centroid = spectral_centroid(buf)
        raw_flatness = spectral_flatness(buf)
        raw_sii = speech_intelligibility_index(
            self.enhanced_chunk, low=self.voice_low, high=self.voice_high)
        is_wind, wind_str = detect_wind_noise(buf)
        is_clipping = detect_clipping(self.enhanced_chunk)
        vad_voice, vad_conf = voice_activity_detect(
            buf, low=self.voice_low, high=self.voice_high)

        # Peak hold update
        for i in range(min(len(self._peak_hold), len(self.raw_spectrum))):
            if self.raw_spectrum[i] > self._peak_hold[i]:
                self._peak_hold[i] = self.raw_spectrum[i]
                self._peak_hold_frames[i] = 0
            else:
                self._peak_hold_frames[i] += 1
                if self._peak_hold_frames[i] > PEAK_HOLD_FRAMES:
                    self._peak_hold[i] *= PEAK_DECAY_RATE

        # Temporal smoothing (exponential moving average)
        a = self._smoothing_alpha
        self._smooth_distance = (1 - a) * self._smooth_distance + a * raw_distance
        self._smooth_db = (1 - a) * self._smooth_db + a * raw_db
        self._smooth_snr = (1 - a) * self._smooth_snr + a * raw_snr
        self._smooth_peak = (1 - a) * self._smooth_peak + a * raw_peak
        self._smooth_direction = (1 - a) * self._smooth_direction + a * raw_direction

        # Use smoothed values for display
        self.direction_bins = self._smooth_direction.copy()
        self.distance_est = self._smooth_distance
        self.db_level = self._smooth_db
        self.snr_est = self._smooth_snr
        self.peak_level = self._smooth_peak
        self.aweighted_db = (1 - a) * self.aweighted_db + a * raw_aweighted
        self.centroid_hz = (1 - a) * self.centroid_hz + a * raw_centroid
        self.flatness = (1 - a) * self.flatness + a * raw_flatness
        self.sii_score = (1 - a) * self.sii_score + a * raw_sii
        self.wind_detected = is_wind
        self.wind_strength = (1 - a) * self.wind_strength + a * wind_str
        self.clipping = is_clipping
        self.vad_confidence = (1 - a) * self.vad_confidence + a * vad_conf
        self.voice_detected = vad_voice and self.peak_level > 0.01

        # Composite confidence score
        raw_conf = composite_confidence_score(
            raw_sii, vad_conf, self.snr_est, clipping=is_clipping)
        self.confidence_score = (1 - a) * self.confidence_score + a * raw_conf

        # Pitch detection
        raw_pitch, raw_pitch_conf = detect_pitch(buf)
        self.pitch_hz = (1 - a) * self.pitch_hz + a * raw_pitch
        self.pitch_conf = (1 - a) * self.pitch_conf + a * raw_pitch_conf

        # THD+N measurement: ratio of non-fundamental energy to total
        if self.has_real_audio and self.pitch_hz > 70 and self.pitch_conf > 0.2:
            spec = self.raw_spectrum
            n_bins = len(spec)
            fund_bin = int(self.pitch_hz / (SAMPLE_RATE / 2) * n_bins)
            if fund_bin > 0 and fund_bin < n_bins:
                fund_energy = spec[fund_bin] ** 2
                total_energy = np.sum(spec ** 2) + 1e-10
                harmonic_energy = 0.0
                for h in range(2, 6):
                    h_bin = int(fund_bin * h)
                    if 0 <= h_bin < n_bins:
                        harmonic_energy += spec[h_bin] ** 2
                noise_energy = max(0, total_energy - fund_energy - harmonic_energy)
                thd_n = math.sqrt((harmonic_energy + noise_energy) / total_energy) * 100
                self._thd_smooth = 0.9 * self._thd_smooth + 0.1 * thd_n
                self.thd_n_percent = self._thd_smooth

        # Audio source classification using spectral features
        if self.has_real_audio and self.db_level > -60:
            if self.db_level < -50:
                vote = "silence"
            elif self.flatness > 0.7:
                vote = "noise"
            elif self.pitch_conf > 0.5 and self.flatness < 0.2:
                vote = "tone"
            elif self.voice_detected:
                vote = "speech"
            elif self.flatness < 0.35 and self.centroid_hz < 2000:
                vote = "speech"
            elif self.flatness < 0.4 and self.centroid_hz > 2000:
                vote = "music"
            else:
                vote = "noise"
            self._class_history.append(vote)
            # Majority vote
            from collections import Counter
            counts = Counter(self._class_history)
            self.source_class = counts.most_common(1)[0][0]

        # RT60 estimation: measure spectral decay rate from spectrogram history
        if self.has_real_audio and len(self.spectrogram_history) > 30:
            specs = list(self.spectrogram_history)
            # Compare energy in recent vs older frames in voice band
            recent = np.mean(specs[-5:], axis=0)
            older = np.mean(specs[-30:-25], axis=0)
            # Focus on voice band bins
            n_bins = len(recent)
            vb_lo = int(self.voice_low / (SAMPLE_RATE / 2) * n_bins)
            vb_hi = int(self.voice_high / (SAMPLE_RATE / 2) * n_bins)
            vb_lo = max(0, vb_lo)
            vb_hi = min(n_bins, vb_hi)
            recent_e = np.mean(recent[vb_lo:vb_hi] ** 2) + 1e-10
            older_e = np.mean(older[vb_lo:vb_hi] ** 2) + 1e-10
            if older_e > recent_e and older_e > 1e-8:
                decay_db = 10 * math.log10(older_e / recent_e)
                # 25 frames elapsed between older and recent (~0.4s at 60fps)
                elapsed_s = 25.0 / 60.0
                if decay_db > 0.1:
                    # Extrapolate to 60 dB decay
                    rt60 = elapsed_s * (60.0 / decay_db) * 1000  # ms
                    rt60 = clamp(rt60, 0, 3000)
                    self._rt60_smooth = 0.95 * self._rt60_smooth + 0.05 * rt60
                    self.rt60_ms = self._rt60_smooth

        # Audio quality grade: composite score from SNR, SII, THD+N, confidence
        if self.has_real_audio:
            snr_score = clamp(self.snr_est / 30.0, 0, 1)      # 30 dB = max
            sii_score = clamp(self.sii_score, 0, 1)
            thd_score = clamp(1.0 - self.thd_n_percent / 30.0, 0, 1)  # 30% = worst
            conf_score = clamp(self.confidence_score, 0, 1)
            rt60_penalty = clamp(1.0 - self.rt60_ms / 2000.0, 0, 1) if self.rt60_ms > 0 else 0.8
            raw_score = (snr_score * 0.25 + sii_score * 0.25 +
                         thd_score * 0.20 + conf_score * 0.20 +
                         rt60_penalty * 0.10) * 100
            self.quality_score = 0.9 * self.quality_score + 0.1 * raw_score
            if self.quality_score >= 85:
                self.quality_grade = "A"
            elif self.quality_score >= 70:
                self.quality_grade = "B"
            elif self.quality_score >= 55:
                self.quality_grade = "C"
            elif self.quality_score >= 40:
                self.quality_grade = "D"
            else:
                self.quality_grade = "F"

        # Spectrum peak tracking: find top 3 spectral peaks
        if self.has_real_audio:
            spec = self.raw_spectrum
            n_bins = len(spec)
            # Find local maxima with minimum separation
            min_sep = max(2, n_bins // 100)
            candidates = []
            for i in range(2, n_bins - 2):
                if spec[i] > spec[i-1] and spec[i] > spec[i+1] and spec[i] > 0.01:
                    candidates.append((i, float(spec[i])))
            candidates.sort(key=lambda c: -c[1])
            # Pick top 3 with minimum separation
            picked = []
            for bin_idx, amp in candidates:
                if all(abs(bin_idx - p[0]) >= min_sep for p in picked):
                    picked.append((bin_idx, amp))
                if len(picked) >= 3:
                    break
            # Convert to frequency and dB, smooth with previous
            new_peaks = []
            for bin_idx, amp in picked:
                freq = bin_idx * (SAMPLE_RATE / 2) / n_bins
                db = 20 * math.log10(amp + 1e-10)
                # Smooth with previous trackers
                smoothed_freq = freq
                smoothed_db = db
                if self._peak_trackers:
                    for j, (pf, pa) in enumerate(self._peak_trackers):
                        if abs(pf - freq) < 50:  # within 50 Hz = same peak
                            smoothed_freq = 0.7 * pf + 0.3 * freq
                            smoothed_db = 0.7 * pa + 0.3 * db
                            break
                new_peaks.append((smoothed_freq, smoothed_db))
            self._peak_trackers = new_peaks
            self.spectral_peaks = [(round(f, 1), round(d, 1)) for f, d in new_peaks]

        # Long-term dB histogram
        if self.has_real_audio:
            db_val = self.db_level
            if -60 <= db_val <= 0:
                bin_idx = int((db_val + 60) / 60.0 * 30)
                bin_idx = clamp(bin_idx, 0, 29)
                self._db_histogram[bin_idx] += 1
                self._db_hist_count += 1

        # Session statistics accumulation
        if self.has_real_audio:
            self._session_snr_sum += self.snr_est
            self._session_snr_count += 1
            self._session_snr_min = min(self._session_snr_min, self.snr_est)
            self._session_snr_max = max(self._session_snr_max, self.snr_est)
            self._session_db_sum += self.db_level
            self._session_db_count += 1
            self._session_db_min = min(self._session_db_min, self.db_level)
            self._session_db_max = max(self._session_db_max, self.db_level)
            self._session_conf_sum += self.confidence_score
            self._session_conf_count += 1

        # VU meter with peak hold
        if self.has_real_audio:
            rms = float(np.sqrt(np.mean(self.loudness_chunk ** 2)))
            vu_target = clamp(rms * 3.0, 0, 1)  # scale for display
            self._vu_level = 0.8 * self._vu_level + 0.2 * vu_target
            if vu_target > self._vu_peak:
                self._vu_peak = vu_target
                self._vu_peak_decay = 0.0
            else:
                self._vu_peak_decay += 1
                if self._vu_peak_decay > 30:  # ~0.5s at 60fps
                    self._vu_peak = max(0, self._vu_peak - 0.02)

        # Spectrum averaging mode
        if self.has_real_audio and self.spectrum_avg_enabled:
            if self._spectrum_avg is None:
                self._spectrum_avg = self.raw_spectrum.copy()
                self._spectrum_avg_count = 1
            else:
                alpha = 0.05  # exponential moving average
                self._spectrum_avg = (1 - alpha) * self._spectrum_avg + alpha * self.raw_spectrum
                self._spectrum_avg_count += 1

        # SNR / dB history for graph
        self.snr_history.append(self.snr_est)
        self.db_history.append(self.db_level)
        self.confidence_history.append(self.confidence_score)

        # Auto-adapt environment detection
        self._auto_adapt_counter += 1
        if self.auto_adapt_enabled and self._auto_adapt_counter >= AUTO_ADAPT_INTERVAL:
            self._auto_adapt_counter = 0
            preset_name, reason = auto_detect_environment(
                buf, snr=self.snr_est, wind=self.wind_detected,
                wind_str=self.wind_strength, centroid=self.centroid_hz,
                flatness=self.flatness, distance=self.distance_est)
            self.auto_adapt_reason = reason
            if preset_name != self.current_preset:
                idx = PRESET_NAMES.index(preset_name) if preset_name in PRESET_NAMES else -1
                if idx >= 0:
                    self._apply_preset(idx)
                    print("Auto-adapt: %s (%s)" % (preset_name, reason))

        # Directional sweep
        self.sweep_angle += SWEEP_SPEED * dt
        self.direction_history.append(self.direction_bins.copy())

        # Latency measurement
        self._latency_ms = (time.time() - self._update_start_time) * 1000.0

    # ---- draw ------------------------------------------------------------

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        if self.mode == MODE_SPECTRUM:
            self.draw_spectrum()
        elif self.mode == MODE_DIRECTION:
            self.draw_direction()
        elif self.mode == MODE_FILTER:
            self.draw_filter()
        elif self.mode == MODE_METER:
            self.draw_meter()
        elif self.mode == MODE_BLUEPRINT:
            self.draw_blueprint()
        # Quality border indicator around view area
        if self.has_real_audio and not self.show_help and not self.show_info:
            rect = self.view_rect()
            conf = self.confidence_score
            if conf > 0.5:
                border_col = C_GOOD
            elif conf > 0.3:
                border_col = C_WARN
            else:
                border_col = C_BAD
            pygame.draw.rect(self.screen, border_col, rect, 2)
        if not self.has_real_audio and self.mode != MODE_BLUEPRINT:
            if not self._no_input_dismissed:
                self.draw_no_input_overlay()
        elif self.mode == MODE_BLUEPRINT and not self.has_real_audio:
            self.draw_no_input_badge()
        self.draw_topbar()
        self.draw_left_panel()
        self.draw_right_panel()
        self.draw_bottom_bar()
        if self.show_help:
            self.draw_help()
        if self.show_info:
            self.draw_info()
        pygame.display.flip()

    def draw_no_input_overlay(self):
        """Show a prominent NO AUDIO INPUT warning over the central view area."""
        rect = self.view_rect()
        # Darken the view area
        s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        s.fill((4, 6, 10, 180))
        self.screen.blit(s, (rect.x, rect.y))
        # Warning box
        bw, bh = 440, 120
        bx = rect.x + (rect.w - bw) // 2
        by = rect.y + (rect.h - bh) // 2
        panel(self.screen, bx, by, bw, bh, alpha=245)
        pygame.draw.rect(self.screen, C_BAD, (bx, by, bw, bh), 2,
                         border_radius=6)
        if not self.capture.available:
            title = self.fbig.render("NO AUDIO INPUT", True, C_BAD)
            msg = self.fs.render(
                "Plug in a microphone and restart to see real data.", True,
                C_TEXT_DIM)
        else:
            title = self.fbig.render("WAITING FOR AUDIO", True, C_WARN)
            msg = self.fs.render(
                "Microphone ready -- waiting for sound input...", True,
                C_TEXT_DIM)
        self.screen.blit(title, (bx + (bw - title.get_width()) // 2, by + 16))
        self.screen.blit(msg, (bx + (bw - msg.get_width()) // 2, by + 58))
        if self.capture.error_msg:
            err = self.fsmall.render(
                "Error: %s" % self.capture.error_msg[:60], True, C_TEXT_DIM)
            self.screen.blit(err, (bx + (bw - err.get_width()) // 2, by + 80))
        # Pulsing indicator
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 300.0)
        r = int(8 + 4 * pulse)
        indicator_col = C_BAD if not self.capture.available else C_WARN
        pygame.draw.circle(self.screen, indicator_col,
                           (bx + bw // 2, by + 100), r, 2)

    def draw_no_input_badge(self):
        """Small non-blocking badge in blueprint mode when no audio is present."""
        rect = self.view_rect()
        bw, bh = 200, 22
        bx = rect.x + 8
        by = rect.y + 46
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 400.0)
        col = (int(C_WARN[0] * pulse + C_PANEL_HI[0] * (1 - pulse)),
               int(C_WARN[1] * pulse + C_PANEL_HI[1] * (1 - pulse)),
               int(C_WARN[2] * pulse + C_PANEL_HI[2] * (1 - pulse)))
        s = pygame.Surface((bw, bh), pygame.SRCALPHA)
        s.fill((C_PANEL[0], C_PANEL[1], C_PANEL[2], 200))
        self.screen.blit(s, (bx, by))
        pygame.draw.rect(self.screen, col, (bx, by, bw, bh), 1, border_radius=4)
        dot_col = C_BAD if not self.capture.available else C_WARN
        pygame.draw.circle(self.screen, dot_col, (bx + 10, by + bh // 2), 4)
        label = "No audio input" if not self.capture.available else "Waiting for audio"
        self.screen.blit(self.fsmall.render(label, True, C_TEXT_DIM),
                         (bx + 20, by + 5))

    def draw_topbar(self):
        pygame.draw.rect(self.screen, C_PANEL,
                         (0, 0, self.W, self.TOP_BAR_H))
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (0, self.TOP_BAR_H),
                         (self.W, self.TOP_BAR_H), 1)
        self.screen.blit(self.fb.render("HEARNOCULAR", True, C_ACCENT),
                         (12, 6))
        self.screen.blit(self.font.render(
            "Directional Hearing  |  " + MODE_NAMES[self.mode],
            True, C_TEXT), (172, 10))
        # FPS and latency in topbar
        if self._fps > 0:
            fps_col = C_GOOD if self._fps > 50 else (C_WARN if self._fps > 30 else C_BAD)
            fps_txt = self.fs.render(
                "%.0f FPS  %.1fms" % (self._fps, self._proc_latency_ms),
                True, fps_col)
            self.screen.blit(fps_txt, (420, 10))
        # Clipping warning indicator
        tx = 540
        if self.clipping:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 200.0)
            clip_col = (int(C_BAD[0] * pulse + C_TEXT_DIM[0] * (1 - pulse)),
                        int(C_BAD[1] * pulse + C_TEXT_DIM[1] * (1 - pulse)),
                        int(C_BAD[2] * pulse + C_TEXT_DIM[2] * (1 - pulse)))
            clip_txt = self.fb.render("CLIP!", True, clip_col)
            self.screen.blit(clip_txt, (tx, 4))
            tx += clip_txt.get_width() + 12
        # Voice detected indicator
        if self.has_real_audio and self.voice_detected:
            vd_txt = self.fs.render("VOICE", True, C_GOOD)
            pygame.draw.circle(self.screen, C_GOOD, (tx + 6, 17), 5)
            self.screen.blit(vd_txt, (tx + 16, 10))
            tx += vd_txt.get_width() + 28
        elif self.has_real_audio:
            vd_txt = self.fs.render("silence", True, C_TEXT_DIM)
            self.screen.blit(vd_txt, (tx + 16, 10))
            tx += vd_txt.get_width() + 28
        # Wind noise indicator
        if self.has_real_audio and self.wind_detected:
            wind_txt = self.fs.render(
                "WIND %.0f%%" % (self.wind_strength * 100), True, C_WARN)
            self.screen.blit(wind_txt, (tx, 10))
            tx += wind_txt.get_width() + 12
        # Auto-adapt reason
        if self.auto_adapt_enabled and self.auto_adapt_reason:
            aa_txt = self.fsmall.render(
                self.auto_adapt_reason[:20], True, C_GOOD)
            self.screen.blit(aa_txt, (tx, 12))
            tx += aa_txt.get_width() + 12
        # Noise profile learning indicator
        if self.noise_profile_learning:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 300.0)
            nl_col = (int(C_COOL[0] * pulse + C_TEXT_DIM[0] * (1 - pulse)),
                      int(C_COOL[1] * pulse + C_TEXT_DIM[1] * (1 - pulse)),
                      int(C_COOL[2] * pulse + C_TEXT_DIM[2] * (1 - pulse)))
            nl_txt = self.fs.render("LEARNING NOISE...", True, nl_col)
            self.screen.blit(nl_txt, (tx, 10))
            tx += nl_txt.get_width() + 12
        # Session timer
        session_dur = time.time() - self.session_start_time
        session_txt = self.fsmall.render(
            "Session: %dm %02ds" % (int(session_dur) // 60, int(session_dur) % 60),
            True, C_TEXT_DIM)
        self.screen.blit(session_txt, (tx, 12))
        tx += session_txt.get_width() + 12
        # Snapshot indicator
        if self.spectrum_snapshot is not None:
            snap_txt = self.fs.render("SNAPSHOT", True, C_ACCENT)
            self.screen.blit(snap_txt, (tx, 10))
            tx += snap_txt.get_width() + 12
        # Golden marker count during recording
        if self.recording and self.golden_markers:
            gm_txt = self.fsmall.render(
                "*%d" % len(self.golden_markers), True, C_ACCENT)
            self.screen.blit(gm_txt, (tx, 12))
            tx += gm_txt.get_width() + 12
        # VU meter with peak hold
        if self.has_real_audio:
            vu_x = tx + 4
            vu_y = 12
            vu_w = 80
            vu_h = 10
            pygame.draw.rect(self.screen, C_PANEL_HI, (vu_x, vu_y, vu_w, vu_h), 1)
            # Level bar
            fill_w = int((vu_w - 2) * self._vu_level)
            if fill_w > 0:
                # Color zones: green <60%, yellow 60-85%, red >85%
                if self._vu_level < 0.6:
                    vu_col = C_GOOD
                elif self._vu_level < 0.85:
                    vu_col = C_WARN
                else:
                    vu_col = C_BAD
                pygame.draw.rect(self.screen, vu_col,
                                 (vu_x + 1, vu_y + 1, fill_w, vu_h - 2))
            # Peak hold marker
            if self._vu_peak > 0.01:
                peak_x = vu_x + 1 + int((vu_w - 2) * self._vu_peak)
                peak_col = C_BAD if self._vu_peak > 0.85 else (C_WARN if self._vu_peak > 0.6 else C_GOOD)
                pygame.draw.line(self.screen, peak_col,
                                 (peak_x, vu_y), (peak_x, vu_y + vu_h), 2)
            vu_lbl = self.fmicro.render("VU", True, C_TEXT_DIM)
            self.screen.blit(vu_lbl, (vu_x - vu_lbl.get_width() - 4, vu_y))
            tx = vu_x + vu_w + 12
        # Recording timer in top bar
        if self.recording:
            rec_dur = time.time() - self.record_start_time
            rec_txt = "REC %.1fs" % rec_dur
            rec_img = self.fb.render(rec_txt, True, C_BAD)
            # Pulsing red dot
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 200.0)
            dot_r = int(4 + 2 * pulse)
            dot_x = self.W - 540 - 30
            pygame.draw.circle(self.screen, C_BAD, (dot_x, 17), dot_r)
            self.screen.blit(rec_img, (dot_x - rec_img.get_width() - 8, 7))
        # Mode tabs
        self._mode_hitboxes = {}
        tab_x = self.W - 540
        tab_y = 4
        tab_h = 28
        for mode in MODES:
            label = MODE_NAMES[mode]
            active = (self.mode == mode)
            tw = self.fs.size(label)[0] + 24
            rect = pygame.Rect(tab_x, tab_y, tw, tab_h)
            panel(self.screen, rect.x, rect.y, rect.w, rect.h,
                  alpha=240 if active else 170)
            col = C_ACCENT if active else C_TEXT_DIM
            self.screen.blit(self.fs.render(label, True, col),
                             (rect.x + 12, rect.y + 8))
            self._mode_hitboxes[mode] = rect
            tab_x += tw + 6
        hint = "H help  I info  P pause  TAB mode"
        img = self.fs.render(hint, True, C_TEXT_DIM)
        self.screen.blit(img, (self.W - img.get_width() - 12,
                               self.TOP_BAR_H - 16))

    # ---- SPECTRUM MODE ---------------------------------------------------

    def draw_spectrum(self):
        rect = self.view_rect()
        # Title
        self.screen.blit(self.fb.render(
            "REAL-TIME SPECTRUM ANALYZER", True, C_ACCENT),
            (rect.x + 8, rect.y + 4))
        if self.spectrum_avg_enabled:
            avg_txt = self.fs.render(
                "AVG (%d frames)" % self._spectrum_avg_count, True, C_ACCENT)
            self.screen.blit(avg_txt, (rect.x + 280, rect.y + 8))

        # --- Spectrum bars ---
        if self.spectrum_avg_enabled and self._spectrum_avg is not None:
            spec = self._spectrum_avg
        else:
            spec = self.raw_spectrum
        clean = self.clean_spectrum
        n_bins = min(256, len(spec))
        bar_area = pygame.Rect(rect.x + 8, rect.y + 36,
                               rect.w - 16, rect.h // 2 - 40)
        panel(self.screen, bar_area.x, bar_area.y,
              bar_area.w, bar_area.h, alpha=180)
        self.screen.blit(self.fs.render(
            "FREQUENCY SPECTRUM  (raw vs enhanced)", True, C_TEXT_DIM),
            (bar_area.x + 10, bar_area.y + 6))

        plot_h = bar_area.h - 30
        plot_y = bar_area.y + 24
        bar_w = bar_area.w / n_bins
        mx = max(np.max(spec[:n_bins]), np.max(clean[:n_bins]), 1e-6)
        # Build spectrum bars using numpy array for performance
        bw_px = max(1, int(bar_w) - 1)
        surf_w = n_bins * bw_px
        spec_norm = np.clip(spec[:n_bins] / mx, 0, 1)
        clean_norm = np.clip(clean[:n_bins] / mx, 0, 1)
        # Vectorized bar drawing: build a 2D pixel array
        raw_heights = (spec_norm * plot_h).astype(np.int32)
        clean_heights = (clean_norm * plot_h).astype(np.int32)
        # Create pixel array (surf_w x plot_h), RGB
        pixels = np.full((surf_w, plot_h, 3), C_PANEL, dtype=np.uint8)
        for i in range(n_bins):
            x0 = i * bw_px
            x1 = x0 + bw_px
            hr = raw_heights[i]
            hc = clean_heights[i]
            if hr > 0:
                pixels[x0:x1, plot_h - hr:plot_h] = C_RAW
            if hc > 0:
                pixels[x0:x1, plot_h - hc:plot_h] = C_CLEAN
        spec_surf = pygame.surfarray.make_surface(pixels)
        self.screen.blit(spec_surf, (bar_area.x + 4, plot_y))

        # dB scale on Y-axis (left side)
        if mx > 1e-6:
            for db_val in [-60, -40, -20, -12, -6, -3, 0]:
                frac = 10 ** (db_val / 20)  # linear amplitude from dB
                y_pos = plot_y + plot_h - int(frac * plot_h)
                if plot_y <= y_pos <= plot_y + plot_h:
                    pygame.draw.line(self.screen, C_TEXT_DIM,
                                     (bar_area.x + 2, y_pos),
                                     (bar_area.x + 6, y_pos), 1)
                    img = self.fmicro.render("%+d" % db_val, True, C_TEXT_DIM)
                    self.screen.blit(img, (bar_area.x + 8, y_pos - 5))

        # Peak hold markers (optimized: only draw visible peaks)
        if self.has_real_audio:
            peak_norm = np.clip(self._peak_hold[:n_bins] / mx, 0, 1)
            peak_heights = (peak_norm * plot_h).astype(np.int32)
            visible = np.where(peak_heights > 2)[0]
            for i in visible:
                x = bar_area.x + 4 + i * bw_px
                ph = peak_heights[i]
                pygame.draw.rect(self.screen, (255, 255, 255),
                                 (x, plot_y + plot_h - ph, bw_px, 2))

        # Clipping indicator
        if self.clipping:
            clip_txt = self.fb.render("CLIPPING!", True, C_BAD)
            self.screen.blit(clip_txt, (bar_area.x + bar_area.w - clip_txt.get_width() - 8,
                                        bar_area.y + 4))

        # Spectral gain reduction overlay (where raw > clean = attenuation)
        if self.has_real_audio and self.gate_enabled:
            gr_diff = spec_norm - clean_norm
            gr_mask = np.where(gr_diff > 0.05)[0]
            for i in gr_mask:
                x = bar_area.x + 4 + i * bw_px
                hr = int(raw_heights[i])
                hc = int(clean_heights[i])
                if hr > hc:
                    gr_h = hr - hc
                    gr_surf = pygame.Surface((bw_px, gr_h), pygame.SRCALPHA)
                    gr_surf.fill((C_BAD[0], C_BAD[1], C_BAD[2], 80))
                    self.screen.blit(gr_surf, (x, plot_y + plot_h - hr))

        # Spectrum snapshot overlay (frozen spectrum for comparison)
        if self.spectrum_snapshot is not None and self.has_real_audio:
            snap_norm = np.clip(self.spectrum_snapshot[:n_bins] / mx, 0, 1)
            snap_heights = (snap_norm * plot_h).astype(np.int32)
            for i in range(n_bins):
                x0 = i * bw_px
                hc = snap_heights[i]
                if hc > 0:
                    snap_surf = pygame.Surface((bw_px, hc), pygame.SRCALPHA)
                    snap_surf.fill((C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], 60))
                    self.screen.blit(snap_surf, (bar_area.x + 4 + x0, plot_y + plot_h - hc))
            snap_lbl = self.fsmall.render(
                "SNAPSHOT (frozen)", True, C_ACCENT)
            self.screen.blit(snap_lbl, (bar_area.x + bar_area.w - 130, bar_area.y + 6))

        # Frequency cursor (click to read frequency + harmonics)
        if self._cursor_freq is not None and self.has_real_audio:
            cursor_bin = int(self._cursor_freq / (SAMPLE_RATE / 2) * n_bins)
            if 0 <= cursor_bin < n_bins:
                cx_line = bar_area.x + 4 + int(cursor_bin * bar_w)
                pygame.draw.line(self.screen, C_WARN,
                                 (cx_line, plot_y), (cx_line, plot_y + plot_h), 1)
                freq_txt = self.fsmall.render(
                    "%.0f Hz" % self._cursor_freq, True, C_WARN)
                self.screen.blit(freq_txt, (cx_line + 4, plot_y + 2))
                # Harmonic series readout
                harmonics = [1, 2, 3, 4, 5]
                for h in harmonics:
                    hf = self._cursor_freq * h
                    if hf >= SAMPLE_RATE / 2:
                        break
                    hbin = int(hf / (SAMPLE_RATE / 2) * n_bins)
                    if 0 <= hbin < n_bins:
                        hx = bar_area.x + 4 + int(hbin * bar_w)
                        pygame.draw.line(self.screen, C_TEXT_DIM,
                                         (hx, plot_y), (hx, plot_y + plot_h), 1)
                        h_db = 20 * math.log10(max(spec[hbin], 1e-6))
                        h_lbl = self.fmicro.render(
                            "H%d %.0fHz %.0fdB" % (h, hf, h_db),
                            True, C_TEXT_DIM)
                        self.screen.blit(h_lbl, (hx + 2, plot_y + 14 + (h - 1) * 12))

        # Hover readout (mouse over spectrum shows freq + dB)
        if self._hover_freq is not None and self._hover_db is not None and self.has_real_audio:
            h_bin = int(self._hover_freq / (SAMPLE_RATE / 2) * n_bins)
            if 0 <= h_bin < n_bins:
                hx = bar_area.x + 4 + int(h_bin * bar_w)
                # Dashed-style vertical line
                for ly in range(plot_y, plot_y + plot_h, 4):
                    pygame.draw.line(self.screen, C_TEXT_DIM,
                                     (hx, ly), (hx, min(ly + 2, plot_y + plot_h)), 1)
                # Readout box at top of plot
                hover_txt = "%.0f Hz  %.1f dB" % (self._hover_freq, self._hover_db)
                hover_img = self.fmicro.render(hover_txt, True, C_TEXT)
                box_w = hover_img.get_width() + 10
                box_x = min(hx + 6, bar_area.x + bar_area.w - box_w - 4)
                box_y = plot_y + 2
                pygame.draw.rect(self.screen, C_PANEL,
                                 (box_x, box_y, box_w, 16))
                pygame.draw.rect(self.screen, C_PANEL_HI,
                                 (box_x, box_y, box_w, 16), 1)
                self.screen.blit(hover_img, (box_x + 5, box_y + 1))

        # Frequency axis labels
        for f_label in [500, 1000, 2000, 4000, 8000]:
            bin_idx = int(f_label / (SAMPLE_RATE / 2) * n_bins)
            if bin_idx < n_bins:
                lx = bar_area.x + 4 + int(bin_idx * bar_w)
                pygame.draw.line(self.screen, C_TEXT_DIM,
                                 (lx, plot_y + plot_h),
                                 (lx, plot_y + plot_h + 4), 1)
                img = self.fmicro.render("%dk" % (f_label // 1000),
                                         True, C_TEXT_DIM)
                self.screen.blit(img, (lx - 6, plot_y + plot_h + 4))

        # Voice band highlight
        vl_bin = int(self.voice_low / (SAMPLE_RATE / 2) * n_bins)
        vh_bin = int(self.voice_high / (SAMPLE_RATE / 2) * n_bins)
        if 0 <= vl_bin < n_bins and 0 <= vh_bin <= n_bins:
            x0 = bar_area.x + 4 + int(vl_bin * bar_w)
            x1 = bar_area.x + 4 + int(vh_bin * bar_w)
            s = pygame.Surface((x1 - x0, plot_h), pygame.SRCALPHA)
            s.fill((C_VOICE[0], C_VOICE[1], C_VOICE[2], 30))
            self.screen.blit(s, (x0, plot_y))
            self.screen.blit(self.fmicro.render(
                "VOICE BAND", True, C_VOICE), (x0 + 2, plot_y + 2))

        # Spectral peak markers (top 3)
        if self.has_real_audio and self.spectral_peaks:
            for i, (freq, db) in enumerate(self.spectral_peaks):
                pk_bin = int(freq / (SAMPLE_RATE / 2) * n_bins)
                if 0 <= pk_bin < n_bins:
                    px = bar_area.x + 4 + int(pk_bin * bar_w)
                    tri_y = plot_y + 2 + i * 8
                    pygame.draw.polygon(self.screen, C_ACCENT, [
                        (px, tri_y + 6),
                        (px - 4, tri_y),
                        (px + 4, tri_y),
                    ])
                    pk_lbl = self.fmicro.render(
                        "P%d %.0fHz" % (i + 1, freq), True, C_ACCENT)
                    self.screen.blit(pk_lbl, (px + 6, tri_y - 2))

        # Confidence gauge bar (top-right of spectrum area)
        if self.has_real_audio:
            cg_x = bar_area.x + bar_area.w - 160
            cg_y = bar_area.y + 6
            cg_w = 140
            cg_h = 12
            self.screen.blit(self.fmicro.render(
                "CONFIDENCE", True, C_TEXT_DIM), (cg_x, cg_y - 12))
            pygame.draw.rect(self.screen, C_PANEL_HI,
                             (cg_x, cg_y, cg_w, cg_h), 1)
            conf_frac = clamp(self.confidence_score)
            conf_col = C_GOOD if conf_frac > 0.5 else (C_WARN if conf_frac > 0.3 else C_BAD)
            pygame.draw.rect(self.screen, conf_col,
                             (cg_x + 1, cg_y + 1, int((cg_w - 2) * conf_frac), cg_h - 2))
            conf_txt = self.fmicro.render(
                "%.0f%%" % (conf_frac * 100), True, conf_col)
            self.screen.blit(conf_txt, (cg_x + cg_w + 4, cg_y - 2))

        # Noise floor estimate line on spectrum
        if self.has_real_audio and hasattr(self, '_noise_floor_est'):
            nf = self._noise_floor_est
            if nf > 0:
                nf_y = plot_y + plot_h - int(nf / mx * plot_h)
                nf_y = clamp(nf_y, plot_y, plot_y + plot_h)
                # Dashed line for noise floor
                for dx in range(0, n_bins * bw_px, 6):
                    pygame.draw.line(self.screen, C_NOISE,
                                     (bar_area.x + 4 + dx, nf_y),
                                     (bar_area.x + 4 + dx + 3, nf_y), 1)
                nf_lbl = self.fmicro.render(
                    "noise floor", True, C_NOISE)
                self.screen.blit(nf_lbl, (bar_area.x + 8, nf_y - 12))

        # Pitch overlay - fundamental frequency line + harmonics
        if self.has_real_audio and self.pitch_hz > 70 and self.pitch_conf > 0.3:
            pitch_bin = int(self.pitch_hz / (SAMPLE_RATE / 2) * n_bins)
            if 0 <= pitch_bin < n_bins:
                px = bar_area.x + 4 + int(pitch_bin * bar_w)
                # Fundamental frequency line
                pygame.draw.line(self.screen, C_DIRECTION,
                                 (px, plot_y), (px, plot_y + plot_h), 2)
                pitch_txt = self.fmicro.render(
                    "F0: %.0f Hz" % self.pitch_hz, True, C_DIRECTION)
                self.screen.blit(pitch_txt, (px + 4, plot_y + 2))
                # Harmonic lines (faint)
                for h in range(2, 5):
                    h_freq = self.pitch_hz * h
                    h_bin = int(h_freq / (SAMPLE_RATE / 2) * n_bins)
                    if 0 <= h_bin < n_bins:
                        hx = bar_area.x + 4 + int(h_bin * bar_w)
                        pygame.draw.line(self.screen, C_DIRECTION,
                                         (hx, plot_y), (hx, plot_y + plot_h), 1)

        # --- Spectrogram waterfall ---
        spec_area = pygame.Rect(rect.x + 8, rect.y + rect.h // 2,
                                rect.w - 16, rect.h // 2 - 8)
        panel(self.screen, spec_area.x, spec_area.y,
              spec_area.w, spec_area.h, alpha=180)
        self.screen.blit(self.fs.render(
            "SPECTROGRAM WATERFALL  (time scrolls left -> right)",
            True, C_TEXT_DIM), (spec_area.x + 10, spec_area.y + 6))

        wf_x = spec_area.x + 10
        wf_y = spec_area.y + 24
        wf_w = spec_area.w - 20
        wf_h = spec_area.h - 34
        n_hist = len(self.spectrogram_history)
        if n_hist > 0 and self.has_real_audio:
            n_freq = min(128, FFT_SIZE // 2)
            # Build a 2D numpy array (n_freq x n_hist) of normalized values
            hist_list = list(self.spectrogram_history)
            cols = []
            for col in hist_list:
                vals = col[:n_freq]
                mx = np.max(vals) + 1e-6
                # Log-scale intensity for better dynamic range
                vals = np.log1p(vals / mx * 100) / np.log(101)
                vals = np.clip(vals, 0, 1)
                cols.append(vals)
            data = np.array(cols).T  # shape: (n_freq, n_hist)
            # Flip vertically so low freqs are at bottom
            data = data[::-1]
            # Map to colors using the spectrum palette
            # C_SPECTRUM is a list of RGB tuples -- build a lookup table
            palette = np.array(C_SPECTRUM, dtype=np.uint8)  # (N, 3)
            n_colors = len(palette)
            indices = (data * (n_colors - 1)).astype(np.int32)
            rgb = palette[indices]  # (n_freq, n_hist, 3)
            # Create a pygame surface from the numpy array
            surf = pygame.surfarray.make_surface(
                np.transpose(rgb, (1, 0, 2)))  # (width, height, 3)
            # Scale to fit the waterfall area
            surf = pygame.transform.scale(surf, (wf_w, wf_h))
            self.screen.blit(surf, (wf_x, wf_y))
            # Frequency axis labels on left side of spectrogram
            n_freq_disp = min(128, FFT_SIZE // 2)
            for f_label in [500, 1000, 2000, 4000, 8000]:
                freq_bin = int(f_label / (SAMPLE_RATE / 2) * n_freq_disp)
                if freq_bin < n_freq_disp:
                    # Y position: high freqs at top, low at bottom (flipped)
                    y_frac = 1.0 - freq_bin / n_freq_disp
                    ly = wf_y + int(y_frac * wf_h)
                    pygame.draw.line(self.screen, C_TEXT_DIM,
                                     (wf_x - 4, ly), (wf_x, ly), 1)
                    img = self.fmicro.render(
                        "%dk" % (f_label // 1000), True, C_TEXT_DIM)
                    self.screen.blit(img, (wf_x - 24, ly - 5))

            # Spectrogram color bar legend (right side)
            cb_x = wf_x + wf_w + 6
            cb_y = wf_y
            cb_w = 8
            cb_h = wf_h
            palette = np.array(C_SPECTRUM, dtype=np.uint8)
            n_colors = len(palette)
            # Vectorized: build color bar with numpy
            fracs = 1.0 - np.arange(cb_h) / max(cb_h - 1, 1)
            indices = (fracs * (n_colors - 1)).astype(np.int32)
            cb_pixels = np.tile(palette[indices][:, np.newaxis, :], (1, cb_w, 1))
            cb_pixels = np.transpose(cb_pixels, (1, 0, 2))  # (cb_w, cb_h, 3)
            cb_surf = pygame.surfarray.make_surface(cb_pixels)
            self.screen.blit(cb_surf, (cb_x, cb_y))
            pygame.draw.rect(self.screen, C_PANEL_HI,
                             (cb_x, cb_y, cb_w, cb_h), 1)
            # Labels
            hi_lbl = self.fmicro.render("hi", True, C_TEXT_DIM)
            lo_lbl = self.fmicro.render("lo", True, C_TEXT_DIM)
            self.screen.blit(hi_lbl, (cb_x + cb_w + 2, cb_y - 2))
            self.screen.blit(lo_lbl, (cb_x + cb_w + 2, cb_y + cb_h - 10))

        # --- Waveform overlay (raw vs enhanced) ---
        wf_rect = pygame.Rect(rect.x + 8, rect.y + rect.h - 80,
                              rect.w - 16, 70)
        panel(self.screen, wf_rect.x, wf_rect.y,
              wf_rect.w, wf_rect.h, alpha=200)
        self.screen.blit(self.fmicro.render(
            "WAVEFORM  (raw=grey  enhanced=green)", True, C_TEXT_DIM),
            (wf_rect.x + 8, wf_rect.y + 4))
        mid_y = wf_rect.y + 38
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (wf_rect.x + 4, mid_y),
                         (wf_rect.x + wf_rect.w - 4, mid_y), 1)
        n = min(len(self.raw_chunk), len(self.enhanced_chunk))
        if n > 1:
            step = max(1, n // (wf_rect.w - 8))
            indices = np.arange(0, n - step, step)
            xs = wf_rect.x + 4 + (indices * (wf_rect.w - 8) // n).astype(int)
            raw_ys = mid_y - (self.raw_chunk[indices] * 28).astype(int)
            clean_ys = mid_y - (self.enhanced_chunk[indices] * 28).astype(int)
            raw_pts = list(zip(xs.tolist(), raw_ys.tolist()))
            clean_pts = list(zip(xs.tolist(), clean_ys.tolist()))
            if len(raw_pts) > 1:
                pygame.draw.lines(self.screen, C_RAW, False, raw_pts, 1)
                pygame.draw.lines(self.screen, C_CLEAN, False, clean_pts, 2)

    # ---- DIRECTION MODE --------------------------------------------------

    def draw_direction(self):
        rect = self.view_rect()
        self.screen.blit(self.fb.render(
            "SPECTRAL ENERGY MAP  (single-mic -- not true direction)", True, C_ACCENT),
            (rect.x + 8, rect.y + 4))

        # --- Compass / radar display ---
        cx = rect.x + rect.w // 2
        cy = rect.y + rect.h // 2 + 20
        radius = min(rect.w, rect.h) // 2 - 60
        panel(self.screen, cx - radius - 30, cy - radius - 30,
              (radius + 30) * 2, (radius + 30) * 2, alpha=160)

        # Radar rings with distance labels
        for r_frac, dist_label in [(0.33, 'near'), (0.66, 'mid'), (1.0, 'far')]:
            r = int(radius * r_frac)
            pygame.draw.circle(self.screen, C_PANEL_HI, (cx, cy), r, 1)
            # Distance label on the ring
            if self.has_real_audio and self.distance_est < 999:
                ring_dist = self.distance_est * (1.0 - r_frac)
                lbl = self.fmicro.render("%dm" % int(ring_dist), True, C_TEXT_DIM)
                self.screen.blit(lbl, (cx + 4, cy - r + 2))

        # Cross hairs
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (cx - radius, cy), (cx + radius, cy), 1)
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (cx, cy - radius), (cx, cy + radius), 1)

        # Direction labels
        for angle_deg, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
            a = math.radians(angle_deg - 90)
            lx = cx + int((radius + 16) * math.cos(a))
            ly = cy + int((radius + 16) * math.sin(a))
            img = self.fs.render(label, True, C_TEXT_DIM)
            self.screen.blit(img, (lx - img.get_width() // 2,
                                   ly - img.get_height() // 2))

        # Directional intensity rose
        bins = self.direction_bins
        n = len(bins)
        for i in range(n):
            angle = i * 2 * math.pi / n - math.pi / 2
            intensity = bins[i]
            r = int(radius * intensity * 0.9)
            if r > 2:
                x1 = cx + int(r * math.cos(angle))
                y1 = cy + int(r * math.sin(angle))
                col = spectrum_color(intensity)
                pygame.draw.line(self.screen, col, (cx, cy), (x1, y1), 2)
                pygame.draw.circle(self.screen, col, (x1, y1), 3)

        # Filled polygon for the directional rose
        pts = []
        for i in range(n):
            angle = i * 2 * math.pi / n - math.pi / 2
            r = int(radius * bins[i] * 0.9)
            pts.append((cx + int(r * math.cos(angle)),
                        cy + int(r * math.sin(angle))))
        if len(pts) > 2:
            poly_size = (radius + 30) * 2
            fill_surf = pygame.Surface((poly_size, poly_size), pygame.SRCALPHA)
            offset_pts = [(p[0] - cx + poly_size // 2, p[1] - cy + poly_size // 2) for p in pts]
            pygame.draw.polygon(fill_surf, (255, 180, 60, 40), offset_pts)
            self.screen.blit(fill_surf, (cx - poly_size // 2, cy - poly_size // 2))
            pygame.draw.polygon(self.screen, C_DIRECTION, pts, 1)

        # Sweep line
        sweep_a = self.sweep_angle - math.pi / 2
        sx = cx + int(radius * math.cos(sweep_a))
        sy = cy + int(radius * math.sin(sweep_a))
        pygame.draw.line(self.screen, C_ACCENT, (cx, cy), (sx, sy), 2)
        # Sweep trail
        for t in range(10):
            a = self.sweep_angle - t * 0.08 - math.pi / 2
            tx = cx + int(radius * math.cos(a))
            ty = cy + int(radius * math.sin(a))
            alpha = int(200 * (1 - t / 10))
            col_s = pygame.Surface((4, 4), pygame.SRCALPHA)
            col_s.fill((90, 200, 255, alpha))
            self.screen.blit(col_s, (tx - 2, ty - 2))

        # Center dot with confidence indicator
        pygame.draw.circle(self.screen, C_ACCENT, (cx, cy), 5)
        pygame.draw.circle(self.screen, C_TEXT, (cx, cy), 5, 1)
        # Confidence ring around center
        if self.has_real_audio:
            conf_r = int(8 + self.confidence_score * 12)
            conf_col = C_GOOD if self.confidence_score > 0.5 else (C_WARN if self.confidence_score > 0.3 else C_BAD)
            pygame.draw.circle(self.screen, conf_col, (cx, cy), conf_r, 1)

        # Focus lock indicator
        if self.focus_lock_enabled:
            lock_a = math.radians(self.focus_angle - 90)
            lx = cx + int((radius + 20) * math.cos(lock_a))
            ly = cy + int((radius + 20) * math.sin(lock_a))
            # Draw lock marker
            pygame.draw.circle(self.screen, C_GOOD, (lx, ly), 10, 2)
            pygame.draw.line(self.screen, C_GOOD, (lx - 14, ly), (lx + 14, ly), 1)
            pygame.draw.line(self.screen, C_GOOD, (lx, ly - 14), (lx, ly + 14), 1)
            lock_txt = self.fsmall.render("LOCK", True, C_GOOD)
            self.screen.blit(lock_txt, (lx - lock_txt.get_width() // 2, ly + 12))

        # --- Direction info panel ---
        info_x = rect.x + 8
        info_y = rect.y + 36
        panel(self.screen, info_x, info_y, 260, 180, alpha=200)
        self.screen.blit(self.fs.render(
            "SPECTRAL ENERGY ANALYSIS", True, C_ACCENT),
            (info_x + 10, info_y + 8))
        # Find peak direction
        peak_dir = int(np.argmax(bins))
        peak_angle = peak_dir * 360 / n
        peak_intensity = bins[peak_dir]
        if self.has_real_audio:
            rows = [
                ("Peak freq bin", "%d deg (%s)" % (
                    peak_angle, self._compass_dir(peak_angle))),
                ("Peak intensity", "%.2f" % peak_intensity),
                ("Confidence", "%.0f%%" % (self.confidence_score * 100)),
                ("Pitch", "%.0f Hz" % self.pitch_hz if self.pitch_hz > 0 else "----"),
                ("Sweep angle", "%d deg" % ((self.sweep_angle * 180 / math.pi) % 360)),
                ("Voice detected", "YES" if self.voice_detected else "no"),
                ("Focus lock", "ON  %d deg" % self.focus_angle if self.focus_lock_enabled else "OFF"),
            ]
        else:
            rows = [
                ("Peak freq bin", "----"),
                ("Peak intensity", "----"),
                ("Confidence", "----"),
                ("Pitch", "----"),
                ("Sweep angle", "%d deg" % ((self.sweep_angle * 180 / math.pi) % 360)),
                ("Voice detected", "NO INPUT"),
                ("Focus lock", "ON  %d deg" % self.focus_angle if self.focus_lock_enabled else "OFF"),
            ]
        yy = info_y + 30
        for lab, val in rows:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (info_x + 12, yy))
            img = self.fs.render(val, True, C_TEXT)
            self.screen.blit(img, (info_x + 250 - img.get_width() - 4, yy))
            yy += 20

        # --- Distance gauge ---
        dist_x = rect.right - 268
        dist_y = rect.y + 36
        panel(self.screen, dist_x, dist_y, 260, 180, alpha=200)
        self.screen.blit(self.fs.render(
            "DISTANCE ESTIMATION", True, C_ACCENT),
            (dist_x + 10, dist_y + 8))
        if self.has_real_audio:
            dist_txt = "%.0f m" % self.distance_est if self.distance_est < 999 \
                else "----"
            dist_col = C_GOOD if self.distance_est < 200 else C_WARN
        else:
            dist_txt = "----"
            dist_col = C_TEXT_DIM
        self.screen.blit(self.fbig.render(dist_txt, True, dist_col),
            (dist_x + 14, dist_y + 30))
        self.screen.blit(self.fs.render("ESTIMATED RANGE", True, C_TEXT_DIM),
                         (dist_x + 14, dist_y + 64))
        if self.has_real_audio:
            rows2 = [
                ("Signal level", "%.1f dB" % self.db_level),
                ("SNR", "%.1f dB" % self.snr_est),
                ("A-Weighted", "%.1f dB(A)" % self.aweighted_db),
                ("Peak output", "%.3f" % self.peak_level),
                ("Voice band", "%d-%d Hz" % (self.voice_low, self.voice_high)),
            ]
        else:
            rows2 = [
                ("Signal level", "----"),
                ("SNR", "----"),
                ("A-Weighted", "----"),
                ("Peak output", "----"),
                ("Voice band", "%d-%d Hz" % (self.voice_low, self.voice_high)),
            ]
        yy = dist_y + 86
        for lab, val in rows2:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (dist_x + 12, yy))
            img = self.fs.render(val, True, C_TEXT)
            self.screen.blit(img, (dist_x + 250 - img.get_width() - 4, yy))
            yy += 20

        # --- Direction history strip ---
        hist_y = rect.bottom - 80
        panel(self.screen, rect.x + 8, hist_y, rect.w - 16, 70, alpha=200)
        self.screen.blit(self.fmicro.render(
            "ENERGY HISTORY  (recent spectral distributions)", True, C_TEXT_DIM),
            (rect.x + 16, hist_y + 4))
        n_hist = len(self.direction_history)
        if n_hist > 1:
            hist_w = rect.w - 32
            col_w = max(1, hist_w // n_hist)
            for hi, hist_bins in enumerate(list(self.direction_history)):
                px = rect.x + 16 + hi * col_w
                for bi in range(n):
                    val = hist_bins[bi]
                    h = int(val * 50)
                    color = spectrum_color(val)
                    by = hist_y + 60 - h
                    pygame.draw.rect(self.screen, color,
                                     (px, by, col_w, max(1, h)))

    def _compass_dir(self, angle):
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((angle + 11.25) / 22.5) % 16
        return dirs[idx]

    # ---- AI FILTER MODE --------------------------------------------------

    def draw_filter(self):
        rect = self.view_rect()
        self.screen.blit(self.fb.render(
            "AUDIO PROCESSING PIPELINE", True, C_ACCENT),
            (rect.x + 8, rect.y + 4))

        stages = [
            ("1. RAW INPUT", self.raw_chunk, C_RAW,
             "Mic capture  ->  float32 PCM", 'raw'),
            ("2. WIND FILTER", self.wind_filtered_chunk, C_COOL,
             "Adaptive HPF  enabled=%s" % str(self.wind_filter_enabled), 'filtered'),
            ("3. BANDPASS", self.filtered_chunk, C_VOICE,
             "Butter 8th-order  %d-%d Hz" % (self.voice_low, self.voice_high), 'gated'),
            ("4. SPECTRAL GATE", self.gated_chunk, C_NOISE,
             "Soft tanh gate  str=%.2f" % self.noise_reduction, 'dereverbed'),
            ("5. DE-REVERB", self.dereverbed_chunk, C_COOL,
             "Spectral smooth  str=%.2f" % self.dereverb_strength, 'decohered'),
            ("6. DECOHERENCE", self.decohered_chunk, C_PROC,
             "Adaptive noise floor  str=%.2f" % self.noise_reduction, 'enhanced'),
            ("7. WIENER", self.enhanced_chunk, C_CLEAN,
             "Spectral enhance  gain=%.2fx" % self.enhance_strength, 'compressed'),
            ("8. COMPRESSOR", self.compressed_chunk, C_HOT,
             "3-band comp  GR: %.0f/%.0f/%.0f dB" % (self.gr_low, self.gr_mid, self.gr_high), 'presence'),
            ("9. PRESENCE", self.presence_chunk, C_VOICE,
             "Consonant boost %.0f dB" % PRESENCE_BOOST_DB, 'formant'),
            ("10. FORMANT", self.formant_chunk, C_ACCENT,
             "F1/F2 vowel enhance  enabled=%s" % str(self.formant_enabled), 'superres'),
            ("11. SUPER-RES + NORM", self.enhanced_chunk, C_GOOD,
             "Harmonic recon + loudness norm  ->  output", 'loudness'),
        ]

        n_stages = len(stages)
        stage_h = (rect.h - 50) // n_stages
        for i, (title, data, color, desc, spec_key) in enumerate(stages):
            sy = rect.y + 36 + i * stage_h
            panel(self.screen, rect.x + 8, sy, rect.w - 16, stage_h - 6,
                  alpha=180)
            self.screen.blit(self.fs.render(title, True, color),
                             (rect.x + 16, sy + 6))
            self.screen.blit(self.fmicro.render(desc, True, C_TEXT_DIM),
                             (rect.x + 16, sy + 22))
            # RMS level for this stage
            if len(data) > 0:
                rms = float(np.sqrt(np.mean(data ** 2)))
                rms_db = 20 * math.log10(rms + 1e-10)
                rms_txt = self.fmicro.render(
                    "RMS: %.1f dB" % rms_db, True, C_TEXT_DIM)
                self.screen.blit(rms_txt, (rect.x + 280, sy + 6))
                # Inter-stage gain indicator
                if i > 0:
                    prev_data = stages[i - 1][1]
                    prev_rms = float(np.sqrt(np.mean(prev_data ** 2))) + 1e-10
                    gain_db = 20 * math.log10(rms / prev_rms + 1e-10)
                    if abs(gain_db) > 0.5:
                        gain_col = C_GOOD if gain_db > 0 else C_BAD
                        gain_txt = self.fmicro.render(
                            "%+.1f dB" % gain_db, True, gain_col)
                        self.screen.blit(gain_txt, (rect.x + 380, sy + 6))

            # Waveform
            wf_y = sy + 38
            wf_h = stage_h - 48
            mid_y = wf_y + wf_h // 2
            pygame.draw.line(self.screen, C_PANEL_HI,
                             (rect.x + 12, mid_y),
                             (rect.x + rect.w - 12, mid_y), 1)
            n = len(data)
            if n > 1:
                step = max(1, n // (rect.w - 24))
                indices = np.arange(0, n - step, step)
                xs = rect.x + 12 + (indices * (rect.w - 24) // n).astype(int)
                ys = mid_y - (data[indices] * (wf_h // 2 - 4)).astype(int)
                pts = list(zip(xs.tolist(), ys.tolist()))
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, color, False, pts, 2)

            # Mini spectrum (cached) - vectorized drawing
            spec = self._mini_specs.get(spec_key, np.zeros(64))
            n_bins = 64
            spec_x = rect.x + rect.w - 180
            spec_w = 160
            spec_h = wf_h - 4
            mx = max(np.max(spec[:n_bins]), 1e-6)
            spec_norm = np.clip(spec[:n_bins] / mx, 0, 1)
            bar_w = max(1, int(spec_w / n_bins) - 1)
            heights = (spec_norm * spec_h).astype(np.int32)
            pix = np.full((n_bins * bar_w, spec_h, 3), C_PANEL, dtype=np.uint8)
            for bi in range(n_bins):
                h = heights[bi]
                if h > 0:
                    pix[bi * bar_w:(bi + 1) * bar_w, spec_h - h:spec_h] = color
            spec_surf = pygame.surfarray.make_surface(pix)
            self.screen.blit(spec_surf, (spec_x, wf_y))

            # Arrow between stages
            if i < n_stages - 1:
                ay = sy + stage_h - 6
                ax = rect.x + rect.w // 2
                pygame.draw.polygon(self.screen, C_ACCENT,
                    [(ax - 6, ay), (ax + 6, ay), (ax, ay + 5)])

    # ---- METER MODE -----------------------------------------------------

    def draw_meter(self):
        rect = self.view_rect()
        self.screen.blit(self.fb.render(
            "PROFESSIONAL AUDIO METERING", True, C_ACCENT),
            (rect.x + 8, rect.y + 4))

        # --- Main VU/PPM meters (input + output, left side) ---
        meter_x = rect.x + 12
        meter_y = rect.y + 40
        meter_w = 50
        meter_h = rect.h - 120

        # Input meter (raw signal)
        panel(self.screen, meter_x - 4, meter_y - 4,
              meter_w + 8, meter_h + 8, alpha=180)
        self.screen.blit(self.fs.render("INPUT", True, C_TEXT_DIM),
                         (meter_x, meter_y - 20))
        for db_mark in range(-60, 1, 6):
            frac = (db_mark + 60) / 60.0
            my = meter_y + meter_h - int(frac * meter_h)
            col = C_GOOD if db_mark < -18 else (C_WARN if db_mark < -6 else C_BAD)
            pygame.draw.line(self.screen, col,
                             (meter_x, my), (meter_x + 6, my), 1)
        if self.has_real_audio:
            raw_rms = float(np.sqrt(np.mean(self.raw_chunk ** 2)))
            raw_db = max(20 * math.log10(raw_rms + 1e-10), -60)
            frac = (raw_db + 60) / 60.0
            bar_h = int(frac * meter_h)
            bar_w = meter_w - 16
            green_h = int(0.6 * meter_h)
            yellow_h = int(0.25 * meter_h)
            red_h = meter_h - green_h - yellow_h
            pygame.draw.rect(self.screen, C_GOOD,
                             (meter_x, meter_y + meter_h - green_h, bar_w, green_h))
            pygame.draw.rect(self.screen, C_WARN,
                             (meter_x, meter_y + meter_h - green_h - yellow_h, bar_w, yellow_h))
            pygame.draw.rect(self.screen, C_BAD,
                             (meter_x, meter_y, bar_w, red_h))
            if bar_h < meter_h:
                pygame.draw.rect(self.screen, C_PANEL,
                                 (meter_x, meter_y, bar_w, meter_h - bar_h))
            # Input peak hold marker
            if raw_db > self._vu_peak_hold_in:
                self._vu_peak_hold_in = raw_db
                self._vu_peak_hold_in_frames = 0
            else:
                self._vu_peak_hold_in_frames += 1
                if self._vu_peak_hold_in_frames > 30:
                    self._vu_peak_hold_in -= 0.5
            in_peak_frac = clamp((self._vu_peak_hold_in + 60) / 60.0)
            in_peak_y = meter_y + meter_h - int(in_peak_frac * meter_h)
            pygame.draw.line(self.screen, (255, 255, 255),
                             (meter_x, in_peak_y), (meter_x + bar_w, in_peak_y), 2)

        # Output meter (enhanced signal)
        meter_x2 = meter_x + meter_w + 8
        panel(self.screen, meter_x2 - 4, meter_y - 4,
              meter_w + 8, meter_h + 8, alpha=180)
        self.screen.blit(self.fs.render("OUTPUT", True, C_TEXT_DIM),
                         (meter_x2, meter_y - 20))

        # Scale markings
        for db_mark in range(-60, 1, 6):
            frac = (db_mark + 60) / 60.0
            my = meter_y + meter_h - int(frac * meter_h)
            col = C_GOOD if db_mark < -18 else (C_WARN if db_mark < -6 else C_BAD)
            pygame.draw.line(self.screen, col,
                             (meter_x2, my), (meter_x2 + 6, my), 1)
            label = self.fmicro.render("%d" % db_mark, True, C_TEXT_DIM)
            self.screen.blit(label, (meter_x2 + 8, my - 5))

        # Meter bar (optimized: draw 3 color zones + fill)
        if self.has_real_audio:
            db_val = max(self.db_level, -60)
            frac = (db_val + 60) / 60.0
            bar_h = int(frac * meter_h)
            bar_w = meter_w - 20
            # Draw 3 color zones as filled rects
            green_h = int(0.6 * meter_h)
            yellow_h = int(0.25 * meter_h)
            red_h = meter_h - green_h - yellow_h
            # Green zone (bottom 60%)
            pygame.draw.rect(self.screen, C_GOOD,
                             (meter_x2, meter_y + meter_h - green_h, bar_w, green_h))
            # Yellow zone (middle 25%)
            pygame.draw.rect(self.screen, C_WARN,
                             (meter_x2, meter_y + meter_h - green_h - yellow_h, bar_w, yellow_h))
            # Red zone (top 15%)
            pygame.draw.rect(self.screen, C_BAD,
                             (meter_x2, meter_y, bar_w, red_h))
            # Overlay dark mask above current level
            if bar_h < meter_h:
                pygame.draw.rect(self.screen, C_PANEL,
                                 (meter_x2, meter_y, bar_w, meter_h - bar_h))
            # Peak marker with slow decay
            peak_db = max(self.db_level, -60)
            if peak_db > self._vu_peak_hold:
                self._vu_peak_hold = peak_db
                self._vu_peak_hold_frames = 0
            else:
                self._vu_peak_hold_frames += 1
                if self._vu_peak_hold_frames > 30:
                    self._vu_peak_hold -= 0.5
            peak_frac = clamp((self._vu_peak_hold + 60) / 60.0)
            peak_y = meter_y + meter_h - int(peak_frac * meter_h)
            pygame.draw.line(self.screen, (255, 255, 255),
                             (meter_x2, peak_y), (meter_x2 + bar_w, peak_y), 2)
            # Total gain reduction marker (sum of 3 bands)
            total_gr = self.gr_low + self.gr_mid + self.gr_high
            if abs(total_gr) > 0.1:
                gr_db = max(total_gr, -60)
                gr_frac = clamp((gr_db + 60) / 60.0)
                gr_y = meter_y + meter_h - int(gr_frac * meter_h)
                pygame.draw.line(self.screen, C_HOT,
                                 (meter_x2, gr_y), (meter_x2 + bar_w, gr_y), 1)
                gr_lbl = self.fmicro.render("GR", True, C_HOT)
                self.screen.blit(gr_lbl, (meter_x2 + bar_w + 2, gr_y - 5))

        # --- Gain reduction meters (3 bands) ---
        gr_x = meter_x2 + meter_w + 16
        gr_w = 50
        gr_labels = ["LOW", "MID", "HIGH"]
        gr_vals = [self.gr_low, self.gr_mid, self.gr_high]
        gr_colors = [C_COOL, C_VOICE, C_HOT]
        panel(self.screen, gr_x - 4, meter_y - 4,
              gr_w * 3 + 16, meter_h + 8, alpha=180)
        self.screen.blit(self.fs.render("GAIN REDUCTION", True, C_TEXT_DIM),
                         (gr_x, meter_y - 20))
        for bi in range(3):
            bx = gr_x + bi * (gr_w + 4)
            pygame.draw.rect(self.screen, C_PANEL_HI,
                             (bx, meter_y, gr_w, meter_h), 1)
            # Label
            self.screen.blit(self.fmicro.render(gr_labels[bi], True, gr_colors[bi]),
                             (bx + 4, meter_y + meter_h + 4))
            if self.has_real_audio and self.compress_enabled:
                gr = gr_vals[bi]
                gr_frac = clamp(-gr / 20.0)  # 0 to -20 dB
                gr_bar_h = int(gr_frac * meter_h)
                pygame.draw.rect(self.screen, gr_colors[bi],
                                 (bx, meter_y + meter_h - gr_bar_h, gr_w, gr_bar_h))
                gr_txt = self.fmicro.render("%.1f" % gr, True, C_TEXT)
                self.screen.blit(gr_txt, (bx + 4, meter_y + meter_h + 18))

        # --- Quality metrics panel (right side) ---
        qm_x = gr_x + gr_w * 3 + 24
        qm_w = rect.right - qm_x - 8
        qm_h = meter_h + 8
        panel(self.screen, qm_x - 4, meter_y - 4, qm_w, qm_h, alpha=180)
        self.screen.blit(self.fs.render("QUALITY METRICS", True, C_ACCENT),
                         (qm_x + 8, meter_y + 4))
        # Headroom indicator
        if self.has_real_audio:
            headroom_db = max(0, -self.db_level)
            hr_col = C_GOOD if headroom_db > 6 else (C_WARN if headroom_db > 3 else C_BAD)
            hr_txt = self.fmicro.render(
                "Headroom: %.1f dB" % headroom_db, True, hr_col)
            self.screen.blit(hr_txt, (qm_x + qm_w - hr_txt.get_width() - 12, meter_y + 4))

        if self.has_real_audio:
            metrics = [
                ("A-Weighted", "%.1f dB(A)" % self.aweighted_db, C_ACCENT),
                ("Centroid", "%.0f Hz" % self.centroid_hz, C_VOICE),
                ("Flatness", "%.3f" % self.flatness,
                 C_GOOD if self.flatness < 0.4 else C_WARN),
                ("SII Score", "%.2f" % self.sii_score,
                 C_GOOD if self.sii_score > 0.5 else (C_WARN if self.sii_score > 0.3 else C_BAD)),
                ("VAD Conf.", "%.2f" % self.vad_confidence,
                 C_GOOD if self.vad_confidence > 0.5 else C_TEXT_DIM),
                ("Wind Noise", "YES %.0f%%" % (self.wind_strength * 100) if self.wind_detected else "no",
                 C_WARN if self.wind_detected else C_GOOD),
                ("Clipping", "YES" if self.clipping else "no",
                 C_BAD if self.clipping else C_GOOD),
                ("Signal", "%.1f dB" % self.db_level, C_ACCENT),
                ("SNR", "%.1f dB" % self.snr_est,
                 C_GOOD if self.snr_est > 10 else (C_WARN if self.snr_est > 0 else C_BAD)),
                ("Peak", "%.3f" % self.peak_level, C_HOT),
                ("Distance", "%.0f m" % self.distance_est if self.distance_est < 999 else "----",
                 C_GOOD if self.distance_est < 200 else C_WARN),
                ("Voice", "DETECTED" if self.voice_detected else "silence",
                 C_GOOD if self.voice_detected else C_TEXT_DIM),
                ("Confidence", "%.0f%%" % (self.confidence_score * 100),
                 C_GOOD if self.confidence_score > 0.5 else (C_WARN if self.confidence_score > 0.3 else C_BAD)),
                ("Pitch", "%.0f Hz" % self.pitch_hz if self.pitch_hz > 0 else "----",
                 C_VOICE if self.pitch_hz > 0 else C_TEXT_DIM),
                ("Proc Latency", "%.1f ms" % self._proc_latency_ms,
                 C_GOOD if self._proc_latency_ms < 10 else (C_WARN if self._proc_latency_ms < 30 else C_BAD)),
            ]
        else:
            metrics = [
                ("A-Weighted", "----", C_TEXT_DIM),
                ("Centroid", "----", C_TEXT_DIM),
                ("Flatness", "----", C_TEXT_DIM),
                ("SII Score", "----", C_TEXT_DIM),
                ("VAD Conf.", "----", C_TEXT_DIM),
                ("Wind Noise", "----", C_TEXT_DIM),
                ("Clipping", "----", C_TEXT_DIM),
                ("Signal", "----", C_TEXT_DIM),
                ("SNR", "----", C_TEXT_DIM),
                ("Peak", "----", C_TEXT_DIM),
                ("Distance", "----", C_TEXT_DIM),
                ("Voice", "NO INPUT", C_TEXT_DIM),
                ("Confidence", "----", C_TEXT_DIM),
                ("Pitch", "----", C_TEXT_DIM),
                ("Proc Latency", "----", C_TEXT_DIM),
            ]
        yy = meter_y + 30
        for lab, val, col in metrics:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (qm_x + 12, yy))
            img = self.fs.render(val, True, col)
            self.screen.blit(img, (qm_x + qm_w - img.get_width() - 16, yy))
            yy += 22

        # --- Confidence history graph ---
        ch_y = yy + 8
        ch_h = 50
        panel(self.screen, qm_x, ch_y, qm_w, ch_h, alpha=200)
        self.screen.blit(self.fs.render(
            "CONFIDENCE HISTORY", True, C_ACCENT), (qm_x + 12, ch_y + 6))
        graph_x = qm_x + 12
        graph_y = ch_y + 24
        graph_w = qm_w - 24
        graph_h2 = ch_h - 32
        pygame.draw.rect(self.screen, C_PANEL_HI,
                         (graph_x, graph_y, graph_w, graph_h2), 1)
        if len(self.confidence_history) > 1 and self.has_real_audio:
            conf_vals = list(self.confidence_history)
            n = len(conf_vals)
            for i in range(n - 1):
                px1 = graph_x + int(i * graph_w / max(n - 1, 1))
                px2 = graph_x + int((i + 1) * graph_w / max(n - 1, 1))
                py1 = graph_y + graph_h2 - int(conf_vals[i] * graph_h2)
                py2 = graph_y + graph_h2 - int(conf_vals[i + 1] * graph_h2)
                col = C_GOOD if conf_vals[i + 1] > 0.5 else (C_WARN if conf_vals[i + 1] > 0.3 else C_BAD)
                pygame.draw.line(self.screen, col, (px1, py1), (px2, py2), 1)
            # 50% threshold line
            mid_y = graph_y + graph_h2 - int(0.5 * graph_h2)
            for dx in range(0, graph_w, 4):
                pygame.draw.line(self.screen, C_TEXT_DIM,
                                 (graph_x + dx, mid_y),
                                 (graph_x + dx + 2, mid_y), 1)
        else:
            idle = self.fmicro.render("no data", True, C_TEXT_DIM)
            self.screen.blit(idle, (graph_x + 8, graph_y + graph_h2 // 2 - 6))

        # --- Feature toggle status bar ---
        ft_y = rect.bottom - 70
        panel(self.screen, rect.x + 8, ft_y, rect.w - 16, 60, alpha=200)
        self.screen.blit(self.fs.render("PROCESSING FEATURES", True, C_ACCENT),
                         (rect.x + 16, ft_y + 6))
        features = [
            ("Gate", self.gate_enabled, C_NOISE),
            ("DeReverb", self.dereverb_enabled, C_COOL),
            ("Compress", self.compress_enabled, C_HOT),
            ("Presence", self.presence_enabled, C_VOICE),
            ("Formant", self.formant_enabled, C_ACCENT),
            ("SuperRes", self.superres_enabled, C_CLEAN),
            ("Loudness", self.loudness_enabled, C_GOOD),
            ("Wind HPF", self.wind_filter_enabled, C_COOL),
            ("AutoAdapt", self.auto_adapt_enabled, C_WARN),
            ("DeepLsn", self.deep_listen, C_GOOD),
            ("VAD-Rec", self.vad_record, C_WARN),
            ("Whisper", self.whisper_mode, C_ACCENT),
            ("NoiseProf", self.noise_profile is not None, C_COOL),
            ("FreqIso", self.freq_isolator_enabled, C_VOICE),
            ("AGC", self.agc_enabled, C_GOOD),
            ("EQ", self.eq_bass != 0 or self.eq_mid != 0 or self.eq_treble != 0, C_ACCENT),
            ("A/B", self.ab_compare, C_WARN),
        ]
        fx = rect.x + 16
        for name, enabled, col in features:
            status = "ON" if enabled else "OFF"
            scol = col if enabled else C_TEXT_DIM
            pygame.draw.circle(self.screen, scol, (fx + 6, ft_y + 34), 5)
            txt = self.fs.render("%s: %s" % (name, status), True, scol)
            self.screen.blit(txt, (fx + 16, ft_y + 28))
            fx += txt.get_width() + 40

        # Preset indicator
        if self.current_preset:
            ptxt = self.fs.render("Preset: %s" % self.current_preset, True, C_ACCENT)
            self.screen.blit(ptxt, (rect.right - ptxt.get_width() - 16, ft_y + 28))

    # ---- BLUEPRINT MODE (3D Hardware Model) ------------------------------

    def draw_blueprint(self):
        rect = self.view_rect()

        # Lazy-init the 3D renderer
        if self._bp_renderer is None:
            parts = build_hearnocular_model()
            self._bp_renderer = BlueprintRenderer(parts, az=0.5, el=0.3)

        self.screen.blit(self.fb.render(
            "HARDWARE BLUEPRINT -- 3D DEVICE MODEL", True, C_ACCENT),
            (rect.x + 8, rect.y + 4))
        # Mode status indicators
        mode_indicators = []
        if self._bp_renderer:
            if self._bp_renderer.auto_rotate:
                mode_indicators.append(("AUTO-ROTATE", C_ACCENT))
            if self._bp_renderer.wireframe:
                mode_indicators.append(("WIREFRAME", C_COOL))
            if self._bp_renderer.exploded > 0.5:
                mode_indicators.append(("EXPLODED", C_WARN))
            if self._bp_renderer.highlight_part >= 0:
                pname = self._bp_renderer.parts[self._bp_renderer.highlight_part].name
                mode_indicators.append(("FOCUS: %s" % pname, C_GOOD))
        mx = rect.x + 8
        for label, col in mode_indicators:
            txt = self.fs.render(label, True, col)
            self.screen.blit(txt, (mx, rect.y + 40))
            mx += txt.get_width() + 12
        self.screen.blit(self.fs.render(
            "Drag=orbit  Wheel=zoom  Shift+Drag=pan  R=reset  L=labels  "
            "E=exploded  W=wireframe  A=auto-rotate  F=focus  T=edges",
            True, C_TEXT_DIM),
            (rect.x + 8, rect.y + 24))

        # Render the 3D model
        self._bp_renderer.render(self.screen, rect,
                                 show_labels=self._bp_show_labels,
                                 font=self.fs,
                                 font_small=self.fmicro)

        # --- Component info panel (right side, scrollable) ---
        info_w = 230
        info_h = min(rect.h - 60, 420)
        info_x = rect.right - info_w - 8
        info_y = rect.y + 42
        panel(self.screen, info_x, info_y, info_w, info_h, alpha=225)
        self.screen.blit(self.fs.render("COMPONENTS (%d parts)" % len(
            self._bp_renderer.parts), True, C_ACCENT),
            (info_x + 10, info_y + 6))
        # Count total meshes
        total_meshes = sum(len(p.meshes) for p in self._bp_renderer.parts)
        self.screen.blit(self.fmicro.render(
            "%d mesh groups" % total_meshes, True, C_TEXT_DIM),
            (info_x + 10, info_y + 20))
        # Scrollable component list
        clip = pygame.Rect(info_x + 8, info_y + 34, info_w - 16, info_h - 42)
        self.screen.set_clip(clip)
        yy = info_y + 36 - self.help_scroll
        for part in self._bp_renderer.parts:
            if part.specs:
                # Part name header
                if yy > info_y + 30 and yy < info_y + info_h:
                    self.screen.blit(self.fmicro.render(
                        part.name, True, C_TEXT),
                        (info_x + 10, yy))
                yy += 12
                for spec in part.specs:
                    if yy > info_y + 30 and yy < info_y + info_h:
                        self.screen.blit(self.fmicro.render(
                            "  " + spec, True, C_TEXT_DIM),
                            (info_x + 10, yy))
                    yy += 11
                yy += 4
        self.screen.set_clip(None)
        # Scrollbar for component list
        content_total = sum(1 + len(p.specs) for p in self._bp_renderer.parts if p.specs)
        content_total = content_total * 12 + sum(4 for p in self._bp_renderer.parts if p.specs)
        max_sc = max(0, content_total - (info_h - 42))
        if max_sc > 0:
            sb_x = info_x + info_w - 8
            sb_y = info_y + 34
            sb_h = info_h - 42
            pygame.draw.rect(self.screen, C_PANEL_HI, (sb_x, sb_y, 3, sb_h), 1)
            thumb_h = max(15, int(sb_h * sb_h / (sb_h + max_sc)))
            thumb_y = sb_y + int((sb_h - thumb_h) * min(self.help_scroll, max_sc) / max_sc)
            pygame.draw.rect(self.screen, C_ACCENT, (sb_x, thumb_y, 3, thumb_h))

        # --- Signal flow diagram (bottom-center) ---
        flow_x = rect.x + 220
        flow_y = rect.bottom - 90
        flow_w = rect.w - 440
        panel(self.screen, flow_x, flow_y, flow_w, 78, alpha=220)
        self.screen.blit(self.fs.render("SIGNAL FLOW", True, C_ACCENT),
                         (flow_x + 10, flow_y + 6))
        flow_stages = [
            ("Mics", C_RAW),
            ("Preamp", C_COOL),
            ("ADC", C_COOL),
            ("DSP", C_GOOD),
            ("Output", C_ACCENT),
        ]
        sx = flow_x + 16
        sy = flow_y + 36
        spacing = (flow_w - 32) // len(flow_stages)
        for i, (name, col) in enumerate(flow_stages):
            bx = sx + i * spacing
            # Box
            bw = spacing - 12
            pygame.draw.rect(self.screen, C_PANEL_HI, (bx, sy, bw, 22), 1, border_radius=4)
            self.screen.blit(self.fmicro.render(name, True, col),
                             (bx + 6, sy + 5))
            # Arrow to next
            if i < len(flow_stages) - 1:
                ax = bx + bw + 2
                pygame.draw.line(self.screen, C_TEXT_DIM,
                                 (ax, sy + 11), (ax + 8, sy + 11), 1)
                pygame.draw.polygon(self.screen, C_TEXT_DIM,
                                    [(ax + 8, sy + 11), (ax + 5, sy + 8), (ax + 5, sy + 14)])
        # Power line indicator
        self.screen.blit(self.fmicro.render(
            "Power: Battery -> BMS -> Preamp + DSP", True, C_WARN),
            (flow_x + 16, flow_y + 62))

        # --- Cost tiers (bottom-left overlay) ---
        cost_x = rect.x + 12
        cost_y = rect.bottom - 90
        panel(self.screen, cost_x, cost_y, 200, 78, alpha=220)
        self.screen.blit(self.fs.render("COST TIERS", True, C_ACCENT),
                         (cost_x + 10, cost_y + 6))
        costs = [
            ("Budget (Pi5+Primo)", "$350", C_GOOD),
            ("Mid-range (DPA+HFB)", "$800", C_WARN),
            ("Premium (Jetson+DPA)", "$1800", C_BAD),
        ]
        for i, (tier, price, col) in enumerate(costs):
            cy_pos = cost_y + 24 + i * 16
            self.screen.blit(self.fmicro.render(tier, True, C_TEXT_DIM),
                             (cost_x + 10, cy_pos))
            img = self.fmicro.render(price, True, col)
            self.screen.blit(img, (cost_x + 150, cy_pos))

        # --- Dimensions overlay (top-right, below components) ---
        dim_x = rect.right - info_w - 8
        dim_y = info_y + info_h + 6
        if dim_y + 80 < rect.bottom - 100:
            panel(self.screen, dim_x, dim_y, info_w, 76, alpha=220)
            self.screen.blit(self.fs.render("DIMENSIONS", True, C_ACCENT),
                             (dim_x + 10, dim_y + 6))
            dims = [
                "Dish: 22-24 in diameter",
                "Stowed: 24x12x6 in",
                "Weight: 10-15 lbs",
                "Runtime: 4-8 hours",
            ]
            for i, d in enumerate(dims):
                self.screen.blit(self.fmicro.render(
                    d, True, C_TEXT_DIM),
                    (dim_x + 10, dim_y + 22 + i * 12))

    # ---- LEFT PANEL (controls) -------------------------------------------

    def draw_left_panel(self):
        w = self.LEFT_PANEL_W - 16
        x = 8
        y = self.TOP_BAR_H + 4
        h = self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 12
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("CONTROLS", True, C_ACCENT),
                         (x + 12, y + 8))

        yy = y + 36
        self._slider_hitboxes = {}
        self._left_btn_hitboxes = {}

        # Preset display box
        preset_name = self.current_preset if self.current_preset else "custom"
        preset_col = C_ACCENT if self.current_preset else C_TEXT_DIM
        box_h = 28
        pygame.draw.rect(self.screen, C_PANEL_HI,
                         (x + 8, yy, w - 16, box_h), 1, border_radius=4)
        self.screen.blit(self.fs.render("PRESET", True, C_TEXT_DIM),
                         (x + 14, yy + 2))
        ptxt = self.fb.render(preset_name.upper(), True, preset_col)
        self.screen.blit(ptxt, (x + 14, yy + 12))
        # Auto-adapt indicator inside preset box
        if self.auto_adapt_enabled:
            aa_lbl = self.fmicro.render("AUTO", True, C_GOOD)
            self.screen.blit(aa_lbl, (x + w - aa_lbl.get_width() - 14, yy + 4))
            aa_reason = self.fmicro.render(
                self.auto_adapt_reason[:18], True, C_GOOD)
            self.screen.blit(aa_reason, (x + w - aa_reason.get_width() - 14, yy + 14))
        yy += box_h + 6

        # Preset selector buttons (1-4)
        btn_w = (w - 24) // 4
        btn_h = 20
        for i, pname in enumerate(PRESET_NAMES):
            bx = x + 8 + i * (btn_w + 2)
            is_active = self.current_preset == pname
            btn_col = C_ACCENT if is_active else C_PANEL_HI
            txt_col = C_ACCENT if is_active else C_TEXT_DIM
            pygame.draw.rect(self.screen, btn_col,
                             (bx, yy, btn_w, btn_h), 1, border_radius=3)
            if is_active:
                pygame.draw.rect(self.screen, C_PANEL,
                                 (bx + 1, yy + 1, btn_w - 2, btn_h - 2),
                                 border_radius=3)
            pbtn_txt = self.fmicro.render(pname[:6], True, txt_col)
            self.screen.blit(pbtn_txt,
                             (bx + (btn_w - pbtn_txt.get_width()) // 2,
                              yy + (btn_h - pbtn_txt.get_height()) // 2))
            self._left_btn_hitboxes["preset_%d" % i] = pygame.Rect(
                bx, yy, btn_w, btn_h)
        yy += btn_h + 6

        # Auto-Adapt toggle button
        aa_btn_w = w - 16
        aa_btn_h = 22
        aa_col = C_GOOD if self.auto_adapt_enabled else C_PANEL_HI
        aa_txt_col = C_GOOD if self.auto_adapt_enabled else C_TEXT_DIM
        pygame.draw.rect(self.screen, aa_col,
                         (x + 8, yy, aa_btn_w, aa_btn_h), 1, border_radius=4)
        if self.auto_adapt_enabled:
            pygame.draw.circle(self.screen, C_GOOD,
                               (x + 18, yy + aa_btn_h // 2), 5)
        else:
            pygame.draw.circle(self.screen, C_TEXT_DIM,
                               (x + 18, yy + aa_btn_h // 2), 5, 1)
        aa_txt = self.fs.render(
            "Auto-Adapt: %s" % ("ON" if self.auto_adapt_enabled else "OFF"),
            True, aa_txt_col)
        self.screen.blit(aa_txt, (x + 30, yy + 4))
        self._left_btn_hitboxes["auto_adapt"] = pygame.Rect(
            x + 8, yy, aa_btn_w, aa_btn_h)
        yy += aa_btn_h + 10

        # Noise Reduction slider
        self._draw_slider(x + 14, yy, w - 28, "Noise Reduction",
                          self.noise_reduction, 0.3, 0.9,
                          "%.2f" % self.noise_reduction, "noise_reduction",
                          C_PROC)
        yy += 50

        # Enhance Strength slider
        self._draw_slider(x + 14, yy, w - 28, "Enhance Strength",
                          self.enhance_strength, 0.5, 2.5,
                          "%.2fx" % self.enhance_strength, "enhance_strength",
                          C_CLEAN)
        yy += 50

        # Voice Low slider
        self._draw_slider(x + 14, yy, w - 28, "Voice Band Low",
                          self.voice_low, 50, 1050,
                          "%d Hz" % self.voice_low, "voice_low",
                          C_VOICE)
        yy += 50

        # Voice High slider
        self._draw_slider(x + 14, yy, w - 28, "Voice Band High",
                          self.voice_high, 1000, 11000,
                          "%d Hz" % self.voice_high, "voice_high",
                          C_VOICE)
        yy += 50

        # Wind Filter Strength slider
        self._draw_slider(x + 14, yy, w - 28, "Wind Filter Strength",
                          self.wind_filter_strength, 0.0, 1.0,
                          "%.2f" % self.wind_filter_strength, "wind_filter_strength",
                          C_COOL)
        yy += 50

        # De-reverb Strength slider
        self._draw_slider(x + 14, yy, w - 28, "De-Reverb Strength",
                          self.dereverb_strength, 0.0, 0.6,
                          "%.2f" % self.dereverb_strength, "dereverb_strength",
                          C_COOL)
        yy += 50

        # Compressor Threshold slider
        self._draw_slider(x + 14, yy, w - 28, "Compressor Threshold",
                          self.compressor_threshold, -40.0, -10.0,
                          "%.0f dB" % self.compressor_threshold, "compressor_threshold",
                          C_HOT)
        yy += 50

        # EQ Bass slider
        self._draw_slider(x + 14, yy, w - 28, "EQ Bass",
                          self.eq_bass, -12.0, 12.0,
                          "%+.0f dB" % self.eq_bass, "eq_bass",
                          C_ACCENT)
        yy += 50

        # EQ Mid slider
        self._draw_slider(x + 14, yy, w - 28, "EQ Mid",
                          self.eq_mid, -12.0, 12.0,
                          "%+.0f dB" % self.eq_mid, "eq_mid",
                          C_ACCENT)
        yy += 50

        # EQ Treble slider
        self._draw_slider(x + 14, yy, w - 28, "EQ Treble",
                          self.eq_treble, -12.0, 12.0,
                          "%+.0f dB" % self.eq_treble, "eq_treble",
                          C_ACCENT)
        yy += 56

        # Status section
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("LIVE STATUS", True, C_ACCENT),
                         (x + 12, yy))
        yy += 22

        status_items = [
            ("Audio Input", "LIVE" if self.capture.available else "NO INPUT",
             C_GOOD if self.capture.available else C_BAD),
            ("Device", self.capture.device_name[:22] if self.capture.available else "----",
             C_TEXT if self.capture.available else C_TEXT_DIM),
            ("Sample Rate", "%d Hz" % SAMPLE_RATE, C_TEXT),
            ("FFT Size", "%d" % FFT_SIZE, C_TEXT),
            ("Input Gain", "%.2fx" % self.input_gain,
             C_GOOD if 0.8 <= self.input_gain <= 1.5 else C_WARN),
            ("Monitor", "ON  vol:%.0f%%" % (self.capture.output_volume * 100) if self.capture.output_enabled else "OFF",
             C_GOOD if self.capture.output_enabled else C_TEXT_DIM),
            ("Recording", "ON" if self.recording else "OFF",
             C_BAD if self.recording else C_TEXT_DIM),
            ("VAD-REC", "ARMED" if self.vad_record else "off",
             C_WARN if self.vad_record else C_TEXT_DIM),
            ("Deep Listen", "ON" if self.deep_listen else "off",
             C_GOOD if self.deep_listen else C_TEXT_DIM),
            ("Whisper", "ON" if self.whisper_mode else "off",
             C_GOOD if self.whisper_mode else C_TEXT_DIM),
            ("Noise Profile", "LEARNING" if self.noise_profile_learning else ("SET" if self.noise_profile is not None else "off"),
             C_WARN if self.noise_profile_learning else (C_GOOD if self.noise_profile is not None else C_TEXT_DIM)),
            ("Freq Isolator", "%.0f-%.0fHz" % (self.freq_isolator_low, self.freq_isolator_high) if self.freq_isolator_enabled else "off",
             C_GOOD if self.freq_isolator_enabled else C_TEXT_DIM),
            ("Focus Lock", "%d deg %s" % (self.focus_angle, self._compass_dir(self.focus_angle)) if self.focus_lock_enabled else "OFF",
             C_GOOD if self.focus_lock_enabled else C_TEXT_DIM),
            ("A/B Compare", "RAW" if self.ab_compare else "processed",
             C_WARN if self.ab_compare else C_TEXT_DIM),
            ("Preset", self.current_preset if self.current_preset else "custom",
             C_ACCENT if self.current_preset else C_TEXT_DIM),
            ("SNR Improve", "%.1f dB" % self.snr_improvement if self.has_real_audio else "----",
             C_GOOD if self.snr_improvement > 5 else (C_WARN if self.snr_improvement > 0 else C_BAD)),
            ("AGC", "ON" if self.agc_enabled else "off",
             C_GOOD if self.agc_enabled else C_TEXT_DIM),
            ("Dyn Range", "%.1f dB" % self.dynamic_range_db if self.has_real_audio else "----",
             C_GOOD if self.dynamic_range_db > 10 else (C_WARN if self.dynamic_range_db > 3 else C_TEXT_DIM)),
            ("EQ B/M/T", "%+.0f/%+.0f/%+.0f" % (self.eq_bass, self.eq_mid, self.eq_treble),
             C_ACCENT if (self.eq_bass != 0 or self.eq_mid != 0 or self.eq_treble != 0) else C_TEXT_DIM),
            ("Spectro Speed", "%.1fx" % self._spectrogram_speed,
             C_TEXT if self._spectrogram_speed != 1.0 else C_TEXT_DIM),
            ("THD+N", "%.1f%%" % self.thd_n_percent if self.has_real_audio else "----",
             C_WARN if self.thd_n_percent > 10 else (C_GOOD if self.thd_n_percent > 0 else C_TEXT_DIM)),
            ("Source", self.source_class,
             C_VOICE if self.source_class == "speech" else (C_ACCENT if self.source_class == "music" else (C_COOL if self.source_class == "tone" else C_TEXT_DIM))),
            ("RT60", "%.0f ms" % self.rt60_ms if self.rt60_ms > 0 else "----",
             C_WARN if self.rt60_ms > 800 else (C_GOOD if self.rt60_ms > 0 else C_TEXT_DIM)),
            ("Quality", "%s (%.0f)" % (self.quality_grade, self.quality_score) if self.has_real_audio else "----",
             C_GOOD if self.quality_score >= 70 else (C_WARN if self.quality_score >= 40 else C_BAD)),
            ("Latency", "%.1f ms" % self._proc_latency_ms,
             C_GOOD if self._proc_latency_ms < 10 else (C_WARN if self._proc_latency_ms < 30 else C_BAD)),
            ("FPS", "%.0f" % self._fps,
             C_GOOD if self._fps > 50 else (C_WARN if self._fps > 30 else C_BAD)),
            ("Processing", "PAUSED" if self.paused else ("RUNNING" if self.has_real_audio else "IDLE"),
             C_WARN if self.paused else (C_GOOD if self.has_real_audio else C_TEXT_DIM)),
            ("Frame", "%d" % self._frame, C_TEXT_DIM),
        ]
        for lab, val, col in status_items:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (x + 12, yy))
            img = self.fs.render(val, True, col)
            self.screen.blit(img, (x + w - img.get_width() - 12, yy))
            yy += 19

        # Processing model info
        yy += 8
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("PROCESSING MODEL", True, C_ACCENT),
                         (x + 12, yy))
        yy += 22
        model_items = [
            ("Pipeline", "11-stage DSP", C_TEXT),
            ("Gate", "Spectral tanh" if self.gate_enabled else "off", C_TEXT_DIM),
            ("DeReverb", "Spectral smooth" if self.dereverb_enabled else "off", C_TEXT_DIM),
            ("Compressor", "3-band comp" if self.compress_enabled else "off", C_TEXT_DIM),
            ("Presence", "Consonant boost" if self.presence_enabled else "off", C_TEXT_DIM),
            ("Formant", "F1/F2 vowel" if self.formant_enabled else "off", C_TEXT_DIM),
            ("SuperRes", "Harmonic recon" if self.superres_enabled else "off", C_TEXT_DIM),
            ("Loudness", "Auto-normalize" if self.loudness_enabled else "off", C_TEXT_DIM),
            ("Wind HPF", "Adaptive" if self.wind_filter_enabled else "off", C_TEXT_DIM),
            ("AutoAdapt", "Environment" if self.auto_adapt_enabled else "off", C_TEXT_DIM),
            ("Deep Listen", "MAX BOOST" if self.deep_listen else "off",
             C_GOOD if self.deep_listen else C_TEXT_DIM),
            ("Whisper", "3x GAIN" if self.whisper_mode else "off",
             C_GOOD if self.whisper_mode else C_TEXT_DIM),
            ("Noise Profile", "LEARNED" if self.noise_profile is not None else "off",
             C_COOL if self.noise_profile is not None else C_TEXT_DIM),
            ("Freq Isolator", "300-3k" if self.freq_isolator_enabled else "off",
             C_VOICE if self.freq_isolator_enabled else C_TEXT_DIM),
            ("AGC", "Auto gain" if self.agc_enabled else "off",
             C_GOOD if self.agc_enabled else C_TEXT_DIM),
            ("3-Band EQ", "B/M/T" if (self.eq_bass != 0 or self.eq_mid != 0 or self.eq_treble != 0) else "off",
             C_ACCENT if (self.eq_bass != 0 or self.eq_mid != 0 or self.eq_treble != 0) else C_TEXT_DIM),
            ("VAD-Rec", "ARMED" if self.vad_record else "off",
             C_WARN if self.vad_record else C_TEXT_DIM),
            ("VAD", "Energy+ZCR+flat", C_TEXT_DIM),
            ("Noise Est.", "Exp. moving avg.", C_TEXT_DIM),
        ]
        for lab, val, col in model_items:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (x + 12, yy))
            img = self.fs.render(val, True, col)
            self.screen.blit(img, (x + w - img.get_width() - 12, yy))
            yy += 19

    def _draw_slider(self, x, y, w, label, val, lo, hi, valtext, key, color):
        self.screen.blit(self.fs.render(label, True, C_TEXT_DIM), (x, y))
        img = self.fs.render(valtext, True, color)
        self.screen.blit(img, (x + w - img.get_width(), y))
        ry = y + 16
        rh = 8
        rect = pygame.Rect(x, ry, w, rh)
        pygame.draw.rect(self.screen, C_PANEL_HI, rect, border_radius=4)
        frac = clamp((val - lo) / (hi - lo))
        pygame.draw.rect(self.screen, color,
                         (x, ry, int(w * frac), rh), border_radius=4)
        kx = x + int(w * frac)
        pygame.draw.circle(self.screen, C_TEXT, (kx, ry + rh // 2), 6)
        pygame.draw.circle(self.screen, color, (kx, ry + rh // 2), 6, 2)
        self._slider_hitboxes[key] = rect

    # ---- RIGHT PANEL (metrics) -------------------------------------------

    def draw_right_panel(self):
        w = self.RIGHT_PANEL_W - 16
        x = self.W - self.RIGHT_PANEL_W + 8
        y = self.TOP_BAR_H + 4
        h = self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 12
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("METRICS", True, C_ACCENT),
                         (x + 12, y + 8))

        yy = y + 36

        # Big distance readout
        if self.has_real_audio:
            dist_col = C_GOOD if self.distance_est < 200 else (
                C_WARN if self.distance_est < 500 else C_BAD)
            dist_txt = "%.0f m" % self.distance_est if self.distance_est < 999 \
                else "----"
        else:
            dist_col = C_TEXT_DIM
            dist_txt = "----"
        self.screen.blit(self.fhuge.render(dist_txt, True, dist_col),
                         (x + 14, yy))
        self.screen.blit(self.fs.render("EST. DISTANCE (approx)", True, C_TEXT_DIM),
                         (x + 14, yy + 50))
        yy += 76

        # Level bars -- show real values or ---- when no input
        if self.has_real_audio:
            bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
                clamp((self.db_level + 60) / 90), C_ACCENT,
                "SIGNAL LEVEL", "%.1f dB" % self.db_level)
        else:
            bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
                0, C_TEXT_DIM, "SIGNAL LEVEL", "----")
        yy += 40

        if self.has_real_audio:
            bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
                clamp((self.snr_est + 10) / 40), C_GOOD,
                "SNR", "%.1f dB" % self.snr_est)
        else:
            bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
                0, C_TEXT_DIM, "SNR", "----")
        yy += 40

        if self.has_real_audio:
            bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
                clamp(self.peak_level), C_HOT,
                "PEAK OUTPUT", "%.3f" % self.peak_level)
        else:
            bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
                0, C_TEXT_DIM, "PEAK OUTPUT", "----")
        yy += 40

        bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
            clamp(self.noise_reduction), C_PROC,
            "NOISE REDUCTION", "%.0f%%" % (self.noise_reduction * 100))
        yy += 40

        bar(self.screen, self.fs, x + 14, yy + 16, w - 28, 12,
            clamp((self.enhance_strength - 0.5) / 2.0), C_CLEAN,
            "ENHANCE GAIN", "%.2fx" % self.enhance_strength)
        yy += 48

        # Voice detection indicator
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("DETECTION", True, C_ACCENT),
                         (x + 12, yy))
        yy += 22
        if self.has_real_audio:
            det_col = C_GOOD if self.voice_detected else C_TEXT_DIM
            det_txt = "VOICE DETECTED" if self.voice_detected else "silence"
        else:
            det_col = C_TEXT_DIM
            det_txt = "NO INPUT"
        pygame.draw.circle(self.screen, det_col, (x + 22, yy + 6), 6)
        self.screen.blit(self.fs.render(det_txt, True, det_col),
                         (x + 36, yy))
        if self.has_real_audio:
            vad_txt = self.fmicro.render("VAD: %.0f%%" % (self.vad_confidence * 100), True,
                                         C_GOOD if self.vad_confidence > 0.5 else C_TEXT_DIM)
            self.screen.blit(vad_txt, (x + w - vad_txt.get_width() - 12, yy + 2))
        yy += 24

        # Direction info
        if self.has_real_audio:
            peak_dir = int(np.argmax(self.direction_bins))
            peak_angle = peak_dir * 360 / NUM_DIRECTIONS
            rows = [
                ("Peak freq bin", "%d deg %s" % (
                    peak_angle, self._compass_dir(peak_angle)), C_DIRECTION),
                ("Peak intensity", "%.2f" % self.direction_bins[peak_dir],
                 C_TEXT),
                ("Compass", "%d bins / 360 deg" % NUM_DIRECTIONS, C_TEXT_DIM),
            ]
        else:
            rows = [
                ("Peak freq bin", "----", C_TEXT_DIM),
                ("Peak intensity", "----", C_TEXT_DIM),
                ("Compass", "%d bins / 360 deg" % NUM_DIRECTIONS, C_TEXT_DIM),
            ]
        for lab, val, col in rows:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (x + 12, yy))
            img = self.fs.render(val, True, col)
            self.screen.blit(img, (x + w - img.get_width() - 12, yy))
            yy += 19

        yy += 8
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("AUDIO QUALITY", True, C_ACCENT),
                         (x + 12, yy))
        yy += 22
        if self.has_real_audio:
            quality_rows = [
                ("A-Weighted", "%.1f dB(A)" % self.aweighted_db, C_ACCENT),
                ("Centroid", "%.0f Hz" % self.centroid_hz, C_VOICE),
                ("Flatness", "%.3f" % self.flatness,
                 C_GOOD if self.flatness < 0.4 else C_WARN),
                ("SII Score", "%.2f" % self.sii_score,
                 C_GOOD if self.sii_score > 0.5 else (C_WARN if self.sii_score > 0.3 else C_BAD)),
                ("Wind", "%.0f%%" % (self.wind_strength * 100) if self.wind_detected else "no",
                 C_WARN if self.wind_detected else C_GOOD),
                ("Clipping", "yes" if self.clipping else "no",
                 C_BAD if self.clipping else C_GOOD),
                ("Confidence", "%.0f%%" % (self.confidence_score * 100),
                 C_GOOD if self.confidence_score > 0.5 else (C_WARN if self.confidence_score > 0.3 else C_BAD)),
                ("Pitch", "%.0f Hz" % self.pitch_hz if self.pitch_hz > 0 else "----",
                 C_VOICE if self.pitch_hz > 0 else C_TEXT_DIM),
                ("Latency", "%.1f ms" % self._proc_latency_ms,
                 C_GOOD if self._proc_latency_ms < 10 else (C_WARN if self._proc_latency_ms < 30 else C_BAD)),
            ]
        else:
            quality_rows = [
                ("A-Weighted", "----", C_TEXT_DIM),
                ("Centroid", "----", C_TEXT_DIM),
                ("Flatness", "----", C_TEXT_DIM),
                ("SII Score", "----", C_TEXT_DIM),
                ("Wind", "----", C_TEXT_DIM),
                ("Clipping", "----", C_TEXT_DIM),
                ("Confidence", "----", C_TEXT_DIM),
                ("Pitch", "----", C_TEXT_DIM),
                ("Latency", "----", C_TEXT_DIM),
            ]
        for lab, val, col in quality_rows:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM),
                             (x + 12, yy))
            img = self.fs.render(val, True, col)
            self.screen.blit(img, (x + w - img.get_width() - 12, yy))
            yy += 19

        # Spectral peaks (top 3)
        if self.has_real_audio and self.spectral_peaks:
            yy += 4
            self.screen.blit(self.fs.render("SPECTRAL PEAKS", True, C_ACCENT),
                             (x + 12, yy))
            yy += 18
            for i, (freq, db) in enumerate(self.spectral_peaks):
                pk_txt = self.fmicro.render(
                    "P%d: %.0f Hz  %.1f dB" % (i + 1, freq, db), True,
                    C_VOICE if i == 0 else C_TEXT)
                self.screen.blit(pk_txt, (x + 16, yy))
                yy += 14

        # Long-term dB histogram
        if self.has_real_audio and self._db_hist_count > 30:
            yy += 6
            self.screen.blit(self.fs.render("LEVEL DISTRIBUTION", True, C_ACCENT),
                             (x + 12, yy))
            yy += 18
            hg_w = w - 28
            hg_h = 36
            hg_x = x + 12
            hg_y = yy
            pygame.draw.rect(self.screen, C_PANEL_HI,
                             (hg_x, hg_y, hg_w, hg_h), 1)
            max_count = max(self._db_histogram) if self._db_histogram else 1
            if max_count > 0:
                bin_w = hg_w / 30
                for i in range(30):
                    bh = int(self._db_histogram[i] / max_count * hg_h)
                    col = C_GOOD if i > 20 else (C_WARN if i > 10 else C_BAD)
                    pygame.draw.rect(self.screen, col,
                                     (hg_x + int(i * bin_w), hg_y + hg_h - bh,
                                      max(1, int(bin_w)), bh))
            # Axis labels
            lbl_lo = self.fmicro.render("-60", True, C_TEXT_DIM)
            lbl_hi = self.fmicro.render("0 dB", True, C_TEXT_DIM)
            self.screen.blit(lbl_lo, (hg_x, hg_y + hg_h + 2))
            self.screen.blit(lbl_hi, (hg_x + hg_w - lbl_hi.get_width(), hg_y + hg_h + 2))
            n_txt = self.fmicro.render(
                "n=%d" % self._db_hist_count, True, C_TEXT_DIM)
            self.screen.blit(n_txt, (hg_x + hg_w // 2 - n_txt.get_width() // 2, hg_y + hg_h + 2))
            yy += hg_h + 18

        # Session statistics
        if self.has_real_audio and self._session_snr_count > 10:
            yy += 6
            pygame.draw.line(self.screen, C_PANEL_HI,
                             (x + 8, yy), (x + w - 8, yy), 1)
            yy += 6
            self.screen.blit(self.fs.render("SESSION STATS", True, C_ACCENT),
                             (x + 12, yy))
            elapsed = time.time() - self._session_start_time
            el_txt = self.fmicro.render("%.0fs" % elapsed, True, C_TEXT_DIM)
            self.screen.blit(el_txt, (x + w - el_txt.get_width() - 12, yy + 2))
            yy += 18
            if self._session_snr_count > 0:
                snr_avg = self._session_snr_sum / self._session_snr_count
                snr_min = self._session_snr_min if self._session_snr_min < 999 else 0
                snr_max = self._session_snr_max if self._session_snr_max > -999 else 0
                db_avg = self._session_db_sum / self._session_db_count
                db_min = self._session_db_min if self._session_db_min < 999 else 0
                db_max = self._session_db_max if self._session_db_max > -999 else 0
                conf_avg = self._session_conf_sum / max(self._session_conf_count, 1)
                stat_rows = [
                    ("SNR avg", "%.1f dB" % snr_avg, C_GOOD if snr_avg > 10 else C_WARN),
                    ("SNR min/max", "%.1f / %.1f" % (snr_min, snr_max), C_TEXT),
                    ("dB avg", "%.1f dB" % db_avg, C_TEXT),
                    ("dB min/max", "%.1f / %.1f" % (db_min, db_max), C_TEXT),
                    ("Conf avg", "%.0f%%" % (conf_avg * 100), C_GOOD if conf_avg > 0.5 else C_WARN),
                ]
                for lab, val, col in stat_rows:
                    self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM),
                                     (x + 14, yy))
                    img = self.fmicro.render(val, True, col)
                    self.screen.blit(img, (x + w - img.get_width() - 12, yy))
                    yy += 14

        yy += 8
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("PROCESSING PIPELINE", True, C_ACCENT),
                         (x + 12, yy))
        yy += 22
        pipeline = [
            ("1. Capture", "PyAudio float32", C_RAW),
            ("2. Wind HPF", "Adaptive filter", C_COOL),
            ("3. Bandpass", "Butter 8th order", C_VOICE),
            ("4. Gate", "Spectral tanh", C_NOISE),
            ("5. DeReverb", "Spectral smooth", C_COOL),
            ("6. Decoherence", "Noise subtraction", C_PROC),
            ("7. Wiener", "Spectral enhance", C_CLEAN),
            ("8. Compress", "3-band comp", C_HOT),
            ("9. Presence", "Consonant boost", C_VOICE),
            ("10. Formant", "F1/F2 vowel", C_ACCENT),
            ("11. SuperRes+Norm", "Harmonic + loudness", C_GOOD),
        ]
        for lab, val, col in pipeline:
            pygame.draw.circle(self.screen, col, (x + 22, yy + 6), 4)
            self.screen.blit(self.fs.render(lab, True, C_TEXT),
                             (x + 32, yy))
            img = self.fmicro.render(val, True, C_TEXT_DIM)
            self.screen.blit(img, (x + w - img.get_width() - 12, yy + 2))
            yy += 19

        # SNR history mini-graph
        yy += 8
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("SNR HISTORY", True, C_ACCENT),
                         (x + 12, yy))
        yy += 18
        graph_h = 40
        graph_w = w - 28
        pygame.draw.rect(self.screen, C_PANEL_HI,
                         (x + 12, yy, graph_w, graph_h), 1)
        if len(self.snr_history) > 1 and self.has_real_audio:
            snr_vals = list(self.snr_history)
            n = len(snr_vals)
            for i in range(n - 1):
                frac1 = clamp((snr_vals[i] + 20) / 40)
                frac2 = clamp((snr_vals[i + 1] + 20) / 40)
                px1 = x + 12 + int(i * graph_w / max(n - 1, 1))
                px2 = x + 12 + int((i + 1) * graph_w / max(n - 1, 1))
                py1 = yy + graph_h - int(frac1 * graph_h)
                py2 = yy + graph_h - int(frac2 * graph_h)
                col = C_GOOD if snr_vals[i + 1] > 10 else (C_WARN if snr_vals[i + 1] > 0 else C_BAD)
                pygame.draw.line(self.screen, col, (px1, py1), (px2, py2), 1)
            snr_now = self.fmicro.render("%.1f dB" % snr_vals[-1], True, C_ACCENT)
            self.screen.blit(snr_now, (x + 16, yy + 2))
        else:
            idle = self.fmicro.render("no data", True, C_TEXT_DIM)
            self.screen.blit(idle, (x + 16, yy + graph_h // 2 - 6))
        yy += graph_h + 8

        # dB level history mini-graph
        self.screen.blit(self.fs.render("LEVEL HISTORY (dB)", True, C_ACCENT),
                         (x + 12, yy))
        yy += 18
        graph_h2 = 40
        pygame.draw.rect(self.screen, C_PANEL_HI,
                         (x + 12, yy, graph_w, graph_h2), 1)
        if len(self.db_history) > 1 and self.has_real_audio:
            db_vals = list(self.db_history)
            n = len(db_vals)
            for i in range(n - 1):
                frac1 = clamp((db_vals[i] + 60) / 60.0)
                frac2 = clamp((db_vals[i + 1] + 60) / 60.0)
                px1 = x + 12 + int(i * graph_w / max(n - 1, 1))
                px2 = x + 12 + int((i + 1) * graph_w / max(n - 1, 1))
                py1 = yy + graph_h2 - int(frac1 * graph_h2)
                py2 = yy + graph_h2 - int(frac2 * graph_h2)
                col = C_GOOD if db_vals[i + 1] > -18 else (C_WARN if db_vals[i + 1] > -6 else C_BAD)
                pygame.draw.line(self.screen, col, (px1, py1), (px2, py2), 1)
            db_now = self.fmicro.render("%.1f dB" % db_vals[-1], True, C_ACCENT)
            self.screen.blit(db_now, (x + 16, yy + 2))
        else:
            idle = self.fmicro.render("no data", True, C_TEXT_DIM)
            self.screen.blit(idle, (x + 16, yy + graph_h2 // 2 - 6))
        yy += graph_h2 + 8

        # Confidence bar
        self.screen.blit(self.fs.render("CONFIDENCE", True, C_ACCENT),
                         (x + 12, yy))
        yy += 18
        cb_w = graph_w
        cb_h = 14
        pygame.draw.rect(self.screen, C_PANEL_HI, (x + 12, yy, cb_w, cb_h), 1)
        conf_frac = clamp(self.confidence_score)
        conf_col = C_GOOD if conf_frac > 0.5 else (C_WARN if conf_frac > 0.3 else C_BAD)
        if conf_frac > 0:
            pygame.draw.rect(self.screen, conf_col,
                             (x + 13, yy + 1, int((cb_w - 2) * conf_frac), cb_h - 2))
        conf_txt = self.fmicro.render(
            "%.0f%%" % (conf_frac * 100), True, conf_col)
        self.screen.blit(conf_txt, (x + 12 + cb_w + 4, yy))
        yy += cb_h + 8

        # Auto-adapt status
        if self.auto_adapt_enabled:
            aa_txt = self.fs.render("AUTO-ADAPT: %s" % self.auto_adapt_reason, True, C_GOOD)
            self.screen.blit(aa_txt, (x + 12, yy))
        yy += 20

        # Event log / status feed
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("EVENT LOG", True, C_ACCENT),
                         (x + 12, yy))
        yy += 18
        for i, event in enumerate(list(self.event_log)):
            ev_col = C_TEXT_DIM if i > 0 else C_TEXT
            ev_txt = self.fsmall.render(event[:34], True, ev_col)
            self.screen.blit(ev_txt, (x + 12, yy))
            yy += 14

        # Activity timeline (voice/sound detection over session)
        if len(self.activity_timeline) > 10:
            yy += 4
            pygame.draw.line(self.screen, C_PANEL_HI,
                             (x + 8, yy), (x + w - 8, yy), 1)
            yy += 6
            self.screen.blit(self.fs.render("ACTIVITY TIMELINE", True, C_ACCENT),
                             (x + 12, yy))
            yy += 18
            tl_w = w - 24
            tl_h = 30
            tl_x = x + 12
            tl_y = yy
            panel(self.screen, tl_x, tl_y, tl_w, tl_h, alpha=160)
            n = len(self.activity_timeline)
            bar_w = max(1, tl_w / n)
            for i, entry in enumerate(self.activity_timeline):
                bx = tl_x + int(i * bar_w)
                if entry['voice']:
                    col = C_GOOD
                elif entry['level'] > -50:
                    col = C_WARN
                else:
                    col = C_TEXT_DIM
                bh = max(2, int((entry['level'] + 60) / 60 * tl_h))
                bh = clamp(bh, 2, tl_h)
                pygame.draw.rect(self.screen, col,
                                 (bx, tl_y + tl_h - bh, max(1, int(bar_w)), bh))
            yy += tl_h + 4

        # Recording timeline with golden markers
        if self.recording or len(self.golden_markers) > 0:
            yy += 4
            pygame.draw.line(self.screen, C_PANEL_HI,
                             (x + 8, yy), (x + w - 8, yy), 1)
            yy += 6
            rec_label = "RECORDING TIMELINE" if self.recording else "RECORDING (stopped)"
            self.screen.blit(self.fs.render(rec_label, True, C_HOT),
                             (x + 12, yy))
            if self.recording:
                elapsed = time.time() - self.record_start_time
                rec_time = self.fmicro.render("%.1fs" % elapsed, True, C_HOT)
                self.screen.blit(rec_time, (x + w - rec_time.get_width() - 12, yy + 2))
            yy += 18
            rt_w = w - 24
            rt_h = 16
            rt_x = x + 12
            rt_y = yy
            pygame.draw.rect(self.screen, C_PANEL_HI,
                             (rt_x, rt_y, rt_w, rt_h), 1)
            if self.recording:
                elapsed = time.time() - self.record_start_time
                # Progress bar (capped at 60s for display)
                prog = clamp(elapsed / 60.0, 0, 1)
                pygame.draw.rect(self.screen, C_HOT,
                                 (rt_x + 1, rt_y + 1, int((rt_w - 2) * prog), rt_h - 2))
            # Golden markers
            if self.recording and self.record_start_time:
                elapsed = time.time() - self.record_start_time
                for marker in self.golden_markers:
                    frac = clamp(marker / max(elapsed, 0.1), 0, 1)
                    mx = rt_x + int(frac * rt_w)
                    pygame.draw.line(self.screen, C_ACCENT,
                                     (mx, rt_y - 2), (mx, rt_y + rt_h + 2), 2)
                    pygame.draw.circle(self.screen, C_ACCENT,
                                       (mx, rt_y + rt_h // 2), 3)
            elif self.golden_markers:
                for i, marker in enumerate(self.golden_markers):
                    frac = clamp(i / max(len(self.golden_markers), 1), 0, 1)
                    mx = rt_x + int(frac * rt_w)
                    pygame.draw.line(self.screen, C_ACCENT,
                                     (mx, rt_y - 2), (mx, rt_y + rt_h + 2), 2)
            marker_txt = self.fmicro.render(
                "%d markers" % len(self.golden_markers), True, C_ACCENT)
            self.screen.blit(marker_txt, (rt_x + 4, rt_y + rt_h + 2))
            yy += rt_h + 20

    # ---- BOTTOM BAR ------------------------------------------------------

    def draw_bottom_bar(self):
        x = 0
        y = self.H - self.BOTTOM_BAR_H
        w = self.W
        h = self.BOTTOM_BAR_H
        pygame.draw.rect(self.screen, C_PANEL, (x, y, w, h))
        pygame.draw.line(self.screen, C_PANEL_HI,
                         (x, y), (x + w, y), 1)
        line1 = "TAB mode  H help  I info  P pause  R reset  M monitor  S record  F focus  B A/B  E auto  K deep  J VAD-rec  A AGC  O replay  ESC close/quit"
        line2 = "G gate  D dereverb  C comp  V presence  T formant  X superres  L loudness  W wind  1-4 presets  N noise  U whisper  Y isolator  Z spec-avg  F2 snap  F3 mark"
        line3 = "EQ: [ ] bass  ; ' mid  , . treble  0 reset  |  F4 CSV  F5 clipboard  PGUP/DN spectro speed  Shift+1-4 save preset"
        self.screen.blit(self.fs.render(line1, True, C_TEXT),
                         (x + 12, y + 10))
        self.screen.blit(self.fs.render(line2, True, C_TEXT_DIM),
                         (x + 12, y + 30))
        self.screen.blit(self.fs.render(line3, True, C_TEXT_DIM),
                         (x + 12, y + 50))
        # Audio status indicator
        if self.capture.available and self.has_real_audio:
            status = "LIVE AUDIO"
            status_col = C_GOOD
        elif self.capture.available and not self.has_real_audio:
            status = "MIC READY (waiting)"
            status_col = C_WARN
        else:
            status = "NO INPUT DEVICE"
            status_col = C_BAD
        img = self.fs.render(status, True, status_col)
        self.screen.blit(img, (x + w - img.get_width() - 12, y + 10))
        # Monitor + recording + focus status
        parts = []
        if self.recording:
            rec_dur = time.time() - self.record_start_time
            parts.append(("REC %.1fs" % rec_dur, C_BAD))
        if self.focus_lock_enabled:
            parts.append(("FOC", C_GOOD))
        mon_state = "MON" if self.capture.output_enabled else None
        if mon_state:
            parts.append((mon_state, C_GOOD))
        if parts:
            label = "  ".join(p[0] for p in parts)
            col = parts[0][1]
            mon_img = self.fs.render(
                "%s  vol:%.0f%%" % (label, self.capture.output_volume * 100),
                True, col)
        else:
            mon_img = self.fs.render(
                "vol:%.0f%%" % (self.capture.output_volume * 100),
                True, C_TEXT_DIM)
        self.screen.blit(mon_img, (x + w - mon_img.get_width() - 12, y + 30))
        if self.paused:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 300.0)
            pause_col = (int(C_WARN[0] * pulse + C_TEXT_DIM[0] * (1 - pulse)),
                         int(C_WARN[1] * pulse + C_TEXT_DIM[1] * (1 - pulse)),
                         int(C_WARN[2] * pulse + C_TEXT_DIM[2] * (1 - pulse)))
            img2 = self.fb.render("|| PAUSED", True, pause_col)
            self.screen.blit(img2, (x + w - img2.get_width() - 200, y + 4))

    # ---- HELP ------------------------------------------------------------

    def draw_help(self):
        w = min(620, self.W - 60)
        x = (self.W - w) // 2
        # Dim background
        dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        dim.fill((4, 6, 10, 200))
        self.screen.blit(dim, (0, 0))
        # Panel height: fit content but cap to window
        h = min(self.H - 60, 700)
        y = (self.H - h) // 2
        panel(self.screen, x, y, w, h, alpha=244)
        self.screen.blit(self.fb.render("CONTROLS", True, C_ACCENT),
                         (x + 20, y + 16))
        self.screen.blit(self.fs.render(
            "scroll: mouse wheel    close: H or ESC", True, C_TEXT_DIM),
            (x + w - 260, y + 24))
        lines = [
            "TAB            cycle SPECTRUM / DIRECTION / FILTER / METER",
            "or click       mode tabs in the top bar",
            "",
            "SPECTRUM MODE",
            "  Real-time frequency spectrum (raw vs enhanced)",
            "  Peak hold markers  +  click to read frequency",
            "  dB scale on Y-axis  +  frequency axis labels",
            "  Spectrogram waterfall with frequency axis + log intensity",
            "  Pitch overlay (F0 line + harmonics) when voice detected",
            "  Noise floor estimate (dashed line)",
            "  Confidence gauge bar (top-right, color-coded)",
            "  Spectral gain reduction overlay (red shading)",
            "  Waveform overlay (raw=grey  enhanced=green)",
            "",
            "DIRECTION MODE",
            "  Spectral energy map (NOT true direction -- single mic)",
            "  Shows voice-band frequency distribution on a compass",
            "  Distance rings with labels  +  confidence ring",
            "  Pitch detection  +  A-weighted dB readout",
            "  Distance estimation (rough, needs SPL calibration)",
            "",
            "FILTER MODE",
            "  11-stage pipeline visualization:",
            "  Raw -> Wind -> Bandpass -> Gate -> DeReverb",
            "       -> Decoherence -> Wiener -> Compressor",
            "       -> Presence -> Formant -> SuperRes -> Output",
            "  Each stage shows waveform + mini spectrum",
            "  RMS level + inter-stage gain/attenuation per stage",
            "",
            "METER MODE",
            "  Dual input/output VU/PPM meters with dB scale",
            "  3-color zones + peak hold with slow decay",
            "  Total gain reduction marker + headroom indicator",
            "  3-band gain reduction meters (LOW / MID / HIGH)",
            "  Quality metrics: A-weighted, centroid, flatness, SII,",
            "  VAD, wind, clipping, confidence, pitch, latency",
            "  Confidence history graph + feature toggle bar",
            "",
            "BLUEPRINT MODE",
            "  3D rotatable hardware model (100+ mesh groups)",
            "  Drag=orbit  Wheel=zoom  Shift+Drag=pan",
            "  R=reset  L=labels  E=exploded  W=wireframe",
            "  A=auto-rotate  F=focus part  T=toggle edges",
            "  Perspective projection + edge rendering + leader labels",
            "  Dish: 48x14 paraboloid + torus rim + hinges + hub + vanes",
            "  Focal mic: cage + grille + windscreen + XLR + strain relief",
            "  MEMS: 8x mics + PCB traces + solder pads + screws",
            "  Preamp: PCB + shield + op-amps + ADC + caps + jacks + screws",
            "  CPU: PCB + SoC + RAM + eMMC + heatsink + fan + 40-pin GPIO",
            "       + USB-A x2 + USB-C + HDMI + Ethernet + SD + LEDs",
            "  Display: bezel + LCD + touch + FPC + driver IC + backlight",
            "  Power: 4x 18650 + BMS + USB-C + switch + LEDs + gauge",
            "  Grip: trigger + spring + dial + buttons + tripod + strap",
            "  Enclosure: rails + vents + gasket + GPS + IMU + temp sensor",
            "  Cables: XLR + I2S + HDMI + power + GPS + ground",
            "",
            "PROCESSING TOGGLES",
            "  G    spectral gate on/off",
            "  D    de-reverberation on/off",
            "  C    multi-band compressor on/off",
            "  V    presence boost on/off",
            "  T    formant enhancement on/off",
            "  X    spectral super-resolution on/off",
            "  L    loudness normalization on/off",
            "  W    adaptive wind filter on/off",
            "  E    auto environment adaptation on/off",
            "  B    A/B compare (raw vs processed output)",
            "  K    deep listen (max all stages, boosted params)",
            "  J    VAD-gated recording (auto-record on voice)",
            "  N    noise profile learning (capture ambient noise)",
            "  U    whisper mode (3x gain for very quiet sources)",
            "  Y    frequency isolator (300-3000 Hz band only)",
            "  Z    spectrum averaging (toggle time-averaged display)",
            "  F2   spectrum snapshot (freeze for comparison)",
            "  F3   golden marker (mark moment during recording)",
            "  A    automatic gain control (fast adaptive gain)",
            "  O    replay last 6 seconds of audio",
            "  EQ   [ ] bass  ; ' mid  , . treble  0 reset (+/-1 dB each)",
            "  F4   export spectrum data to CSV file",
            "  F5   copy current metrics to clipboard",
            "  PGUP/DN  spectrogram scroll speed (0.2x to 3x)",
            "  Shift+1-4  save current settings to preset file",
            "  1-4  load preset (custom file if exists, else built-in)",
            "",
            "LEFT PANEL (clickable):",
            "  Preset buttons  click indoor/outdoor/distant/noisy",
            "  Auto-Adapt      click toggle button (or E key)",
            "  Sliders         drag to adjust 10 processing params",
            "",
            "M              toggle audio monitor (hear enhanced output)",
            "UP / DOWN      adjust monitor volume",
            "S              start / stop recording to WAV file",
            "F              toggle focus lock on peak direction",
            "P / SPACE      pause / resume processing",
            "R              reset sliders to defaults",
            "H              this help     I  full info     ESC  close/quit",
            "",
            "Press ESC or H to close this overlay and return to the app.",
        ]
        clip = pygame.Rect(x + 12, y + 46, w - 24, h - 56)
        self.screen.set_clip(clip)
        line_h = 16
        content_h = len(lines) * line_h
        max_scroll = max(0, content_h - (h - 60))
        self.help_scroll = min(self.help_scroll, max_scroll)
        for i, ln in enumerate(lines):
            yy = y + 52 + i * line_h - self.help_scroll
            if yy < y + 40 or yy > y + h:
                continue
            col = C_ACCENT if ln and not ln.startswith("  ") and ln.isupper() \
                else C_TEXT
            if ln.startswith("  "):
                col = C_TEXT_DIM
            self.screen.blit(self.font.render(ln, True, col),
                             (x + 24, yy))
        self.screen.set_clip(None)
        # Scrollbar
        if max_scroll > 0:
            sb_x = x + w - 10
            sb_y = y + 46
            sb_h = h - 56
            pygame.draw.rect(self.screen, C_PANEL_HI, (sb_x, sb_y, 4, sb_h), 1)
            thumb_h = max(20, int(sb_h * sb_h / (sb_h + max_scroll)))
            thumb_y = sb_y + int((sb_h - thumb_h) * self.help_scroll / max_scroll)
            pygame.draw.rect(self.screen, C_ACCENT, (sb_x, thumb_y, 4, thumb_h))

    # ---- INFO ------------------------------------------------------------

    def draw_info(self):
        w = min(800, self.W - 80)
        h = self.H - 90
        x = (self.W - w) // 2
        y = 50
        s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        s.fill((4, 6, 10, 249))
        self.screen.blit(s, (0, 0))
        panel(self.screen, x, y, w, h, alpha=244)
        self.screen.blit(self.fbig.render(
            "HEARNOCULAR -- FULL SPECIFICATION", True, C_ACCENT),
            (x + 20, y + 14))
        self.screen.blit(self.fs.render(
            "scroll: mouse wheel / arrows    close: I", True, C_TEXT_DIM),
            (x + w - 280, y + 22))

        sections = [
            ("SYSTEM OVERVIEW", [
                "The Hearnocular is a directional hearing and distance",
                "perception system inspired by the 'AetherEar' concept:",
                "a parabolic collector + MEMS mic array + multi-stage AI",
                "processing to pull intelligible speech from distant,",
                "noisy environments.",
                "",
                "This digital twin runs an 11-stage DSP pipeline in real",
                "time on live microphone input: audio capture -> wind",
                "filter -> bandpass -> spectral gate -> de-reverberation",
                "-> spectral decoherence -> Wiener filter -> multi-band",
                "compression -> presence boost -> formant enhancement",
                "-> spectral super-resolution -> loudness normalization",
                "-> directional analysis -> audio output.",
                "",
                "Target: 400-700+ ft intelligible speech in quiet",
                "conditions, 250-450 ft in moderate noise.",
            ]),
            ("HARDWARE CONCEPT", [
                "Reflector: 20-24 inch deployable parabolic dish",
                "  (carbon fiber segments, folding, magnetic locks)",
                "Microphones: Central focal mic + 8-16 MEMS array",
                "  (analog parabolic focus + digital beamforming)",
                "Processing: Edge processor for low-latency DSP",
                "  running Wiener filter spectral subtraction",
                "Preamp: Ultra-low-noise chain with Peltier cooling",
                "Form: Pistol-grip, ~10-15 lbs, shoulder strap",
                "Power: 4-8 hour battery target",
                "Size: 24x12x6 inches stowed, deploys to ~24 inch dish",
            ]),
            ("BILL OF MATERIALS (BOM)", [
                "REFLECTOR ASSEMBLY:",
                "  Parabolic dish: 22-24 inch carbon fiber composite",
                "    Kymeta-style folding segments, 6-panel deployable",
                "    Surface accuracy: < 0.5mm RMS for voice-band focus",
                "    Weight target: < 3 lbs",
                "  Reflector coating: RF-transparent matte finish",
                "    (reduces secondary reflections, weatherproof)",
                "",
                "MICROPHONE ARRAY:",
                "  Focal mic: Primo EM-272Z1 omnidirectional capsule",
                "    (or DPA 4060/4061 if budget allows)",
                "    Self-noise: < 14 dB-A, flat response to 20 kHz",
                "    Mounted at dish focal point on shock-isolated arm",
                "  MEMS array: 8x InvenSense ICS-43434 (or 16x TDK",
                "    ICS-40730) in circular arrangement around rim",
                "    Spacing: 85mm diameter circle (half-wavelength",
                "    at 2 kHz for beamforming)",
                "    Each channel: independent preamp + 24-bit ADC",
                "",
                "PREAMP & ADC CHAIN:",
                "  Preamp: Texas Instruments OPA1612 op-amp stages",
                "    Gain: 20-60 dB programmable, THD < 0.0009%",
                "    Input-referred noise: < 1 nV/sqrt(Hz)",
                "  ADC: Cirrus Logic CS5368 (8-channel, 24-bit,",
                "    up to 216 kHz, 111 dB dynamic range)",
                "    or Texas Instruments PCM1864 (4-channel)",
                "  Anti-aliasing filter: 2nd-order Bessel at 20 kHz",
                "",
                "PROCESSING UNIT:",
                "  Primary: Raspberry Pi 5 (8GB) or Jetson Orin Nano",
                "    Pi 5: 2.4 GHz quad-core, 8GB LPDDR4X, < 8W",
                "    Jetson: 1024 CUDA cores, GPU acceleration for FFT",
                "  Audio I/O: Pi-specific HAT or USB interface",
                "    Pi: HiFiBerry DAC+ ADC Pro (192 kHz/24-bit)",
                "    Jetson: External USB audio (Focusrite Scarlett)",
                "  Storage: 64GB microSD (OS) + 128GB USB SSD (recordings)",
                "",
                "POWER SYSTEM:",
                "  Battery: 4x 18650 Li-ion cells (3400 mAh each)",
                "    14.8V nominal, ~13.6 Ah total capacity",
                "  DC-DC converters: 5V/5A (Pi), 12V/2A (preamp)",
                "  Power management: Texas Instruments BQ25895 charger",
                "  Runtime: 4-8 hours depending on processing load",
                "  Charging: USB-C PD 30W fast charge",
                "",
                "ENCLOSURE & MOUNTING:",
                "  Body: 3D-printed PETG/ABS pistol-grip housing",
                "    Weatherproof gasket seals (IP54 target)",
                "  Grip: Ergonomic pistol grip with trigger-style",
                "    record button + thumb joystick for focus control",
                "  Display: 5-inch HDMI LCD (800x480) mounted on side",
                "    or headless mode with Bluetooth phone app",
                "  Mounting: 1/4-20 tripod thread + shoulder strap",
                "  Weight target: 10-15 lbs fully assembled",
                "",
                "RECOMMENDED INSTRUMENTS & TOOLS:",
                "  Assembly: Torx driver set, soldering iron (fine tip),",
                "    hot glue gun, multimeter, caliper",
                "  Testing: SPL meter (e.g. Reed R8050 for calibration),",
                "    oscilloscope (for preamp tuning), audio analyzer",
                "    (e.g. Room EQ Wizard for frequency response)",
                "  Calibration: BSWA TECH 211 sound calibrator (94 dB,",
                "    1 kHz) for absolute SPL reference",
                "  Field testing: Wind meter (anemometer), tape measure",
                "    for known-distance source tests",
                "",
                "ESTIMATED COST:",
                "  Budget build (Pi 5 + Primo mic + basic preamp): ~$350",
                "  Mid-range (Pi 5 + DPA 4060 + HiFiBerry): ~$800",
                "  Premium (Jetson + DPA + CS5368 + carbon dish): ~$1800",
            ]),
            ("PROCESSING PIPELINE", [
                "Stage 1: ADAPTIVE WIND FILTER",
                "  Detects wind noise via low-frequency energy ratio.",
                "  Applies adaptive high-pass filter (300-500 Hz) when",
                "  wind is present. Cutoff scales with wind strength.",
                "  Toggle: W key.",
                "",
                "Stage 2: BANDPASS FILTER",
                "  Butterworth 8th-order bandpass isolating the voice",
                "  band (default 250-4800 Hz, adjustable via sliders).",
                "  Removes out-of-band noise and low-frequency rumble.",
                "",
                "Stage 3: SPECTRAL GATE",
                "  Soft tanh gate that attenuates frequency bins below a",
                "  noise-derived threshold. Prevents musical noise artifacts",
                "  via smooth transition. Toggle: G key.",
                "",
                "Stage 4: DE-REVERBERATION",
                "  Temporal spectral magnitude smoothing suppresses late",
                "  reflections and echo smearing. Improves clarity in",
                "  reverberant environments. Toggle: D key.",
                "",
                "Stage 5: SPECTRAL DECOHERENCE",
                "  Adaptive spectral subtraction with percentile-based",
                "  noise floor estimate. Preserves low-energy speech.",
                "  Strength parameter controls aggressiveness (0.3-0.9).",
                "",
                "Stage 6: WIENER FILTER ENHANCEMENT",
                "  Per-bin Wiener filter: H(f) = SNR(f) / (SNR(f) + 1).",
                "  Noise floor learned via exponential moving average on",
                "  non-voice frames. Gain floor 0.1 prevents musical noise.",
                "",
                "Stage 7: MULTI-BAND COMPRESSION",
                "  3-band compressor (low/mid/high crossover at 500 Hz /",
                "  4 kHz) with per-band thresholds, ratios, and makeup gain.",
                "  Dramatically improves distant/quiet speech clarity.",
                "  Toggle: C key. Gain reduction shown in METER mode.",
                "",
                "Stage 8: PRESENCE BOOST",
                "  Enhances 2-5 kHz consonant band where speech",
                "  intelligibility cues live (s, t, k, p sounds).",
                "  Toggle: V key.",
                "",
                "Stage 9: FORMANT ENHANCEMENT",
                "  Boosts F1 (~500 Hz) and F2 (~1500 Hz) vowel formants",
                "  using narrow bandpass filters. Improves vowel clarity",
                "  for distant/quiet speech where formants are weakened.",
                "  Toggle: T key.",
                "",
                "Stage 10: SPECTRAL SUPER-RESOLUTION",
                "  Detects fundamental frequency peaks in voice band and",
                "  reconstructs weakened harmonic series above them. Helps",
                "  recover high-frequency detail lost to air absorption",
                "  at distance. Toggle: X key.",
                "",
                "Stage 11: LOUDNESS NORMALIZATION",
                "  Fast-attack/slow-release RMS normalization targeting",
                "  -16 dB. Maintains consistent output level regardless",
                "  of source distance. Toggle: L key.",
                "",
                "A/B COMPARE: B key toggles raw vs processed output.",
                "",
                "DEEP LISTEN: K key engages maximum processing mode.",
                "  Boosts noise reduction, enhance strength, and de-reverb",
                "  while enabling all processing stages. For pulling",
                "  barely-audible distant speech out of heavy noise.",
                "",
                "VAD-GATED RECORDING: J key arms automatic recording",
                "  that starts when voice is detected and stops after",
                "  2 seconds of silence. Creates separate WAV + JSON",
                "  files per speech segment. Eliminates dead air.",
                "",
                "SNR IMPROVEMENT: Real-time measurement of processing",
                "  gain by comparing raw input SNR (voice-band energy",
                "  vs out-of-band noise) to processed output SNR.",
                "  Displayed in left panel, color-coded by improvement.",
                "",
                "HARMONIC CURSOR: Click on spectrum to place a frequency",
                "  cursor showing F0 + harmonics 2F0-5F0 with dB",
                "  readings at each harmonic frequency. Useful for",
                "  identifying fundamental pitch and harmonic structure.",
                "",
                "EVENT LOG: Rolling status feed in right panel showing",
                "  recent events (deep listen, VAD-rec, auto-adapt",
                "  changes, recording start/stop). 8-entry deque.",
                "",
                "AUTO-ADAPT: E key enables automatic environment detection",
                "  that analyzes SNR, wind, flatness, and distance to",
                "  auto-select the best preset every ~5 seconds.",
                "",
                "SELECTIVE TUNING:",
                "  Voice-band focus (250-4800 Hz default, adjustable).",
                "  Focus lock on a selected direction, suppressing others.",
            ]),
            ("DISTANCE ESTIMATION", [
                "Method: Inverse-square law on received signal level.",
                "  Each 6 dB drop ~ doubles distance from source.",
                "  Reference: ~65 dB for conversational voice at 1 m.",
                "",
                "  distance = 2 ^ ((dB_ref - dB_received) / 6)",
                "",
                "NOTE: Uses digital dBFS, not calibrated SPL.",
                "  Accuracy depends on mic sensitivity, preamp gain,",
                "  and source loudness. Treat as rough estimate only.",
                "  True distance requires SPL calibration.",
            ]),
            ("ADVANCED METRICS", [
                "A-Weighted dB: Perceptual loudness using A-weighting",
                "  curve, more accurate for human hearing sensitivity.",
                "",
                "Spectral Centroid: Weighted average frequency. Higher =",
                "  brighter/sharper, lower = duller/muffled.",
                "",
                "Spectral Flatness: 0 = pure tone, 1 = white noise.",
                "  Speech typically 0.1-0.4, noise 0.6-0.9.",
                "",
                "Speech Intelligibility Index (SII): Approximate measure",
                "  of speech comprehensibility based on band SNR and",
                "  spectral characteristics. 0-1 (higher = better).",
                "",
                "Voice Activity Detection (VAD): Multi-feature detector",
                "  using energy, spectral flatness, zero-crossing rate,",
                "  and voice-band energy ratio. Returns confidence 0-1.",
                "",
                "Wind Noise Detection: Identifies wind by high low-freq",
                "  energy (< 200 Hz) relative to total spectrum.",
                "",
                "Clipping Detection: Flags when output samples reach",
                "  digital full-scale (0.98 amplitude threshold).",
                "",
                "Peak Hold: Spectrum peak markers with slow decay for",
                "  identifying transient peaks and frequency content.",
                "",
                "Composite Confidence: Weighted blend of SII (40%),",
                "  VAD confidence (30%), and normalized SNR (30%).",
                "  Clipping forces score to 0. Single quality metric.",
                "",
                "Pitch Detection: Normalized autocorrelation in the",
                "  70-400 Hz range. Returns F0 and confidence. Useful",
                "  for voice characterization and gender estimation.",
                "",
                "Formant Enhancement: Boosts F1 (~500 Hz) and F2",
                "  (~1500 Hz) vowel resonances. Improves vowel clarity",
                "  for distant speech where formants are weakened.",
                "",
                "Spectral Super-Resolution: Detects fundamental peaks",
                "  and reconstructs weakened harmonic series. Recovers",
                "  high-frequency detail lost to air absorption at range.",
                "",
                "Adaptive Wind Filter: High-pass filter (300-500 Hz)",
                "  that activates when wind noise is detected. Cutoff",
                "  scales with wind strength for adaptive filtering.",
                "",
                "Loudness Normalization: Fast-attack/slow-release RMS",
                "  normalization targeting -16 dB. Consistent output",
                "  regardless of source distance.",
                "",
                "Auto-Adapt: Analyzes SNR, wind, flatness, and distance",
                "  every ~5 seconds to auto-select the best environment",
                "  preset. Toggle with E key.",
                "",
                "Processing Latency: Real-time measurement of DSP pipeline",
                "  execution time using high-resolution timer. Displayed",
                "  in topbar, meter mode, right panel, and saved in",
                "  recording metadata. Green <10ms, yellow <30ms, red >30ms.",
                "",
                "Headroom Indicator: Shows dB headroom before clipping",
                "  (0 dBFS). Green >6dB, yellow >3dB, red <3dB.",
                "",
                "Peak Hold Meters: VU/PPM peak markers with slow decay",
                "  (0.5 dB/frame after 30 frame hold). Standard PPM behavior",
                "  on both input and output meters.",
                "",
                "Level History Graph: Real-time dB level over time in the",
                "  right panel, color-coded by zone (green/yellow/red).",
                "",
                "Confidence History Graph: 200-sample rolling graph of",
                "  composite confidence score with 50% threshold line.",
                "  Shown in meter mode and right panel confidence bar.",
                "",
                "FPS Counter: Real-time frames-per-second display in",
                "  topbar and left panel. Green >50, yellow >30, red <30.",
                "",
                "Deep Listen (K): Engages maximum processing mode by",
                "  boosting noise reduction (+0.1), enhance strength",
                "  (+0.3), de-reverb (+0.1), and enabling all stages.",
                "  Designed for pulling barely-audible distant speech",
                "  from heavy noise environments.",
                "",
                "VAD-Gated Recording (J): Arms automatic recording that",
                "  starts when voice activity is detected and stops",
                "  after 2 seconds of silence. Each speech segment gets",
                "  its own WAV + JSON metadata file. Eliminates dead",
                "  air from recordings. Status shown in left panel.",
                "",
                "SNR Improvement: Real-time measurement comparing raw",
                "  input SNR (voice-band energy vs out-of-band noise)",
                "  to processed output SNR. Smoothed with 0.05 alpha.",
                "  Color-coded: green >5dB, yellow >0dB, red <0dB.",
                "  Saved in recording metadata as snr_improvement_db.",
                "",
                "Harmonic Cursor: Click on spectrum to place a frequency",
                "  cursor showing F0 + harmonics 2F0 through 5F0. Each",
                "  harmonic shows its frequency and dB level. Useful",
                "  for identifying fundamental pitch and verifying",
                "  harmonic reconstruction from super-resolution stage.",
                "",
                "Event Log: Rolling 8-entry status feed in right panel",
                "  showing recent events (deep listen, VAD-rec, auto-",
                "  adapt changes, recording start/stop). Most recent",
                "  entry highlighted, older entries dimmed.",
                "",
                "Topbar Indicators: Pulsing CLIP! warning on clipping,",
                "  green VOICE indicator when voice detected, WIND %%",
                "  when wind noise detected, auto-adapt reason text",
                "  when auto-adapt is active, noise learning indicator,",
                "  session timer, snapshot indicator, golden marker count.",
                "",
                "Noise Profile Learning (N): Captures ambient noise",
                "  spectrum for 2 seconds (120 frames) during silence.",
                "  Learned profile replaces adaptive noise floor in",
                "  spectral decoherence stage for more accurate",
                "  subtraction. Auto-stops after 120 frames. Pulsing",
                "  LEARNING NOISE indicator in topbar while active.",
                "",
                "Whisper Mode (U): Triples input gain and boosts noise",
                "  reduction (+0.15) and enhance strength (+0.5) for",
                "  capturing very quiet or distant whispers. Enables",
                "  compressor, loudness, and super-resolution. Reverses",
                "  gain on toggle off.",
                "",
                "Frequency Isolator (Y): Applies bandpass filter at",
                "  300-3000 Hz after loudness normalization to isolate",
                "  a specific frequency range. Useful for focusing on",
                "  particular voice characteristics or filtering out",
                "  unwanted high/low frequency content.",
                "",
                "Spectrum Snapshot (F2): Freezes current spectrum as",
                "  semi-transparent overlay for comparison with live",
                "  spectrum. Shows how noise/signal changes over time.",
                "  Press F2 again to clear. SNAPSHOT indicator in topbar.",
                "",
                "Golden Markers (F3): During recording, press F3 to",
                "  mark important moments. Timestamps saved in",
                "  recording JSON metadata as golden_markers array.",
                "  Count shown in topbar with asterisk during recording.",
                "",
                "Session Timer: Shows total elapsed time since app",
                "  start in topbar (minutes:seconds format).",
                "",
                "Automatic Gain Control (A): Fast adaptive gain that",
                "  targets 0.05 RMS level. Smoother than the base",
                "  auto-normalization (0.02 alpha vs 0.1 alpha). Stacks",
                "  on top of base input gain for extra loudness control.",
                "  Useful for sources with varying distance/volume.",
                "",
                "Activity Timeline: 600-entry rolling log of voice/",
                "  sound detection state shown as bar graph in right",
                "  panel. Green = voice detected, yellow = sound above",
                "  -50 dB, dim = silence. Shows detection patterns over",
                "  the session (~10 seconds at 60 fps).",
                "",
                "Dynamic Range: Real-time measurement of peak dB vs",
                "  smoothed average dB. Peak holds and decays at 0.1",
                "  dB/frame. Average smoothed at 0.05 alpha. Shows",
                "  dynamic range in left panel. Green >10 dB (wide",
                "  range), yellow >3 dB, dim <3 dB (compressed).",
                "",
                "3-Band EQ: Bass (20-200 Hz), mid (200-4000 Hz), and",
                "  treble (4000+ Hz) shelving filters. Adjustable from",
                "  -12 to +12 dB in 1 dB steps. Applied after frequency",
                "  isolator, before final output. Keys: [ ] bass,",
                "  ; ' mid, , . treble. Current values shown in left",
                "  panel as EQ B/M/T.",
                "",
                "Audio Replay (O): Captures last 6 seconds from ring",
                "  buffer and plays it through the audio output. Useful",
                "  for reviewing what was just heard without recording.",
                "  Requires monitor output to be enabled (M key).",
                "",
                "Spectrum CSV Export (F4): Exports current raw and clean",
                "  spectrum data to a CSV file in exports/ directory.",
                "  Columns: frequency_hz, raw_db, clean_db, gain_db.",
                "  Useful for post-analysis or importing into other tools.",
                "",
                "Clipboard Metrics (F5): Copies a formatted summary of",
                "  all current metrics to the system clipboard. Includes",
                "  distance, SNR, SII, confidence, pitch, dynamic range,",
                "  EQ settings, and all processing toggle states. Uses",
                "  Windows 'clip' command.",
                "",
                "Spectrogram Speed (PGUP/PGDN): Controls how fast the",
                "  spectrogram waterfall scrolls. PGUP slows down (min",
                "  0.2x), PGDN speeds up (max 3x). Default 1x. Affects",
                "  how many frames are appended per update cycle.",
                "",
                "Preset Save/Load: Press Shift+1-4 to save all current",
                "  settings (including EQ, AGC, freq isolator, all toggle",
                "  states) to a JSON file in presets/ directory. Press",
                "  1-4 to load: if a custom preset file exists it loads",
                "  that, otherwise falls back to built-in presets.",
                "",
                "THD+N: Total Harmonic Distortion + Noise measurement.",
                "  Computes ratio of non-fundamental energy (harmonics",
                "  2x-5x plus noise floor) to total spectral energy.",
                "  Shown as percentage. Updated when pitch is detected",
                "  (>70 Hz, confidence >0.2). Smoothed at 0.1 alpha.",
                "  Green <10%, yellow >10%. Useful for assessing audio",
                "  chain quality and detecting distortion artifacts.",
                "",
                "Source Classification: Real-time audio type detection",
                "  using spectral flatness, centroid, pitch confidence,",
                "  and VAD. Categories: speech, music, tone, noise,",
                "  silence. Uses majority vote over 30-frame rolling",
                "  window for stability. Shown in left panel with color",
                "  coding (green=speech, accent=music, cool=tone).",
                "",
                "RT60 Estimation: Approximate reverberation time by",
                "  measuring spectral energy decay in the voice band",
                "  over 25 frames (~0.4s) of spectrogram history.",
                "  Extrapolates decay rate to 60 dB. Smoothed at 0.05",
                "  alpha. Shown in ms. Green <800ms, yellow >800ms.",
                "  Rough estimate only -- true RT60 requires impulse",
                "  response measurement.",
                "",
                "Audio Quality Grade: Composite A-F letter grade from",
                "  weighted score: SNR (25%%), SII (25%%), THD+N (20%%),",
                "  confidence (20%%), RT60 penalty (10%%). Score 0-100,",
                "  smoothed at 0.1 alpha. A>=85, B>=70, C>=55, D>=40,",
                "  F<40. Shown in left panel with color coding. Useful",
                "  for quick at-a-glance audio chain assessment.",
                "",
                "EQ Reset (0): Press 0 to reset all three EQ bands",
                "  (bass/mid/treble) to 0 dB instantly.",
                "",
                "Recording Timeline: Visual progress bar shown in right",
                "  panel during recording. Displays elapsed time and",
                "  golden markers (F3) as accent-colored vertical lines",
                "  positioned proportionally along the timeline. Shows",
                "  marker count below the bar. Persists after recording",
                "  stops until new session begins.",
                "",
                "Spectral Peak Tracking: Real-time detection of top 3",
                "  spectral peaks using local maxima with minimum bin",
                "  separation. Peaks are smoothed at 0.7/0.3 alpha for",
                "  frequency and amplitude stability. Shown as triangle",
                "  markers on the spectrum plot with frequency labels,",
                "  and as P1/P2/P3 lines in the right panel AUDIO QUALITY",
                "  section. Useful for identifying dominant frequencies.",
                "",
                "Level Distribution Histogram: Long-term dB level",
                "  histogram with 30 bins from -60 to 0 dB. Accumulates",
                "  over the entire session. Shown as bar graph in right",
                "  panel with color coding (red=quiet, yellow=moderate,",
                "  green=loud). Sample count shown below. Useful for",
                "  understanding typical signal levels over time.",
            ]),
            ("DIRECTIONAL ANALYSIS", [
                "36 bins mapping voice-band frequency sub-bands to a",
                "circular display for visualization.",
                "",
                "NOTE: With a SINGLE microphone, true direction-of-arrival",
                "estimation is NOT possible. This shows spectral energy",
                "distribution, not actual sound direction. Real direction",
                "estimation requires a microphone array (2+ mics).",
                "",
                "Display: Radar-style compass with energy distribution,",
                "sweep line animation, and history strip.",
                "Peak bin is highlighted with compass heading.",
            ]),
            ("PERFORMANCE TARGETS", [
                "DSP Latency: < 10 ms target (green), < 30 ms acceptable",
                "  Measured per-frame with time.perf_counter() around the",
                "  11-stage pipeline. Saved in recording JSON metadata.",
                "Display: 60 fps target with numpy-vectorized rendering",
                "SNR improvement: 10-20 dB with full processing pipeline",
                "  Real-time measurement saved as snr_improvement_db.",
                "Voice band: 250-4800 Hz (adjustable via sliders)",
                "Sliders: 7 total (noise, enhance, band low/high,",
                "  wind filter, de-reverb, compressor threshold)",
                "Sample rate: 44100 Hz, 32-bit float32",
                "FFT: 2048-point with Hann window",
                "Buffer: 6 seconds ring buffer for context",
                "Processing: 11-stage pipeline at ~60 fps display",
                "Presets: indoor / outdoor / distant / noisy (keys 1-4",
                "  or click preset buttons in left panel)",
                "Deep Listen: K key for max processing mode",
                "VAD-Recording: J key for voice-activated recording",
                "Noise Profile: N key for ambient noise learning (2s)",
                "Whisper Mode: U key for 3x gain on quiet sources",
                "Freq Isolator: Y key for 300-3000 Hz band isolation",
                "Spectrum Avg: Z key to toggle time-averaged spectrum display",
                "Spectrum Snapshot: F2 to freeze/clear spectrum overlay",
                "Golden Markers: F3 to mark moments during recording",
                "AGC: A key for fast automatic gain control",
                "3-Band EQ: [ ] ; ' , . for bass/mid/treble (+/-1 dB)",
                "Audio Replay: O key to replay last 6 seconds",
                "Activity Timeline: 600-entry voice/sound detection log",
                "Dynamic Range: real-time peak vs average dB measurement",
                "CSV Export: F4 to export spectrum data to CSV file",
                "Clipboard: F5 to copy all metrics to clipboard",
                "Spectrogram Speed: PGUP/PGDN to control scroll speed",
                "Preset Save/Load: Shift+1-4 save, 1-4 load (custom/built-in)",
                "THD+N: real-time total harmonic distortion + noise %",
                "Source Class: speech/music/tone/noise/silence detection",
                "RT60: approximate reverberation time estimation in ms",
                "Quality Grade: A-F composite score (SNR/SII/THD/conf/RT60)",
                "EQ Reset: 0 key to reset all EQ bands to 0 dB",
                "Recording Timeline: visual progress bar with golden markers",
                "Spectral Peaks: top 3 real-time peak frequency tracking",
                "Level Histogram: long-term dB distribution over session",
                "Session Stats: min/max/avg SNR, dB, confidence over session",
                "VU Meter: top bar level meter with peak hold indicator",
                "Spectrum Averaging: Z key for time-averaged spectrum display",
                "Recording: WAV + JSON metadata sidecar with full metrics",
                "  (distance, SNR, SII, VAD, confidence, pitch, latency,",
                "   SNR improvement, all processing params, preset,",
                "   focus lock, algorithm, deep_listen, vad_record)",
            ]),
            ("LEGAL & ETHICAL", [
                "Idaho one-party consent: recording only when you are",
                "a participant in the conversation, or for public-domain",
                "sounds with no privacy expectation.",
                "",
                "No automatic phone signal decoding.",
                "Encrypted selective capture with metadata (direction,",
                "estimated distance, AI confidence).",
            ]),
            ("CONTROLS SUMMARY", [
                "TAB / click    cycle SPECTRUM / DIRECTION / FILTER / METER",
                "P / SPACE      pause / resume processing",
                "R              reset sliders to defaults",
                "M              toggle audio monitor (hear enhanced output)",
                "UP / DOWN      adjust monitor volume",
                "S              start / stop recording to WAV file",
                "F              toggle focus lock on peak direction",
                "B              A/B compare (raw vs processed output)",
                "G              toggle spectral gate",
                "D              toggle de-reverberation",
                "C              toggle multi-band compressor",
                "V              toggle presence boost",
                "T              toggle formant enhancement",
                "X              toggle spectral super-resolution",
                "L              toggle loudness normalization",
                "W              toggle adaptive wind filter",
                "E              toggle auto environment adaptation",
                "K              toggle deep listen (max processing)",
                "J              toggle VAD-gated recording",
                "N              toggle noise profile learning",
                "U              toggle whisper mode (3x gain)",
                "Y              toggle frequency isolator (300-3000 Hz)",
                "Z              toggle spectrum averaging (time-averaged)",
                "F2             spectrum snapshot (freeze/clear)",
                "F3             golden marker (mark during recording)",
                "A              toggle automatic gain control (AGC)",
                "O              replay last 6 seconds of audio",
                "EQ             [ ] bass  ; ' mid  , . treble  0 reset (+/-1 dB)",
                "F4             export spectrum data to CSV file",
                "F5             copy current metrics to clipboard",
                "PGUP/PGDN      spectrogram scroll speed (0.2x-3x)",
                "Shift+1-4      save current settings to preset file",
                "1-4            load preset (custom if exists, else built-in)",
                "H              toggle this help panel",
                "I              toggle full info (this panel)",
                "ESC            quit",
                "",
                "Left panel sliders (drag to adjust):",
                "  Noise Reduction  -  spectral subtraction strength",
                "  Enhance Strength -  Wiener filter output gain",
                "  Voice Band Low   -  low cutoff frequency",
                "  Voice Band High  -  high cutoff frequency",
                "  Wind Filter      -  wind HPF cutoff scaling",
                "  De-Reverb        -  spectral smoothing strength",
                "  Comp Threshold   -  mid-band compressor threshold",
                "  EQ Bass          -  bass shelving gain (-12 to +12 dB)",
                "  EQ Mid           -  mid shelving gain (-12 to +12 dB)",
                "  EQ Treble        -  treble shelving gain (-12 to +12 dB)",
                "",
                "Spectrum mode: click on spectrum for harmonic cursor",
                "  Shows F0 + harmonics 2F0-5F0 with dB readings",
                "  dB scale on Y-axis, frequency axis labels",
                "  Pitch overlay (F0 + harmonics) when voice detected",
                "  Noise floor estimate (dashed line)",
                "  Confidence gauge bar (top-right)",
                "  Spectral gain reduction overlay (red shading)",
                "  Spectrogram frequency axis on left side",
                "  Spectrogram color bar legend (right side)",
                "Quality border: view area border color = confidence",
                "Filter mode: RMS + inter-stage gain per stage",
                "Meter mode: dual input/output VU meters + peak hold",
                "  + headroom + GR marker + confidence history graph",
                "  + latency + feature toggle status bar (17 features)",
                "Blueprint mode: 3D hardware model (100+ mesh groups)",
                "  + perspective projection + edge rendering + leader labels",
                "  + drag=orbit, wheel=zoom, R=reset, L=labels",
                "  + E=exploded, W=wireframe, A=auto-rotate, F=focus, T=edges",
                "  + dish: 48x14 paraboloid + torus rim + hinges + mag locks",
                "  + focal mic: cage + grille + XLR pins + strain relief",
                "  + MEMS: 8x + PCB traces + solder pads + mounting screws",
                "  + preamp: PCB + shield + op-amps + ADC + caps + jacks",
                "  + CPU: PCB + SoC + RAM + eMMC + heatsink + fan + 40 GPIO",
                "  + USB-A x2 + USB-C + HDMI + Ethernet + SD + LEDs + screws",
                "  + display: bezel + LCD + touch + FPC + driver + backlight",
                "  + power: 18650s + BMS + USB-C PD + switch + LEDs + gauge",
                "  + grip: trigger + spring + dial + buttons + tripod + strap",
                "  + enclosure: rails + vents + gasket + GPS + IMU + temp",
                "  + cables: XLR + I2S + HDMI + power + GPS + ground",
                "Right panel: SNR/dB/confidence history graphs",
                "  + confidence bar + auto-adapt status + event log",
                "  + activity timeline (voice/sound detection graph)",
                "Topbar: FPS/latency + CLIP! + VOICE + WIND%% + auto-adapt",
            ]),
        ]

        clip = pygame.Rect(x + 16, y + 46, w - 32, h - 60)
        self.screen.set_clip(clip)
        yy = y + 50 - self.info_scroll
        maxpx = w - 60
        total = 0
        for head, lines in sections:
            if y + 30 < yy < y + h:
                self.screen.blit(self.fb.render(head, True, C_WARN),
                                 (x + 24, yy))
            yy += 26
            total += 26
            for ln in lines:
                for wl in wrap_text(self.font, ln, maxpx):
                    if y + 30 < yy < y + h:
                        self.screen.blit(self.font.render(wl, True, C_TEXT),
                                         (x + 30, yy))
                    yy += 19
                    total += 19
            yy += 12
            total += 12
        self.screen.set_clip(None)
        self.info_scroll = max(0, min(self.info_scroll,
                                       max(0, total - (h - 70))))

    # ---- MAIN LOOP -------------------------------------------------------

    def run(self):
        _print_banner(self.capture.available, self.capture.device_name)
        while self.running:
            dt = min(0.05, self.clock.tick(60) / 1000.0)
            self.handle_events(dt)
            self.update(dt)
            self.draw()
            # FPS counter
            self._fps_frames += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self._fps = self._fps_frames / (now - self._fps_timer)
                self._fps_frames = 0
                self._fps_timer = now
        self.capture.close()
        pygame.quit()


# =============================================================================
# SECTION 8 -- STARTUP
# =============================================================================

def _print_banner(audio_ok, device_name=""):
    print("=" * 70)
    print(" HEARNOCULAR  --  Directional Hearing Digital Twin")
    print("=" * 70)
    print(" Modes (TAB):  SPECTRUM  |  DIRECTION  |  FILTER  |  METER  |  BLUEPRINT")
    print()
    if audio_ok:
        print(" Audio: LIVE INPUT (PyAudio %d Hz, %d-ch)" % (
            SAMPLE_RATE, CHANNELS))
        print("        Device: %s" % device_name)
    else:
        print(" Audio: NO INPUT DEVICE (no microphone detected)")
        print("        All data panels will show NO INPUT until a mic is plugged in.")
    print()
    print(" Pipeline:     11-stage DSP")
    print("   Wind HPF -> Bandpass -> Spectral Gate -> De-Reverb -> Decoherence")
    print("   -> Wiener -> Compressor -> Presence -> Formant -> Super-Res -> Loudness")
    print()
    print(" Metrics:      A-weighted dB, centroid, flatness, SII, VAD,")
    print("               wind detection, clipping, confidence, pitch")
    print(" Direction:    %d-bin spectral energy map (single-mic, NOT true direction)" % NUM_DIRECTIONS)
    print(" Distance:     Inverse-square law (rough estimate, dBFS not calibrated SPL)")
    print()
    print(" Controls:")
    print("   TAB  cycle mode      H  help    I  info    P  pause    R  reset  ESC  close/quit")
    print("   M  monitor audio     S  record  F  focus   B  A/B     E  auto-adapt")
    print("   G  gate  D  dereverb C  compress  V  presence  T  formant")
    print("   X  superres  L  loudness  W  wind  1-4  presets")
    print("   K  deep listen  J  VAD-record  UP/DN  volume")
    print("   N  noise profile  U  whisper mode  Y  freq isolator  Z  spec-avg")
    print("   F2 spectrum snapshot  F3 golden marker")
    print("   A  AGC  O  replay 6s  EQ: [ ] bass ; ' mid , . treble  0 reset")
    print("   F4 CSV export  F5 clipboard metrics  PGUP/DN spectro speed")
    print("   Shift+1-4 save preset  1-4 load preset (custom or built-in)")
    print("   Left panel: click presets, toggle auto-adapt, drag sliders")
    print("=" * 70)


def main():
    App().run()


if __name__ == "__main__":
    main()