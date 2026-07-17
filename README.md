# HEARNOCULAR
## Directional Hearing & Distance Perception Digital Twin

A 100% standalone Python application that captures, processes, visualizes, and enhances distant audio in real time using an 11-stage DSP pipeline. Built as the software counterpart to the "AetherEar" parabolic directional-hearing concept device. Includes a fully interactive 3D hardware blueprint mode with perspective projection, exploded view, wireframe, auto-rotate, and per-part focus highlighting -- modeling 100+ mesh groups across 10 major hardware components.

200m as the effective range for usable voice detection, with ~500m as the outer limit.
---

## Quick Start

```bash
python Hearnocular.py
```

Dependencies (auto-installed on first run): `numpy`, `pygame`, `scipy`, `pyaudio`, `soundfile`

---

## Five Display Modes (cycle with TAB)

### 1. SPECTRUM
Real-time frequency spectrum + spectrogram waterfall. Watch the voice band light up as sound arrives. The enhancement pipeline runs live with input vs output waveforms shown side-by-side.

**Features:**
- dB scale on Y-axis, peak hold markers
- Frequency cursor (click to read frequency + harmonics F0-F5)
- Clipping indicator, confidence gauge bar
- Noise floor estimate (dashed line)
- Pitch overlay (F0 line + harmonics) when voice detected
- Spectral gain reduction overlay (red shading)
- Spectrogram waterfall with frequency axis labels and color bar legend
- Voice band highlight
- Hover mouse over spectrum for live freq/dB readout
- Spectrum averaging toggle (Z key) for stable display
- Spectrum snapshot (F2) for frozen comparison overlay

### 2. DIRECTION

Spectral energy map. Voice-band frequency sub-bands mapped to a 360-degree compass for visualization.

> **NOTE:** Single microphone -- NOT true direction-of-arrival. Shows voice-band frequency distribution on a compass.

**Features:** Distance rings, confidence ring, pitch detection, A-weighted dB readout, distance estimation (rough, needs SPL calibration).

### 3. FILTER
The full 11-stage processing pipeline visualized stage-by-stage. Each stage shows its waveform, mini spectrum, RMS level, and inter-stage gain/attenuation indicator.

**Pipeline:** Raw -> Wind -> Bandpass -> Gate -> DeReverb -> Decoherence -> Wiener -> Compressor -> Presence -> Formant -> SuperRes -> Output

### 4. METER
Professional audio metering with dual input/output VU/PPM meters.

**Features:**
- dB scale with 3-color zones (green/yellow/red)
- Peak hold with slow decay
- Total gain reduction marker + headroom indicator
- 3-band gain reduction meters (LOW / MID / HIGH)
- Quality metrics panel (A-weighted dB, centroid, flatness, SII, VAD, wind, clipping, confidence, pitch, latency)
- Confidence history graph
- Feature toggle status bar (17 features)

### 5. BLUEPRINT
3D rotatable hardware model of the physical Hearnocular device. Software-rendered with perspective projection, Lambert shading, backface culling, depth-sorted alpha blending, edge rendering, and 3D label projection with leader lines (no external 3D engine needed).

**Controls:**
- **Drag** -- orbit camera around model
- **Shift+Drag** -- pan
- **Wheel / buttons 4/5** -- zoom in/out
- **R** -- reset view (auto-frame)
- **L** -- toggle component labels (with leader lines)
- **E** -- toggle exploded view (parts separate for inspection)
- **W** -- toggle wireframe mode
- **A** -- toggle auto-rotate
- **F** -- cycle focus/highlight individual parts (dims others)
- **T** -- toggle edge rendering

**Components modeled (10 major parts, 100+ mesh groups):**
- **Parabolic Dish** -- 48x14 segment paraboloid, torus rim, 6 panel ridges with hinges, center hub + hex cap, 4 spider vanes with screws, 4 backing ribs, 6 magnetic locks, focus mark ring
- **Focal Microphone** -- torus suspension cage, 4 shock mount supports, high-res mic capsule, grille cap, translucent windscreen, XLR connector with 3 pins, cone strain relief, 4 mounting screws
- **MEMS Array** -- PCB annulus ring, PCB trace grid, 8 mic capsules with port holes + solder pads, ribbon cable, 4 mounting screws
- **Preamp + ADC** -- PCB with trace grid, shield can with mounting tabs, 3x OPA1612 op-amps with pin-1 dots, CS5368 ADC, 4 electrolytic caps with stripes, 2x pin headers, 2x input jacks with center pins, 4 corner screws
- **Processing Unit** -- PCB with traces, SoC with pin-1 dot, RAM, eMMC, Ethernet magnetics, 7 heatsink fins, fan with housing + 4 blades + hub, 40-pin GPIO header, 2x USB-A with inner inserts, USB-C, HDMI, SD card slot + card, Ethernet RJ45 jack, status + power LEDs, 4 standoffs, 4 PCB screws
- **Display** -- backing panel, bezel, glowing LCD, capacitive touch overlay, FPC ribbon cable, mount bracket with screws, driver IC, backlight LED strip
- **Power System** -- enclosure, 4x 18650 cells with + and - terminals + wrapper lines, BMS board with traces + 2 ICs + 3 caps, USB-C PD port, rocker power switch, 4 battery level LEDs, holder clips, 4 enclosure screws, battery gauge display
- **Pistol Grip** -- tapered 2-section body, pommel cap, trigger guard, REC trigger button, trigger spring torus, mode dial with indicator notch, 3 side buttons, 1/4-20 tripod mount with thread ring, strap loop with torus, 6 texture ridges, 3 assembly screws, wrist strap anchor
- **Enclosure/Frame** -- support rail, 2 translucent side panels, 5 ventilation slots, torus weather seal gasket, 6 frame screws, cable pass-through grommet, GPS antenna pad, IMU module, temperature sensor
- **Cables & Connectors** -- shielded mic cable with XLR plug, I2S ribbon with 2 connectors, display cable with HDMI plug, power bus with PWR plug, preamp power tap, ground strap, GPS antenna cable

**UI Overlays:**
- Scrollable component spec sheet (right panel, mouse wheel to scroll, with scrollbar)
- Parts count + mesh group count in panel header
- Signal flow diagram: Mics -> Preamp -> ADC -> DSP -> Output
- Cost tiers: Budget $350 / Mid-range $800 / Premium $1800
- Dimensions: dish size, stowed size, weight, runtime
- Mode status indicators (AUTO-ROTATE / WIREFRAME / EXPLODED / FOCUS)

---

## 11-Stage DSP Pipeline

| Stage | Name | Function |
|-------|------|----------|
| 1 | Wind HPF | Adaptive high-pass filter for wind noise |
| 2 | Bandpass | Voice-band filter (configurable low/high) |
| 3 | Spectral Gate | Wiener-filter spectral subtraction |
| 4 | De-Reverb | Reverberation tail suppression |
| 5 | Decoherence | Error decoherence for noise reduction |
| 6 | Wiener | Optimal Wiener filter enhancement |
| 7 | Compressor | 3-band multi-band compressor |
| 8 | Presence Boost | 2-5 kHz consonant band enhancement |
| 9 | Formant | Formant frequency enhancement |
| 10 | Super-Resolution | Spectral super-resolution upscaling |
| 11 | Loudness | Loudness normalization |

---

## Advanced Metrics

- **A-weighted dB** -- perceptually-weighted sound level
- **Spectral centroid** -- brightness/timbre indicator
- **Spectral flatness** -- noise vs tonal content
- **SII** -- Speech Intelligibility Index
- **VAD** -- Voice Activity Detection
- **Wind detection** -- adaptive wind noise flag
- **Clipping detection** -- real-time clip warning
- **Confidence score** -- composite signal quality metric
- **Pitch detection** -- fundamental frequency (F0) via autocorrelation
- **THD+N** -- Total Harmonic Distortion + Noise
- **RT60** -- Reverberation time estimation
- **Audio quality grade** -- A-F composite score
- **SNR improvement** -- real-time processing gain in dB
- **Processing latency** -- DSP pipeline execution time

---

## Controls Reference

### Navigation
| Key | Action |
|-----|--------|
| TAB | Cycle mode (SPECTRUM / DIRECTION / FILTER / METER / BLUEPRINT) |
| H | Help overlay |
| I | Full info specification panel |
| ESC | Close overlay / quit (closes help/info first) |
| P / SPACE | Pause / resume processing |
| R | Reset sliders (blueprint: reset 3D view) |
| L | Toggle labels (blueprint mode) |
| E | Toggle exploded view (blueprint mode) |
| W | Toggle wireframe (blueprint mode) |
| A | Toggle auto-rotate (blueprint mode) |
| F | Cycle focus/highlight part (blueprint mode) |
| T | Toggle edge rendering (blueprint mode) |

### Audio
| Key | Action |
|-----|--------|
| M | Toggle audio monitor (hear enhanced output) |
| UP / DOWN | Adjust monitor volume |
| S | Start / stop recording to WAV |
| J | VAD-gated recording (auto-record on voice) |
| O | Replay last 6 seconds |
| F | Toggle focus lock on peak direction |

### Processing Toggles
| Key | Feature |
|-----|---------|
| G | Spectral gate |
| D | De-reverberation |
| C | Multi-band compressor |
| V | Presence boost |
| T | Formant enhancement |
| X | Spectral super-resolution |
| L | Loudness normalization |
| W | Adaptive wind filter |
| E | Auto environment adaptation |
| A | Automatic Gain Control (AGC) |
| K | Deep Listen (max all stages) |
| N | Noise profile learning |
| U | Whisper mode (3x gain) |
| Y | Frequency isolator (300-3000 Hz) |
| Z | Spectrum averaging |
| B | A/B compare (raw vs processed) |

### EQ
| Key | Action |
|-----|--------|
| [ / ] | Bass -/+ |
| ; / ' | Mid -/+ |
| , / . | Treble -/+ |
| 0 | Reset all EQ to 0 dB |

### Presets
| Key | Action |
|-----|--------|
| 1-4 | Load preset (custom if exists, else built-in) |
| Shift+1-4 | Save current settings to preset file |

### Data Export
| Key | Action |
|-----|--------|
| F2 | Spectrum snapshot (freeze for comparison) |
| F3 | Golden marker (mark moment during recording) |
| F4 | Export spectrum data to CSV |
| F5 | Copy all metrics to clipboard |
| PGUP / PGDN | Spectrogram scroll speed (0.2x to 3x) |

### Blueprint 3D
| Input | Action |
|-------|--------|
| Drag | Orbit camera |
| Shift+Drag | Pan |
| Wheel | Zoom |
| R | Reset view |
| L | Toggle labels |

---

## Left Panel (Clickable)

- **Preset buttons** -- indoor / outdoor / distant / noisy
- **Auto-Adapt toggle** -- click or press E
- **10 draggable sliders** -- noise reduction, enhance, voice band low/high, wind filter strength, de-reverb strength, compressor threshold, EQ bass/mid/treble

---

## Session Features

- **Session statistics** -- min/max/avg SNR, dB, confidence over entire session
- **VU meter** -- top bar level indicator with peak hold and color zones
- **Spectral peak tracking** -- top 3 real-time peaks with frequency/dB display
- **Level distribution histogram** -- long-term dB level distribution
- **Activity timeline** -- 600-entry voice/sound detection bar graph
- **Event log** -- rolling status feed in right panel
- **Recording timeline** -- visual progress bar with golden markers
- **SNR/dB/confidence history graphs** -- rolling time-series in right panel

---

## Recording

- **WAV + JSON metadata** -- each recording creates a WAV file and a JSON sidecar with all metrics, processing parameters, latency, SNR improvement, session stats, and golden marker timestamps
- **VAD-gated recording (J)** -- auto-records on voice detection, stops after 2s silence, creates per-segment WAV + JSON files
- **Golden markers (F3)** -- mark important moments during recording, saved in JSON metadata

---

## Hardware BOM (Bill of Materials)

### Reflector Assembly
- Parabolic dish: 22-24 inch carbon fiber composite, 6-panel deployable
- Surface accuracy: <0.5mm RMS, weight target <3 lbs

### Microphone Array
- Focal mic: Primo EM-272Z1 (or DPA 4060/4061), <14 dB-A self-noise
- MEMS array: 8x InvenSense ICS-43434, 85mm diameter circle

### Preamp & ADC
- TI OPA1612 op-amp, 20-60 dB programmable gain, THD <0.0009%
- Cirrus Logic CS5368 (8-ch, 24-bit, 216 kHz, 111 dB DR)

### Processing
- Raspberry Pi 5 (8GB) or Jetson Orin Nano
- HiFiBerry DAC+ ADC Pro (Pi) or USB interface (Jetson)

### Power
- 4x 18650 Li-ion (3400 mAh each), 14.8V, ~13.6 Ah
- 4-8 hour runtime, USB-C PD 30W fast charge

### Enclosure
- 3D-printed PETG/ABS pistol-grip, IP54 weatherproof
- 5-inch HDMI LCD (800x480), 1/4-20 tripod + shoulder strap
- Weight target: 10-15 lbs fully assembled

### Cost Tiers
| Tier | Components | Price |
|------|-----------|-------|
| Budget | Pi 5 + Primo mic + basic preamp | ~$350 |
| Mid-range | Pi 5 + DPA 4060 + HiFiBerry | ~$800 |
| Premium | Jetson + DPA + CS5368 + carbon dish | ~$1800 |

---

## Project Structure

```
Hearnocular/
  Hearnocular.py        Main application (standalone monolith)
  Goal                  Project goals and requirements document
  README.md             This file
  ReferenceCode/        Reference programs from related projects
  presets/              Custom preset JSON files (created on save)
  __pycache__/          Python bytecode cache
```

---

## Technical Notes

- **Audio:** PyAudio 44100 Hz, 1-channel input
- **FFT:** scipy.fft for spectral analysis
- **3D Engine:** Custom software renderer (no OpenGL dependency) with Lambert flat shading, backface culling, and depth-sorted alpha blending
- **UI:** pygame-based custom UI with panels, sliders, clickable buttons, and drag interactions
- **Clipboard:** Windows `clip` command for F5 metrics copy
- **Export:** CSV via numpy, JSON via standard library

---

## Legal & Ethical

Recording should only occur when you are a participant (Idaho one-party consent law) or for public domain sounds with no privacy expectation. No automatic phone signal decoding or unauthorized interception features.
