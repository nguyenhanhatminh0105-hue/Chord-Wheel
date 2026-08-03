# Chord Wheel

An Electron desktop app that lets you explore and play chords interactively. Click the wheel to trigger notes, or enable the webcam and use hand gestures via MediaPipe to play hands-free.

## Features

- **Interactive chord wheel** — click any slice to play a chord
- **Hand tracking** — optional webcam mode using MediaPipe; point to play
- **9 scales** — Major, Minor, Pentatonic, Blues, Dorian, Phrygian, Lydian, Mixolydian, Harmonic Minor
- **10 chord qualities** — maj, min, dim, aug, maj7, min7, dom7, sus2, sus4, m7b5
- **Built-in synthesizer** — sine, triangle, sawtooth, or square wave with attack/release envelope
- **Reverb** and **volume** controls
- **Note labeling** — Sharp, Flat, or Solfège
- **Snap mode** — lock to scale tones only

## Getting Started

```bash
npm install
npm start
```

Requires Node.js and a webcam (optional, for hand tracking).

On Windows, `launch.vbs` (which calls `launch.bat`) offers a double-click, no-terminal way to start the app once the repo path inside `launch.bat` is updated to match your local checkout.
