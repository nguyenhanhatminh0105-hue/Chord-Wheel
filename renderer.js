// ── renderer.js ──────────────────────────────────────────────────────────────
'use strict';

// The same page runs as the Electron app and as a plain web page. Only the
// window chrome and the way the hand-tracking model is loaded differ.
const isElectron = typeof process === 'object' && !!(process.versions && process.versions.electron);
document.body.classList.add(isElectron ? 'desktop' : 'web');
document.getElementById('close-btn').addEventListener('click', () => {
  window.close();
});

// ── Constants ────────────────────────────────────────────────────────────────
// Notes, scales and chord spelling live in theory.js (unit-tested under Node).
const theory = window.ChordTheory;
const { QUALITIES, midiToFreq } = theory;

// Computer-keyboard layout: the home row is the white keys, the row above the black ones.
const NOTE_KEYS = ['a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j'];
const QUALITY_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'];

// ── State ────────────────────────────────────────────────────────────────────
let selectedNote = -1;      // 0-11, -1 = none
let selectedQuality = 0;    // index into QUALITIES
let isPlaying = false;
let snapEnabled = false;
let reverbEnabled = true;
let camEnabled = false;
let labelMode = 'sharp';
let waveType = 'sine';
let attackTime = 0.05;
let releaseTime = 1.2;
let masterVolume = 0.6;
let currentScale = 'major';
let scaleRoot = 0; // C

// Wheel layout state
let noteHeld = false;  // mouse/touch currently held on note wheel
let qualHeld = false;  // mouse/touch currently held on quality wheel

// MediaPipe hand state
let leftHandPos  = null; // {x, y} normalized [0,1]
let rightHandPos = null;
let leftHandOnNote  = false;
let rightHandOnQual = false;

// ── Audio ────────────────────────────────────────────────────────────────────
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
const masterGain = audioCtx.createGain();
masterGain.gain.value = masterVolume;

// Convolver reverb
let convolver = null;
let dryGain = audioCtx.createGain();
let wetGain = audioCtx.createGain();
dryGain.gain.value = 0.7;
wetGain.gain.value = 0.5;
dryGain.connect(masterGain);
wetGain.connect(masterGain);
masterGain.connect(audioCtx.destination);

function buildReverb() {
  const len = audioCtx.sampleRate * 2.5;
  const buf = audioCtx.createBuffer(2, len, audioCtx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.5);
  }
  if (convolver) convolver.disconnect();
  convolver = audioCtx.createConvolver();
  convolver.buffer = buf;
  convolver.connect(wetGain);
}
buildReverb();

function getAudioDest() {
  return reverbEnabled ? convolver : dryGain;
}

let activeOscillators = [];

function startChord(noteIdx, qualIdx) {
  stopChord(true);
  if (noteIdx < 0) return;
  const { bass: bassMidi, tones } = theory.chordMidiNotes(noteIdx, QUALITIES[qualIdx]);

  const dest = getAudioDest();
  const chordGain = audioCtx.createGain();
  chordGain.gain.setValueAtTime(0, audioCtx.currentTime);
  chordGain.gain.linearRampToValueAtTime(0.7, audioCtx.currentTime + attackTime);
  chordGain.connect(dest);
  if (reverbEnabled) { chordGain.connect(dryGain); }

  const oscs = [];

  // Bass note
  const bassOsc = audioCtx.createOscillator();
  bassOsc.type = waveType;
  bassOsc.frequency.value = midiToFreq(bassMidi);
  const bassGain = audioCtx.createGain();
  bassGain.gain.value = 0.5;
  bassOsc.connect(bassGain);
  bassGain.connect(chordGain);
  bassOsc.start();
  oscs.push(bassOsc);

  // Chord tones
  for (const midi of tones) {
    const osc = audioCtx.createOscillator();
    osc.type = waveType;
    osc.frequency.value = midiToFreq(midi);
    osc.connect(chordGain);
    osc.start();
    oscs.push(osc);
  }

  activeOscillators = oscs;
  activeOscillators._gainNode = chordGain;
  isPlaying = true;
}

function stopChord(immediate = false) {
  if (!activeOscillators.length) return;
  const gain = activeOscillators._gainNode;
  if (gain) {
    const now = audioCtx.currentTime;
    gain.gain.cancelScheduledValues(now);
    gain.gain.setValueAtTime(gain.gain.value, now);
    if (immediate) {
      gain.gain.linearRampToValueAtTime(0, now + 0.02);
    } else {
      gain.gain.exponentialRampToValueAtTime(0.0001, now + releaseTime);
    }
    const oscs = [...activeOscillators];
    setTimeout(() => {
      oscs.forEach(o => { try { o.stop(); o.disconnect(); } catch(e){} });
    }, (immediate ? 50 : releaseTime * 1000 + 100));
  }
  activeOscillators = [];
  isPlaying = false;
}

// ── Canvas setup ─────────────────────────────────────────────────────────────
const wheelCanvas   = document.getElementById('wheel-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const rippleCanvas  = document.getElementById('ripple-canvas');
const wCtx  = wheelCanvas.getContext('2d');
const oCtx  = overlayCanvas.getContext('2d');
const rCtx  = rippleCanvas.getContext('2d');

let W = 0, H = 0;

function resize() {
  const main = document.getElementById('main');
  W = main.clientWidth;
  H = main.clientHeight;
  // Draw at the screen's pixel density so the wheels stay sharp.
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  [[wheelCanvas, wCtx], [overlayCanvas, oCtx], [rippleCanvas, rCtx]].forEach(([c, ctx]) => {
    c.width = Math.round(W * ratio); c.height = Math.round(H * ratio);
    c.style.width = W + 'px'; c.style.height = H + 'px';
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  });
  placeHUD();
  drawWheels();
}

window.addEventListener('resize', resize);

// ── Geometry helpers ─────────────────────────────────────────────────────────
// Wide windows put a wheel under each hand with the chord readout between
// them. Tall, narrow ones (a phone) stack the wheels instead.
function isStacked() { return W < 720 && H > W; }
function noteWheelCenter()  { return isStacked() ? { x: W * 0.5, y: H * 0.27 } : { x: W * 0.255, y: H * 0.52 }; }
function noteWheelRadius()  { return isStacked() ? Math.min(W * 0.42, H * 0.22) : Math.min(W * 0.19, H * 0.40); }
function qualWheelCenter()  { return isStacked() ? { x: W * 0.5, y: H * 0.81 } : { x: W * 0.775, y: H * 0.52 }; }
function qualWheelRadius()  { return isStacked() ? Math.min(W * 0.31, H * 0.165) : Math.min(W * 0.145, H * 0.30); }

// The readout sits in the gap the wheels leave.
function placeHUD() {
  const hud = document.getElementById('hud');
  const n = noteWheelCenter(), nr = noteWheelRadius();
  const q = qualWheelCenter(), qr = qualWheelRadius();
  document.body.classList.toggle('stacked', isStacked());
  if (isStacked()) {
    hud.style.left = '50%';
    hud.style.top = ((n.y + nr + q.y - qr - 26) / 2) + 'px';
    hud.style.width = Math.min(W - 32, 320) + 'px';
  } else {
    const left = n.x + nr, right = q.x - qr;
    hud.style.left = ((left + right) / 2) + 'px';
    hud.style.top = '50%';
    hud.style.width = Math.max(140, right - left - 32) + 'px';
  }
}
const INNER_RATIO = 0.28; // inner button relative to wheel radius
const SEGMENT_GAP = 0.022; // radians gap between segments

function segmentAngle(total) { return (2 * Math.PI) / total; }

function getNoteAtPoint(x, y) {
  const c = noteWheelCenter();
  const r = noteWheelRadius();
  const dx = x - c.x, dy = y - c.y;
  const dist = Math.sqrt(dx*dx + dy*dy);
  if (dist < r * INNER_RATIO) return -2; // inner button (OFF)
  if (dist > r) return -1;
  const angle = Math.atan2(dy, dx);
  const sa = segmentAngle(12);
  const offset = -Math.PI / 2 - sa / 2;
  let idx = Math.floor(((angle - offset + 2 * Math.PI * 3) % (2 * Math.PI)) / sa);
  idx = ((idx % 12) + 12) % 12;
  if (snapEnabled) {
    const scaleNotes = getScaleNotes();
    if (!scaleNotes.includes(idx)) return -1;
  }
  return idx;
}

function getQualAtPoint(x, y) {
  const c = qualWheelCenter();
  const r = qualWheelRadius();
  const dx = x - c.x, dy = y - c.y;
  const dist = Math.sqrt(dx*dx + dy*dy);
  if (dist < r * INNER_RATIO || dist > r) return -1;
  const angle = Math.atan2(dy, dx);
  const sa = segmentAngle(QUALITIES.length);
  const offset = -Math.PI / 2 - sa / 2;
  let idx = Math.floor(((angle - offset + 2 * Math.PI * 3) % (2 * Math.PI)) / sa);
  idx = ((idx % QUALITIES.length) + QUALITIES.length) % QUALITIES.length;
  return idx;
}

function getScaleNotes() {
  return theory.scaleNotes(currentScale, scaleRoot);
}

// ── Drawing ───────────────────────────────────────────────────────────────────
function getNoteName(idx) {
  return theory.noteName(idx, labelMode);
}

function drawSegment(ctx, cx, cy, r0, r1, startAngle, endAngle, fillColor, strokeColor, glowing) {
  ctx.save();
  if (glowing) {
    ctx.shadowColor = fillColor;
    ctx.shadowBlur = 18;
  }
  ctx.beginPath();
  ctx.arc(cx, cy, r1, startAngle + SEGMENT_GAP, endAngle - SEGMENT_GAP);
  ctx.arc(cx, cy, r0, endAngle - SEGMENT_GAP, startAngle + SEGMENT_GAP, true);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

// Labels are always upright, wherever they sit on the wheel.
function drawText(ctx, text, cx, cy, r, angle, color, size, weight = 'bold') {
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px 'Space Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, cx + r * Math.cos(angle), cy + r * Math.sin(angle));
  ctx.restore();
}

function drawArc(ctx, cx, cy, r, startAngle, endAngle, color, width) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle + SEGMENT_GAP, endAngle - SEGMENT_GAP);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

function drawWheelTitle(text, c, r) {
  wCtx.save();
  wCtx.fillStyle = '#8b90a8';
  wCtx.font = `12px 'Space Mono', monospace`;
  wCtx.textAlign = 'center';
  wCtx.textBaseline = 'alphabetic';
  wCtx.fillText(text, c.x, c.y - r - 14);
  wCtx.restore();
}

function drawWheels() {
  wCtx.clearRect(0, 0, W, H);
  // With the camera on, the wheels go translucent so your hands show through.
  wCtx.globalAlpha = camEnabled ? 0.62 : 1;

  const nc = noteWheelCenter();
  const nr = noteWheelRadius();
  const qc = qualWheelCenter();
  const qr = qualWheelRadius();
  const scaleNotes = getScaleNotes();
  const showKeys = !isStacked() && !camEnabled;

  // ── Note wheel ──
  const sa12 = segmentAngle(12);
  for (let i = 0; i < 12; i++) {
    const start = -Math.PI / 2 + i * sa12 - sa12 / 2;
    const end   = start + sa12;
    const mid   = (start + end) / 2;
    const active = i === selectedNote && isPlaying;
    const inScale = scaleNotes.includes(i);
    const isRoot = i === scaleRoot;
    const dimmed = snapEnabled && !inScale;

    const fill = active ? '#f97316' : dimmed ? '#101018' : inScale ? '#1d2233' : '#14141f';
    const stroke = active ? '#fbbf24' : dimmed ? '#1a1a26' : inScale ? '#343b55' : '#23233a';
    drawSegment(wCtx, nc.x, nc.y, nr * INNER_RATIO + 4, nr, start, end, fill, stroke, active);

    // Notes of the current scale carry an orange rim; the key note's is brighter.
    if (inScale && !active) {
      drawArc(wCtx, nc.x, nc.y, nr - 3, start, end, isRoot ? '#f97316' : '#f9731666', isRoot ? 4 : 3);
    }

    const textColor = active ? '#1a0d02' : dimmed ? '#3d4052' : inScale ? '#e8ebf4' : '#8b90a8';
    drawText(wCtx, getNoteName(i), nc.x, nc.y, nr * 0.72, mid, textColor, Math.max(12, nr * 0.105));
    if (showKeys && !dimmed) {
      drawText(wCtx, NOTE_KEYS[i].toUpperCase(), nc.x, nc.y, nr * 0.46, mid, active ? '#4a2206' : '#5f647d', Math.max(10, nr * 0.055), 'normal');
    }
  }

  // Inner OFF button
  wCtx.save();
  wCtx.beginPath();
  wCtx.arc(nc.x, nc.y, nr * INNER_RATIO, 0, Math.PI * 2);
  wCtx.fillStyle = isPlaying ? '#1a1a28' : '#0d0d0f';
  wCtx.fill();
  wCtx.strokeStyle = isPlaying ? '#4a4f6b' : '#2a2a40';
  wCtx.lineWidth = 1.5;
  wCtx.stroke();
  wCtx.fillStyle = isPlaying ? '#e8ebf4' : '#5f647d';
  wCtx.font = `bold ${Math.max(11, nr * 0.075)}px 'Space Mono', monospace`;
  wCtx.textAlign = 'center';
  wCtx.textBaseline = 'middle';
  wCtx.fillText('OFF', nc.x, nc.y);
  wCtx.restore();

  drawWheelTitle('Root note', nc, nr);

  // ── Quality wheel ──
  const saQ = segmentAngle(QUALITIES.length);
  for (let i = 0; i < QUALITIES.length; i++) {
    const start = -Math.PI / 2 + i * saQ - saQ / 2;
    const end   = start + saQ;
    const mid   = (start + end) / 2;
    const chosen = i === selectedQuality;
    const active = chosen && isPlaying;

    const fill = active ? '#a78bfa' : chosen ? '#2a2452' : '#15142a';
    const stroke = active ? '#c4b5fd' : chosen ? '#a78bfa' : '#29284a';
    drawSegment(wCtx, qc.x, qc.y, qr * INNER_RATIO + 3, qr, start, end, fill, stroke, active);

    const textColor = active ? '#140d2e' : chosen ? '#ede9fe' : '#a59bd6';
    drawText(wCtx, QUALITIES[i], qc.x, qc.y, qr * 0.74, mid, textColor, Math.max(11, qr * 0.105));
    if (showKeys) {
      drawText(wCtx, QUALITY_KEYS[i], qc.x, qc.y, qr * 0.46, mid, active ? '#3b2d7a' : '#5c5890', Math.max(10, qr * 0.07), 'normal');
    }
  }

  // Inner quality circle
  wCtx.save();
  wCtx.beginPath();
  wCtx.arc(qc.x, qc.y, qr * INNER_RATIO, 0, Math.PI * 2);
  wCtx.fillStyle = '#0d0d0f';
  wCtx.fill();
  wCtx.strokeStyle = '#29284a';
  wCtx.lineWidth = 1.5;
  wCtx.stroke();
  wCtx.restore();

  drawWheelTitle('Chord quality', qc, qr);
  wCtx.globalAlpha = 1;
}

// ── HUD update ────────────────────────────────────────────────────────────────
function updateHUD() {
  const chordNameEl   = document.getElementById('chord-name');
  const qualLabelEl   = document.getElementById('quality-label');
  const notePillsEl   = document.getElementById('note-pills');

  const idle = !isPlaying || selectedNote < 0;
  document.getElementById('hud').classList.toggle('idle', idle);
  if (idle) {
    chordNameEl.textContent = '';
    qualLabelEl.textContent = '';
    notePillsEl.innerHTML = '';
    return;
  }

  const root = getNoteName(selectedNote);
  const qual = QUALITIES[selectedQuality];
  chordNameEl.textContent = `${root} ${qual}`;
  qualLabelEl.textContent = theory.qualityFullName(qual);

  const noteIndices = theory.chordPitchClasses(selectedNote, qual);
  notePillsEl.innerHTML = noteIndices.map(ni => {
    const name = getNoteName(ni);
    return `<span class="note-pill active">${name}</span>`;
  }).join('');
}

// ── Ripples ───────────────────────────────────────────────────────────────────
const ripples = [];

function spawnRipple(x, y, color) {
  ripples.push({ x, y, r: 0, maxR: 60, alpha: 0.7, color });
}

function animRipples() {
  rCtx.clearRect(0, 0, W, H);
  for (let i = ripples.length - 1; i >= 0; i--) {
    const rp = ripples[i];
    rp.r += 2.5;
    rp.alpha -= 0.02;
    if (rp.alpha <= 0 || rp.r >= rp.maxR) { ripples.splice(i, 1); continue; }
    rCtx.save();
    rCtx.beginPath();
    rCtx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
    rCtx.strokeStyle = rp.color;
    rCtx.globalAlpha = rp.alpha;
    rCtx.lineWidth = 2;
    rCtx.stroke();
    rCtx.restore();
  }
  requestAnimationFrame(animRipples);
}
animRipples();

// ── Interaction logic ─────────────────────────────────────────────────────────
function onNotePress(noteIdx, x, y) {
  if (noteIdx === -2) { // OFF button
    if (isPlaying) {
      stopChord(false);
      selectedNote = -1;
      drawWheels();
      updateHUD();
    }
    return;
  }
  if (noteIdx < 0) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  if (noteIdx === selectedNote && isPlaying) {
    stopChord(false);
    drawWheels();
    updateHUD();
    return;
  }
  selectedNote = noteIdx;
  startChord(selectedNote, selectedQuality);
  spawnRipple(x, y, '#f97316');
  drawWheels();
  updateHUD();
}

function onNoteSlide(noteIdx, x, y) {
  if (noteIdx < 0 || noteIdx === -2) return;
  if (noteIdx === selectedNote) return;
  selectedNote = noteIdx;
  if (isPlaying) {
    startChord(selectedNote, selectedQuality);
    spawnRipple(x, y, '#f97316');
    drawWheels();
    updateHUD();
  }
}

function onQualPress(qualIdx, x, y) {
  if (qualIdx < 0) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  selectedQuality = qualIdx;
  if (isPlaying) {
    startChord(selectedNote, selectedQuality);
    spawnRipple(x, y, '#a78bfa');
    drawWheels();
    updateHUD();
  } else {
    drawWheels();
  }
}

function onQualSlide(qualIdx, x, y) {
  if (qualIdx < 0 || qualIdx === selectedQuality) return;
  selectedQuality = qualIdx;
  if (isPlaying) {
    startChord(selectedNote, selectedQuality);
    spawnRipple(x, y, '#a78bfa');
    drawWheels();
    updateHUD();
  } else {
    drawWheels();
  }
}

function onRelease() {
  if (isPlaying) {
    stopChord(false);
    drawWheels();
    updateHUD();
  }
}

// ── Mouse events ──────────────────────────────────────────────────────────────
wheelCanvas.addEventListener('mousedown', e => {
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const { offsetX: x, offsetY: y } = e;
  const ni = getNoteAtPoint(x, y);
  const qi = getQualAtPoint(x, y);
  if (ni !== -1) { noteHeld = true; onNotePress(ni, x, y); }
  else if (qi !== -1) { qualHeld = true; onQualPress(qi, x, y); }
});

wheelCanvas.addEventListener('mousemove', e => {
  const { offsetX: x, offsetY: y } = e;
  if (noteHeld) { const ni = getNoteAtPoint(x, y); onNoteSlide(ni, x, y); }
  if (qualHeld) { const qi = getQualAtPoint(x, y); onQualSlide(qi, x, y); }
});

wheelCanvas.addEventListener('mouseup', () => {
  if (noteHeld) onRelease();
  noteHeld = false; qualHeld = false;
});

wheelCanvas.addEventListener('mouseleave', () => {
  if (noteHeld || qualHeld) onRelease();
  noteHeld = false; qualHeld = false;
});

// ── Touch events ──────────────────────────────────────────────────────────────
wheelCanvas.addEventListener('touchstart', e => {
  e.preventDefault();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const t = e.touches[0];
  const rect = wheelCanvas.getBoundingClientRect();
  const x = t.clientX - rect.left, y = t.clientY - rect.top;
  const ni = getNoteAtPoint(x, y);
  const qi = getQualAtPoint(x, y);
  if (ni !== -1) { noteHeld = true; onNotePress(ni, x, y); }
  else if (qi !== -1) { qualHeld = true; onQualPress(qi, x, y); }
}, { passive: false });

wheelCanvas.addEventListener('touchmove', e => {
  e.preventDefault();
  const t = e.touches[0];
  const rect = wheelCanvas.getBoundingClientRect();
  const x = t.clientX - rect.left, y = t.clientY - rect.top;
  if (noteHeld) { const ni = getNoteAtPoint(x, y); onNoteSlide(ni, x, y); }
  if (qualHeld) { const qi = getQualAtPoint(x, y); onQualSlide(qi, x, y); }
}, { passive: false });

wheelCanvas.addEventListener('touchend', e => {
  e.preventDefault();
  if (noteHeld || qualHeld) onRelease();
  noteHeld = false; qualHeld = false;
}, { passive: false });

// ── Toolbar controls ──────────────────────────────────────────────────────────
const keySelect = document.getElementById('key-select');
function fillKeySelect() {
  keySelect.innerHTML = '';
  for (let i = 0; i < 12; i++) {
    const option = document.createElement('option');
    option.value = i;
    option.textContent = getNoteName(i);
    keySelect.appendChild(option);
  }
  keySelect.value = scaleRoot;
}
fillKeySelect();
keySelect.addEventListener('change', e => {
  scaleRoot = parseInt(e.target.value, 10);
  drawWheels();
});

document.getElementById('scale-select').addEventListener('change', e => {
  currentScale = e.target.value;
  drawWheels();
});

document.getElementById('label-select').addEventListener('change', e => {
  labelMode = e.target.value;
  fillKeySelect();
  drawWheels();
  updateHUD();
});

function setPressed(id, on) {
  document.getElementById(id).setAttribute('aria-pressed', String(on));
}

// A short message in the readout, for things the user needs to know about the camera.
function setNotice(text) {
  document.getElementById('notice').textContent = text;
}

document.getElementById('wave-select').addEventListener('change', e => {
  waveType = e.target.value;
});

document.getElementById('attack-slider').addEventListener('input', e => {
  attackTime = parseFloat(e.target.value);
});

document.getElementById('release-slider').addEventListener('input', e => {
  releaseTime = parseFloat(e.target.value);
});

document.getElementById('volume-slider').addEventListener('input', e => {
  masterVolume = parseFloat(e.target.value);
  masterGain.gain.setTargetAtTime(masterVolume, audioCtx.currentTime, 0.05);
});

document.getElementById('snap-btn').addEventListener('click', () => {
  snapEnabled = !snapEnabled;
  setPressed('snap-btn', snapEnabled);
  drawWheels();
});

document.getElementById('reverb-btn').addEventListener('click', () => {
  reverbEnabled = !reverbEnabled;
  setPressed('reverb-btn', reverbEnabled);
});

// ── Webcam ────────────────────────────────────────────────────────────────────
const videoEl = document.getElementById('video-bg');
let videoStream = null;

document.getElementById('cam-btn').addEventListener('click', async () => {
  camEnabled = !camEnabled;
  setPressed('cam-btn', camEnabled);

  if (camEnabled) {
    try {
      setNotice('Starting the camera…');
      videoStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      videoEl.srcObject = videoStream;
      videoEl.style.display = 'block';
      drawWheels();
      await initHandTracking();
    } catch (err) {
      console.error('Camera error:', err);
      camEnabled = false;
      setPressed('cam-btn', false);
      setNotice('The camera is not available. Check that this page is allowed to use it.');
      drawWheels();
    }
  } else {
    setNotice('');
    videoEl.style.display = 'none';
    if (videoStream) {
      videoStream.getTracks().forEach(t => t.stop());
      videoStream = null;
    }
    stopHandTracking();
    drawWheels();
  }
});

// ── MediaPipe Hands ───────────────────────────────────────────────────────────
let handsInstance = null;
let cameraInstance = null;
const blobUrlCache = {};  // reuse blob URLs across reinits

// Smoothing + grace-period state
const SMOOTH_ALPHA  = 0.35;  // EMA blend (0=frozen, 1=raw)
const GRACE_FRAMES  = 12;    // frames to hold last position after tracking loss
let smoothLeft  = null, smoothRight  = null;
let graceLeft   = 0,   graceRight   = 0;
let noHandsFrames = 0;       // consecutive frames with no hands (reinit watchdog)

const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/';

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error('Could not load ' + src));
    document.head.appendChild(script);
  });
}

const HANDS_OPTIONS = {
  maxNumHands: 2,
  modelComplexity: 1,       // full model — more robust to hand shape changes
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.3,
};

// In a browser the model and its files come from the CDN.
async function buildHandsInstanceWeb() {
  if (!window.Hands) await loadScript(MEDIAPIPE_CDN + 'hands.js');
  const h = new window.Hands({ locateFile: file => MEDIAPIPE_CDN + file });
  h.setOptions(HANDS_OPTIONS);
  h.onResults(onHandResults);
  await h.initialize();
  return h;
}

async function buildHandsInstance() {
  if (!isElectron) return buildHandsInstanceWeb();
  const { Hands } = require('@mediapipe/hands');
  const nodePath = require('path');
  const nodeFs   = require('fs');
  const mpDir    = nodePath.join(__dirname, 'node_modules/@mediapipe/hands');

  const locateFile = file => {
    if (blobUrlCache[file]) return blobUrlCache[file];
    const abs = nodePath.join(mpDir, file);
    try {
      const buf  = nodeFs.readFileSync(abs);
      const ext  = nodePath.extname(file);
      const mime = ext === '.wasm' ? 'application/wasm'
                 : ext === '.js'   ? 'application/javascript'
                 : 'application/octet-stream';
      const url = URL.createObjectURL(new Blob([buf], { type: mime }));
      blobUrlCache[file] = url;
      return url;
    } catch(e) {
      console.error('locateFile failed for', file, e.message);
      return file;
    }
  };

  // Hide process.versions.node so packed-assets loader uses XHR, not fs.readFile
  const savedNodeVer = process.versions.node;
  delete process.versions.node;
  const h = new Hands({ locateFile });
  h.setOptions(HANDS_OPTIONS);
  h.onResults(onHandResults);
  await h.initialize();
  process.versions.node = savedNodeVer;
  return h;
}

async function initHandTracking() {
  try {
    setNotice('Loading hand tracking…');
    handsInstance = await buildHandsInstance();
    if (!camEnabled) { stopHandTracking(); return; } // switched off while loading
    setNotice('Left hand picks the note, right hand picks the chord.');

    // rAF loop — sends each ready video frame to the model
    let active = true;
    let reiniting = false;
    async function frameLoop() {
      if (!active) return;
      if (videoEl.readyState >= 2) {
        try { await handsInstance.send({ image: videoEl }); } catch(e) {}
      }
      // Watchdog: after ~3 s with no hands, force a fresh Hands instance so
      // re-detection restarts cleanly when hands re-enter the frame
      if (noHandsFrames > 90 && !reiniting) {
        reiniting = true;
        noHandsFrames = 0;
        try {
          try { handsInstance.close(); } catch(e) {}
          handsInstance = await buildHandsInstance();
        } catch(e) { console.error('Hands reinit failed:', e); }
        reiniting = false;
      }
      requestAnimationFrame(frameLoop);
    }
    frameLoop();
    cameraInstance = { stop: () => { active = false; } };

  } catch (err) {
    console.error('MediaPipe init failed:', err && (err.message || String(err)));
    setNotice('Hand tracking could not be loaded. Clicking and the keyboard still work.');
  }
}

function stopHandTracking() {
  if (cameraInstance) { try { cameraInstance.stop(); } catch(e){} cameraInstance = null; }
  if (handsInstance)  { try { handsInstance.close(); } catch(e){} handsInstance = null; }
  oCtx.clearRect(0, 0, W, H);
  leftHandPos = null; rightHandPos = null;
  smoothLeft = null; smoothRight = null;
  graceLeft = 0; graceRight = 0; noHandsFrames = 0;
}

function onHandResults(results) {
  oCtx.clearRect(0, 0, W, H);

  let rawLeft = null, rawRight = null;

  if (results.multiHandLandmarks) {
    for (let h = 0; h < results.multiHandLandmarks.length; h++) {
      const landmarks  = results.multiHandLandmarks[h];
      const handedness = results.multiHandedness[h].label;

      const palmIdx = [0, 5, 9, 13, 17];
      const palmX = palmIdx.reduce((s, i) => s + landmarks[i].x, 0) / palmIdx.length;
      const palmY = palmIdx.reduce((s, i) => s + landmarks[i].y, 0) / palmIdx.length;
      const px = (1 - palmX) * W;
      const py = palmY * H;

      if (handedness === 'Right') rawLeft  = { x: px, y: py };
      else                         rawRight = { x: px, y: py };

      // Draw finger landmarks (raw)
      for (const lm of landmarks) {
        const lx = (1 - lm.x) * W, ly = lm.y * H;
        oCtx.save();
        oCtx.beginPath();
        oCtx.arc(lx, ly, 3, 0, Math.PI * 2);
        oCtx.fillStyle = handedness === 'Right' ? '#f9731644' : '#a78bfa44';
        oCtx.fill();
        oCtx.restore();
      }
    }
  }

  // EMA smoothing + grace period — left hand (orange, note wheel)
  if (rawLeft) {
    graceLeft = GRACE_FRAMES;
    if (!smoothLeft) smoothLeft = { ...rawLeft };
    else {
      smoothLeft.x = SMOOTH_ALPHA * rawLeft.x + (1 - SMOOTH_ALPHA) * smoothLeft.x;
      smoothLeft.y = SMOOTH_ALPHA * rawLeft.y + (1 - SMOOTH_ALPHA) * smoothLeft.y;
    }
  } else {
    graceLeft = Math.max(0, graceLeft - 1);
    if (graceLeft === 0) smoothLeft = null;
  }

  // EMA smoothing + grace period — right hand (purple, quality wheel)
  if (rawRight) {
    graceRight = GRACE_FRAMES;
    if (!smoothRight) smoothRight = { ...rawRight };
    else {
      smoothRight.x = SMOOTH_ALPHA * rawRight.x + (1 - SMOOTH_ALPHA) * smoothRight.x;
      smoothRight.y = SMOOTH_ALPHA * rawRight.y + (1 - SMOOTH_ALPHA) * smoothRight.y;
    }
  } else {
    graceRight = Math.max(0, graceRight - 1);
    if (graceRight === 0) smoothRight = null;
  }

  // Watchdog counter
  if (!smoothLeft && !smoothRight) noHandsFrames++;
  else noHandsFrames = 0;

  // Draw smoothed palm dots
  if (smoothLeft) {
    const { x: px, y: py } = smoothLeft;
    oCtx.save();
    oCtx.beginPath();
    oCtx.arc(px, py, 14, 0, Math.PI * 2);
    oCtx.fillStyle = '#f9731655';
    oCtx.fill();
    oCtx.strokeStyle = '#f97316';
    oCtx.lineWidth = 2.5;
    oCtx.stroke();
    oCtx.restore();
  }
  if (smoothRight) {
    const { x: px, y: py } = smoothRight;
    oCtx.save();
    oCtx.beginPath();
    oCtx.arc(px, py, 14, 0, Math.PI * 2);
    oCtx.fillStyle = '#a78bfa55';
    oCtx.fill();
    oCtx.strokeStyle = '#a78bfa';
    oCtx.lineWidth = 2.5;
    oCtx.stroke();
    oCtx.restore();
  }

  // Left hand controls note wheel
  if (smoothLeft) {
    const ni = getNoteAtPoint(smoothLeft.x, smoothLeft.y);
    if (ni >= 0) {
      if (!leftHandOnNote) {
        leftHandOnNote = true;
        onNotePress(ni, smoothLeft.x, smoothLeft.y);
      } else {
        onNoteSlide(ni, smoothLeft.x, smoothLeft.y);
      }
    } else {
      if (leftHandOnNote) { leftHandOnNote = false; onRelease(); }
    }
  } else {
    if (leftHandOnNote) { leftHandOnNote = false; onRelease(); }
  }

  // Right hand controls quality wheel
  if (smoothRight) {
    const qi = getQualAtPoint(smoothRight.x, smoothRight.y);
    if (qi >= 0) {
      if (!rightHandOnQual) {
        rightHandOnQual = true;
        onQualPress(qi, smoothRight.x, smoothRight.y);
      } else {
        onQualSlide(qi, smoothRight.x, smoothRight.y);
      }
    } else {
      rightHandOnQual = false;
    }
  } else {
    rightHandOnQual = false;
  }

  leftHandPos  = smoothLeft;
  rightHandPos = smoothRight;
}

// ── Computer keyboard ─────────────────────────────────────────────────────────
// Hold a note key to play; the number row changes the chord, even mid-note.
let heldNoteKey = null;

window.addEventListener('keydown', e => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  if (['SELECT', 'INPUT', 'BUTTON', 'A'].includes(document.activeElement.tagName)) return;
  const key = e.key.toLowerCase();
  const ni = NOTE_KEYS.indexOf(key);
  const qi = QUALITY_KEYS.indexOf(key);
  if (ni >= 0) {
    if (snapEnabled && !getScaleNotes().includes(ni)) return;
    heldNoteKey = key;
    const c = noteWheelCenter(), r = noteWheelRadius() * 0.72;
    const angle = -Math.PI / 2 + ni * segmentAngle(12);
    if (!(ni === selectedNote && isPlaying)) onNotePress(ni, c.x + r * Math.cos(angle), c.y + r * Math.sin(angle));
  } else if (qi >= 0) {
    const c = qualWheelCenter(), r = qualWheelRadius() * 0.74;
    const angle = -Math.PI / 2 + qi * segmentAngle(QUALITIES.length);
    onQualPress(qi, c.x + r * Math.cos(angle), c.y + r * Math.sin(angle));
  }
});

window.addEventListener('keyup', e => {
  if (e.key.toLowerCase() !== heldNoteKey) return;
  heldNoteKey = null;
  onRelease();
});

window.addEventListener('blur', () => {
  if (heldNoteKey) { heldNoteKey = null; onRelease(); }
});

// ── Initial draw ──────────────────────────────────────────────────────────────
// Defer first resize so flex layout is fully computed before we read clientWidth/Height
window.addEventListener('load', () => {
  resize();
  updateHUD();
});
// The labels are drawn on canvas, so repaint once the font has arrived.
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (W) drawWheels(); });
// Fallback in case load already fired
requestAnimationFrame(() => { if (W === 0) { resize(); updateHUD(); } });
