// ── renderer.js ──────────────────────────────────────────────────────────────
'use strict';

// Close button fix for frameless window
const { ipcRenderer } = require('electron');
document.getElementById('close-btn').addEventListener('click', () => {
  window.close();
});

// ── Constants ────────────────────────────────────────────────────────────────
const NOTES_SHARP  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const NOTES_FLAT   = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'];
const NOTES_SOLFEGE= ['Do','Di','Re','Ri','Mi','Fa','Fi','Sol','Si','La','Li','Ti'];

const QUALITIES = ['maj','min','dim','aug','maj7','min7','dom7','sus2','sus4','m7b5'];

// Semitone intervals for each quality (relative to root)
const QUALITY_INTERVALS = {
  maj:  [0, 4, 7],
  min:  [0, 3, 7],
  dim:  [0, 3, 6],
  aug:  [0, 4, 8],
  maj7: [0, 4, 7, 11],
  min7: [0, 3, 7, 10],
  dom7: [0, 4, 7, 10],
  sus2: [0, 2, 7],
  sus4: [0, 5, 7],
  m7b5: [0, 3, 6, 10],
};

const SCALES = {
  major:         [0,2,4,5,7,9,11],
  minor:         [0,2,3,5,7,8,10],
  pentatonic:    [0,2,4,7,9],
  blues:         [0,3,5,6,7,10],
  dorian:        [0,2,3,5,7,9,10],
  phrygian:      [0,1,3,5,7,8,10],
  lydian:        [0,2,4,6,7,9,11],
  mixolydian:    [0,2,4,5,7,9,10],
  harmonic_minor:[0,2,3,5,7,8,11],
};

// Base MIDI note numbers for C4 = 60
function midiToFreq(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }
const C4 = 60;

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
  const intervals = QUALITY_INTERVALS[QUALITIES[qualIdx]];
  const rootMidi = C4 + noteIdx;
  const bassMidi = rootMidi - 12;

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
  for (const interval of intervals) {
    const osc = audioCtx.createOscillator();
    osc.type = waveType;
    osc.frequency.value = midiToFreq(rootMidi + interval);
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
  [wheelCanvas, overlayCanvas, rippleCanvas].forEach(c => {
    c.width = W; c.height = H;
  });
  drawWheels();
}

window.addEventListener('resize', resize);

// ── Geometry helpers ─────────────────────────────────────────────────────────
function noteWheelCenter()  { return { x: W * 0.22, y: H * 0.70 }; }
function noteWheelRadius()  { return Math.min(W * 0.17, H * 0.26); }
function qualWheelCenter()  { return { x: W * 0.78, y: H * 0.70 }; }
function qualWheelRadius()  { return Math.min(W * 0.12, H * 0.18); }
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
  const intervals = SCALES[currentScale];
  return intervals.map(i => (scaleRoot + i) % 12);
}

// ── Drawing ───────────────────────────────────────────────────────────────────
function getNoteName(idx) {
  if (labelMode === 'flat') return NOTES_FLAT[idx];
  if (labelMode === 'solfege') return NOTES_SOLFEGE[idx];
  return NOTES_SHARP[idx];
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

function drawText(ctx, text, cx, cy, r, angle, color, size) {
  ctx.save();
  ctx.translate(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
  ctx.rotate(angle + Math.PI / 2);
  ctx.fillStyle = color;
  ctx.font = `bold ${size}px 'Space Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function drawWheels() {
  wCtx.clearRect(0, 0, W, H);
  wCtx.globalAlpha = 0.55;

  const nc = noteWheelCenter();
  const nr = noteWheelRadius();
  const qc = qualWheelCenter();
  const qr = qualWheelRadius();
  const scaleNotes = getScaleNotes();

  // ── Note wheel ──
  const sa12 = segmentAngle(12);
  for (let i = 0; i < 12; i++) {
    const start = -Math.PI / 2 + i * sa12 - sa12 / 2;
    const end   = start + sa12;
    const mid   = (start + end) / 2;
    const active = i === selectedNote && isPlaying;
    const inScale = scaleNotes.includes(i);
    const dimmed = snapEnabled && !inScale;

    let fill = active
      ? '#f97316'
      : dimmed ? '#141420' : (inScale ? '#1e2a3a' : '#171727');
    let stroke = active ? '#fbbf24' : dimmed ? '#222' : '#334';
    const isBlack = [1,3,6,8,10].includes(i);
    if (!active && !dimmed) fill = isBlack ? '#12121e' : '#1a1a2e';

    drawSegment(wCtx, nc.x, nc.y, nr * INNER_RATIO + 4, nr, start, end, fill, stroke, active);

    const textR = nr * 0.72;
    const textColor = active ? '#fff' : dimmed ? '#333' : (inScale ? '#cbd5e1' : '#555');
    const fontSize = nr * 0.1;
    drawText(wCtx, getNoteName(i), nc.x, nc.y, textR, mid, textColor, Math.max(10, fontSize));
  }

  // Inner OFF button
  wCtx.save();
  const offActive = selectedNote === -1 && !isPlaying;
  wCtx.beginPath();
  wCtx.arc(nc.x, nc.y, nr * INNER_RATIO, 0, Math.PI * 2);
  wCtx.fillStyle = offActive ? '#1e1e2e' : '#0d0d0f';
  wCtx.fill();
  wCtx.strokeStyle = '#333';
  wCtx.lineWidth = 1.5;
  wCtx.stroke();
  wCtx.fillStyle = '#444';
  wCtx.font = `bold ${nr * 0.08}px 'Space Mono', monospace`;
  wCtx.textAlign = 'center';
  wCtx.textBaseline = 'middle';
  wCtx.fillText('OFF', nc.x, nc.y);
  wCtx.restore();

  // Wheel label
  wCtx.save();
  wCtx.fillStyle = '#333';
  wCtx.font = `10px 'Space Mono', monospace`;
  wCtx.textAlign = 'center';
  wCtx.fillText('NOTE', nc.x, nc.y + nr + 16);
  wCtx.restore();

  // ── Quality wheel ──
  const saQ = segmentAngle(QUALITIES.length);
  for (let i = 0; i < QUALITIES.length; i++) {
    const start = -Math.PI / 2 + i * saQ - saQ / 2;
    const end   = start + saQ;
    const mid   = (start + end) / 2;
    const active = i === selectedQuality && isPlaying;

    const fill = active ? '#a78bfa' : '#13132a';
    const stroke = active ? '#c4b5fd' : '#2a2a4a';
    drawSegment(wCtx, qc.x, qc.y, qr * INNER_RATIO + 3, qr, start, end, fill, stroke, active);

    const textR = qr * 0.72;
    const textColor = active ? '#fff' : '#6666aa';
    const fontSize = Math.max(9, qr * 0.12);
    drawText(wCtx, QUALITIES[i], qc.x, qc.y, textR, mid, textColor, fontSize);
  }

  // Inner quality circle
  wCtx.save();
  wCtx.beginPath();
  wCtx.arc(qc.x, qc.y, qr * INNER_RATIO, 0, Math.PI * 2);
  wCtx.fillStyle = '#0d0d0f';
  wCtx.fill();
  wCtx.strokeStyle = '#2a2a4a';
  wCtx.lineWidth = 1.5;
  wCtx.stroke();
  wCtx.restore();

  // Wheel label
  wCtx.save();
  wCtx.fillStyle = '#333';
  wCtx.font = `10px 'Space Mono', monospace`;
  wCtx.textAlign = 'center';
  wCtx.fillText('QUALITY', qc.x, qc.y + qr + 16);
  wCtx.restore();
  wCtx.globalAlpha = 1;
}

// ── HUD update ────────────────────────────────────────────────────────────────
function updateHUD() {
  const chordNameEl   = document.getElementById('chord-name');
  const qualLabelEl   = document.getElementById('quality-label');
  const notePillsEl   = document.getElementById('note-pills');

  if (!isPlaying || selectedNote < 0) {
    chordNameEl.textContent = '—';
    qualLabelEl.textContent = '';
    notePillsEl.innerHTML = '';
    return;
  }

  const root = getNoteName(selectedNote);
  const qual = QUALITIES[selectedQuality];
  chordNameEl.textContent = `${root} ${qual}`;
  qualLabelEl.textContent = qualFullName(qual);

  const intervals = QUALITY_INTERVALS[qual];
  const noteIndices = intervals.map(i => (selectedNote + i) % 12);
  notePillsEl.innerHTML = noteIndices.map(ni => {
    const name = getNoteName(ni);
    return `<span class="note-pill active">${name}</span>`;
  }).join('');
}

function qualFullName(q) {
  const map = {
    maj:'Major', min:'Minor', dim:'Diminished', aug:'Augmented',
    maj7:'Major 7th', min7:'Minor 7th', dom7:'Dominant 7th',
    sus2:'Suspended 2nd', sus4:'Suspended 4th', m7b5:'Half-Diminished',
  };
  return map[q] || q;
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
document.getElementById('scale-select').addEventListener('change', e => {
  currentScale = e.target.value;
  drawWheels();
});

document.getElementById('label-select').addEventListener('change', e => {
  labelMode = e.target.value;
  drawWheels();
  updateHUD();
});

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
  document.getElementById('snap-btn').classList.toggle('active', snapEnabled);
  drawWheels();
});

document.getElementById('reverb-btn').addEventListener('click', () => {
  reverbEnabled = !reverbEnabled;
  document.getElementById('reverb-btn').classList.toggle('active-purple', reverbEnabled);
});

// ── Webcam ────────────────────────────────────────────────────────────────────
const videoEl = document.getElementById('video-bg');
let videoStream = null;

document.getElementById('cam-btn').addEventListener('click', async () => {
  camEnabled = !camEnabled;
  document.getElementById('cam-btn').classList.toggle('active', camEnabled);

  if (camEnabled) {
    try {
      videoStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      videoEl.srcObject = videoStream;
      videoEl.style.display = 'block';
      initHandTracking();
    } catch (err) {
      console.error('Camera error:', err);
      camEnabled = false;
      document.getElementById('cam-btn').classList.remove('active');
    }
  } else {
    videoEl.style.display = 'none';
    if (videoStream) {
      videoStream.getTracks().forEach(t => t.stop());
      videoStream = null;
    }
    stopHandTracking();
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

async function buildHandsInstance() {
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
  h.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,       // full model — more robust to hand shape changes
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.3,
  });
  h.onResults(onHandResults);
  await h.initialize();
  process.versions.node = savedNodeVer;
  return h;
}

async function initHandTracking() {
  try {
    handsInstance = await buildHandsInstance();

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

// ── Initial draw ──────────────────────────────────────────────────────────────
// Defer first resize so flex layout is fully computed before we read clientWidth/Height
window.addEventListener('load', () => {
  resize();
  updateHUD();
  document.getElementById('reverb-btn').classList.toggle('active-purple', reverbEnabled);
});
// Fallback in case load already fired
requestAnimationFrame(() => { if (W === 0) { resize(); updateHUD(); } });
