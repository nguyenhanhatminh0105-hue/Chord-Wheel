// Chord-Wheel gestures in the browser. MediaPipe's Hand Landmarker finds the hands; the app's own
// feature transform, classifier and smoother (copied next to this file by build_space.py) name each pose.
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const LANDMARKER = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const COLORS = ['#f97316', '#a78bfa'];
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const MAX_W = 960, MAX_H = 720;

const { handFeatures } = window.chordWheel.features.exports;
const { GestureClassifier } = window.chordWheel.classifier.exports;
const { PoseSmoother } = window.chordWheel.smoother.exports;

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const results = document.getElementById('results');
const upload = document.getElementById('upload');
const uploadLabel = document.getElementById('upload-label');
const webcamBtn = document.getElementById('webcam-btn');

let landmarker = null, classifier = null, mode = 'IMAGE';
let stream = null, video = null, smoothers = {};

function status(text) {
  results.replaceChildren(Object.assign(document.createElement('p'), { id: 'status', textContent: text }));
}

function top3(probs) {
  return Array.from(probs, (p, k) => ({ label: classifier.labels[k], p }))
    .sort((a, b) => b.p - a.p).slice(0, 3);
}

// One panel per hand: the three most likely poses with their probabilities.
function showResults(probsList, smoothed = []) {
  if (probsList.length === 0) {
    status(video ? 'No hand in view.' : 'No hand found. Try a photo where one hand is clearly visible.');
    return;
  }
  results.replaceChildren(...probsList.map((probs, i) => {
    const box = Object.assign(document.createElement('div'), { className: 'hand' });
    const title = probsList.length > 1 ? `Hand ${i + 1}` : 'Hand';
    const pose = smoothed[i] ? ` · ${smoothed[i].pose}` : '';
    box.append(Object.assign(document.createElement('h2'), { textContent: title + pose, style: `color:${COLORS[i % 2]}` }));
    if (!probs) {
      box.append(Object.assign(document.createElement('p'), { textContent: 'Seen edge-on or too small to classify.' }));
      return box;
    }
    for (const { label, p } of top3(probs)) {
      const row = Object.assign(document.createElement('div'), { className: 'row' });
      const bar = Object.assign(document.createElement('div'), { className: 'bar' });
      bar.append(Object.assign(document.createElement('span'), { style: `width:${(p * 100).toFixed(1)}%;background:${COLORS[i % 2]}` }));
      row.append(Object.assign(document.createElement('span'), { textContent: label }), bar,
                 Object.assign(document.createElement('span'), { className: 'pct', textContent: `${Math.round(p * 100)}%` }));
      box.append(row);
    }
    return box;
  }));
}

// Draws the picture (mirrored for the webcam, like a mirror) with each hand's skeleton over it.
function draw(source, width, height, hands, mirror, captions = []) {
  const scale = Math.min(MAX_W / width, MAX_H / height, 1);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const X = x => (mirror ? 1 - x : x) * canvas.width;
  const Y = y => y * canvas.height;
  ctx.save();
  if (mirror) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  ctx.restore();
  const r = Math.max(2, Math.min(canvas.width, canvas.height) / 160);
  hands.forEach((lm, i) => {
    ctx.strokeStyle = ctx.fillStyle = COLORS[i % 2];
    ctx.lineWidth = r * 0.8;
    for (const [a, b] of BONES) {
      ctx.beginPath(); ctx.moveTo(X(lm[a].x), Y(lm[a].y)); ctx.lineTo(X(lm[b].x), Y(lm[b].y)); ctx.stroke();
    }
    for (const p of lm) { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), r, 0, 2 * Math.PI); ctx.fill(); }
    if (captions[i]) {
      ctx.font = `700 ${Math.round(r * 6)}px 'Space Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(captions[i], X(lm[0].x), Y(lm[0].y) + r * 9);
    }
  });
}

async function setMode(next) {
  if (mode !== next) { await landmarker.setOptions({ runningMode: next }); mode = next; }
}

const classify = (hands, width, height) => classifier.classify(hands.map(lm => handFeatures(lm, width, height)));

async function onPhoto(file) {
  await stopWebcam();
  const img = new Image();
  img.src = URL.createObjectURL(file);
  try { await img.decode(); } catch { status('That file is not an image this browser can open.'); return; }
  await setMode('IMAGE');
  const found = landmarker.detect(img);
  const probs = await classify(found.landmarks, img.naturalWidth, img.naturalHeight);
  const captions = probs.map(p => (p ? top3(p)[0].label : ''));
  draw(img, img.naturalWidth, img.naturalHeight, found.landmarks, false, captions);
  showResults(probs);
  URL.revokeObjectURL(img.src);
}

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
  } catch (err) {
    status(`The webcam could not be opened (${err.name}). Uploading a photo still works.`);
    return;
  }
  video = Object.assign(document.createElement('video'), { srcObject: stream, muted: true, playsInline: true });
  await video.play();
  await setMode('VIDEO');
  smoothers = {};
  webcamBtn.classList.add('active');
  webcamBtn.textContent = 'Stop the webcam';
  requestAnimationFrame(frame);
}

async function stopWebcam() {
  if (!stream) return;
  stream.getTracks().forEach(t => t.stop());
  stream = null; video = null; smoothers = {};
  webcamBtn.classList.remove('active');
  webcamBtn.textContent = 'Use the webcam';
  status('Ready: upload a photo of a hand, or use the webcam.');
}

// One webcam frame: find hands, classify, smooth per hand (keyed by MediaPipe's handedness, as the
// app keys its two roles), draw. The next frame is requested only after this one is done.
async function frame() {
  if (!video) return;
  if (video.readyState >= 2) {
    const found = landmarker.detectForVideo(video, performance.now());
    const keys = [];
    for (const [i, h] of found.handedness.entries()) {
      const k = h[0].categoryName;
      keys.push(keys.includes(k) ? `${k}${i}` : k);   // two hands with the same label keep separate smoothers
    }
    for (const k of Object.keys(smoothers)) if (!keys.includes(k)) delete smoothers[k];
    const probs = await classify(found.landmarks, video.videoWidth, video.videoHeight);
    if (!video) return;
    const smoothed = probs.map((p, i) => {
      const s = smoothers[keys[i]] || (smoothers[keys[i]] = new PoseSmoother(classifier.labels));
      s.push(p);
      return s.pose ? { pose: s.pose, confidence: s.confidence } : null;
    });
    draw(video, video.videoWidth, video.videoHeight, found.landmarks, true,
         smoothed.map(s => (s ? `${s.pose} ${Math.round(s.confidence * 100)}%` : '')));
    showResults(probs, smoothed);
  }
  requestAnimationFrame(frame);
}

async function init() {
  ort.env.wasm.numThreads = 1;   // threads need cross-origin isolation, which a Space page does not have
  const labels = await (await fetch('labels.json')).json();
  const session = await ort.InferenceSession.create('gestures.onnx', { executionProviders: ['wasm'] });
  classifier = new GestureClassifier(session, labels, ort);
  const vision = await FilesetResolver.forVisionTasks(WASM);
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: LANDMARKER, delegate: 'CPU' }, runningMode: 'IMAGE', numHands: 2,
  });
  upload.disabled = false;
  uploadLabel.removeAttribute('aria-disabled');
  webcamBtn.disabled = false;
  upload.addEventListener('change', () => { if (upload.files[0]) onPhoto(upload.files[0]).finally(() => { upload.value = ''; }); });
  webcamBtn.addEventListener('click', () => (stream ? stopWebcam() : startWebcam()));
  status('Ready: upload a photo of a hand, or use the webcam.');
  document.body.dataset.ready = 'true';
}

init().catch(err => status(`Loading failed: ${err.message}. Reloading the page usually helps.`));
