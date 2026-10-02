'use strict';
// Hand landmarks to the 42 numbers the classifier sees (design section 3).
// Must match ml/gesturenet/features.py exactly; test/gestures.features.test.js checks it.

const NUM_LANDMARKS = 21;
const NUM_FEATURES = 2 * NUM_LANDMARKS;
const MIN_SCALE = 1e-6;

function handFeatures(landmarks, width = 1, height = 1) {
  if (!landmarks || landmarks.length !== NUM_LANDMARKS) {
    throw new Error(`expected 21 landmarks, got ${landmarks ? landmarks.length : 0}`);
  }
  const px = new Float64Array(NUM_LANDMARKS);
  const py = new Float64Array(NUM_LANDMARKS);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = landmarks[i];
    px[i] = (Array.isArray(p) ? p[0] : p.x) * width;   // 1. pixel-proportional coordinates
    py[i] = (Array.isArray(p) ? p[1] : p.y) * height;
  }
  const wx = px[0], wy = py[0];
  for (let i = 0; i < NUM_LANDMARKS; i++) { px[i] -= wx; py[i] -= wy; }   // 2. wrist at the origin
  const vx = px[9], vy = py[9];
  const s = Math.hypot(vx, vy);
  if (!(s >= MIN_SCALE)) return null;                  // also catches NaN
  const out = new Float64Array(NUM_FEATURES);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const x = px[i], y = py[i];
    out[2 * i] = ((-vy * x + vx * y) / s) / s;          // 3. rotate upright, 4. scale
    out[2 * i + 1] = ((-vx * x - vy * y) / s) / s;
  }
  return out;
}

module.exports = { handFeatures, NUM_FEATURES, NUM_LANDMARKS };
