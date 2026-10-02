'use strict';
// Runs models/gestures.onnx with ONNX Runtime (design section 5). One inference at a time:
// frames that arrive while one is running are skipped, not queued.

const NUM_FEATURES = 42;

class GestureClassifier {
  constructor(session, labels, ort) {
    this.session = session;
    this.labels = labels;
    this.ort = ort;
    this.busy = false;
  }

  static async load(modelPath, labelsPath, ort = require('onnxruntime-node'), fs = require('fs')) {
    const labels = JSON.parse(fs.readFileSync(labelsPath, 'utf8'));
    const session = await ort.InferenceSession.create(modelPath);
    return new GestureClassifier(session, labels, ort);
  }

  // rows: Float64Array(42) or null per hand. Resolves to Float32Array(probs) or null per hand.
  async classify(rows) {
    const out = rows.map(() => null);
    const live = [];
    rows.forEach((r, i) => { if (r) live.push(i); });
    if (live.length === 0) return out;
    const flat = new Float32Array(live.length * NUM_FEATURES);
    live.forEach((row, j) => flat.set(rows[row], j * NUM_FEATURES));
    const input = new this.ort.Tensor('float32', flat, [live.length, NUM_FEATURES]);
    const result = await this.session.run({ landmarks: input });
    const probs = result.probs.data;
    const k = this.labels.length;
    live.forEach((row, j) => { out[row] = probs.slice(j * k, (j + 1) * k); });
    return out;
  }

  // For the frame loop: start an inference unless one is running. Returns false when skipped.
  tryClassify(rows, onResult, onError = () => {}) {
    if (this.busy) return false;
    this.busy = true;
    this.classify(rows).then(onResult, onError).finally(() => { this.busy = false; });
    return true;
  }
}

module.exports = { GestureClassifier, NUM_FEATURES };
