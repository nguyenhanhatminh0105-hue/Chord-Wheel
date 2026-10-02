'use strict';
// Per-hand smoothing of classifier output (design section 1): a pose takes effect when it wins
// a majority of the last `window` frames with mean probability >= threshold; 'other' never does.

class PoseSmoother {
  constructor(classes, { window = 5, threshold = 0.7, otherLabel = 'other' } = {}) {
    this.classes = classes;
    this.window = window;
    this.threshold = threshold;
    this.otherLabel = otherLabel;
    this.need = Math.floor(window / 2) + 1;
    this.reset();
  }

  reset() {
    this.history = [];
    this.pose = null;
    this.confidence = 0;
  }

  // probs: class probabilities, or null when this frame had no usable prediction.
  push(probs) {
    let label = this.otherLabel;
    let prob = 1;
    if (probs) {
      let best = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
      label = this.classes[best];
      prob = probs[best];
    }
    this.history.push({ label, prob });
    if (this.history.length > this.window) this.history.shift();

    const tally = new Map();
    for (const h of this.history) {
      const t = tally.get(h.label) || { n: 0, sum: 0 };
      t.n += 1;
      t.sum += h.prob;
      tally.set(h.label, t);
    }
    for (const [label2, t] of tally) {
      if (label2 !== this.otherLabel && t.n >= this.need && t.sum / t.n >= this.threshold) {
        this.pose = label2;
        this.confidence = t.sum / t.n;
      }
    }
    return this.pose;
  }
}

module.exports = { PoseSmoother };
