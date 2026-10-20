'use strict';
// Per-hand smoothing of classifier output (design section 1). A pose takes effect when it is the top
// class in a majority of the last `window` frames with mean probability >= threshold. A pose already in
// effect gives way only to one that holds a frame more than a majority, so a hand wavering between two
// poses does not flip back and forth. 'other' never takes effect.

class PoseSmoother {
  constructor(classes, { window = 5, threshold = 0.7, otherLabel = 'other' } = {}) {
    this.classes = classes;
    this.window = window;
    this.threshold = threshold;
    this.otherLabel = otherLabel;
    this.need = Math.floor(window / 2) + 1;               // to take effect: 3 of 5
    this.needToSwitch = Math.min(window, this.need + 1);  // to replace the pose in effect: 4 of 5
    this.reset();
  }

  reset() {
    this.history = [];
    this.pose = null;
    this.confidence = 0;
    this.holding = false;   // whether the latest frames show a recognised pose
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
    let leader = null;   // the pose with a confident majority, if any
    for (const [label2, t] of tally) {
      if (label2 !== this.otherLabel && t.n >= this.need && t.sum / t.n >= this.threshold) {
        leader = { label: label2, n: t.n, mean: t.sum / t.n };
      }
    }
    this.holding = leader !== null;
    if (leader && (this.pose === null || leader.label === this.pose || leader.n >= this.needToSwitch)) {
      this.pose = leader.label;
      this.confidence = leader.mean;
    }
    return this.pose;
  }
}

module.exports = { PoseSmoother };
