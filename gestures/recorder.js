'use strict';
// Recording mode for the webcam test set (design section 1): R toggles it, holding a number key
// labels frames, and each labelled hand is appended as one JSON line.

const KEY_LABELS = Object.freeze({
  1: 'fist', 2: 'palm', 3: 'one', 4: 'peace', 5: 'three', 6: 'four', 7: 'other',
});

function sessionName(date) {
  const pad = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-` +
         `${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

class Recorder {
  constructor(dir, { fs = require('fs'), path = require('path'), now = () => new Date() } = {}) {
    this.dir = dir;
    this.fs = fs;
    this.path = path;
    this.now = now;
    this.on = false;
    this.label = null;
    this.session = null;
    this.file = null;
  }

  toggle() {
    this.on = !this.on;
    if (this.on) {
      this.session = sessionName(this.now());
      this.fs.mkdirSync(this.dir, { recursive: true });
      this.file = this.path.join(this.dir, `${this.session}.jsonl`);
    } else {
      this.label = null;
    }
    return this.on;
  }

  setLabel(label) {
    this.label = label;
  }

  // hands: [{ landmarks: [{x, y, z}], handedness: 'Left' | 'Right' }]. Returns lines written.
  record(hands, width, height) {
    if (!this.on || !this.label || hands.length === 0) return 0;
    const t = this.now().getTime();
    const lines = hands.map(h => JSON.stringify({
      session: this.session, t, label: this.label, handedness: h.handedness, width, height,
      landmarks: h.landmarks.map(p => [p.x, p.y, p.z]),
    }));
    this.fs.appendFileSync(this.file, lines.join('\n') + '\n');
    return hands.length;
  }
}

module.exports = { Recorder, KEY_LABELS, sessionName };
