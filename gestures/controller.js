'use strict';
// Glue for the frame loop (design section 7): features -> classifier -> per-hand smoother ->
// actions. The app's press/slide/release/setQuality functions are passed in as `ui`.

const { handFeatures } = require('./features');
const { PoseSmoother } = require('./smoother');
const { noteHand, qualityFor, IDLE } = require('./actions');

const ROLES = ['note', 'quality'];

class GestureController {
  constructor({ classifier, classes, ui, smootherOptions = {} }) {
    this.classifier = classifier;
    this.ui = ui;
    this.smoothers = {};
    this.gen = {};
    for (const role of ROLES) {
      this.smoothers[role] = new PoseSmoother(classes, smootherOptions);
      this.gen[role] = 0;
    }
    this.noteState = IDLE;
  }

  // detected: hands seen this frame, [{ role, landmarks, handedness }]. Starts an inference.
  observe(detected, width, height) {
    const byRole = {};
    for (const h of detected) byRole[h.role] = h;   // last hand of a role wins, as palm tracking does
    for (const role of ROLES) if (!byRole[role]) this._forget(role);
    const present = ROLES.filter(r => byRole[r]);
    if (present.length === 0) return;
    const gens = present.map(r => this.gen[r]);
    const rows = present.map(r => handFeatures(byRole[r].landmarks, width || 1, height || 1));
    const deliver = probsFor => present.forEach((r, i) => {
      if (this.gen[r] === gens[i]) this.smoothers[r].push(probsFor(i));   // ignore results for a lost hand
    });
    this.classifier.tryClassify(rows, probs => deliver(i => probs[i]), () => deliver(() => null));
  }

  // notePos / qualityPos: smoothed palm positions in canvas pixels, or null.
  control(notePos, qualityPos) {
    if (!notePos) {
      if (this.noteState.playing) {
        this.noteState = IDLE;
        this.ui.release();
      }
    } else {
      const over = this.ui.noteAtPoint(notePos.x, notePos.y);
      const { state, actions } = noteHand(this.smoothers.note.pose, over >= 0 ? over : -1, this.noteState);
      this.noteState = state;
      for (const a of actions) {
        if (a.type === 'press') this.ui.press(a.note, notePos.x, notePos.y);
        else if (a.type === 'slide') this.ui.slide(a.note, notePos.x, notePos.y);
        else this.ui.release();
      }
    }
    const quality = qualityPos ? qualityFor(this.smoothers.quality.pose) : null;
    if (quality) this.ui.setQuality(quality, qualityPos.x, qualityPos.y);
    return { qualityHandled: quality !== null };
  }

  labels() {
    const out = {};
    for (const role of ROLES) {
      const s = this.smoothers[role];
      out[role] = s.pose ? { pose: s.pose, confidence: s.confidence } : null;
    }
    return out;
  }

  reset() {
    for (const role of ROLES) this._forget(role);
    if (this.noteState.playing) this.ui.release();
    this.noteState = IDLE;
  }

  _forget(role) {
    this.smoothers[role].reset();
    this.gen[role] += 1;
  }
}

module.exports = { GestureController, ROLES };
