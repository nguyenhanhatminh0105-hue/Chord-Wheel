'use strict';
// Pure decisions for gesture control (design section 1): no DOM, no audio.

const QUALITY_BY_POSE = Object.freeze({
  one: 'maj', peace: 'min', three: 'dom7', four: 'maj7', palm: 'min7', fist: 'sus4',
});
const IDLE = Object.freeze({ playing: false, note: -1 });

// Note-wheel hand. pose: smoothed pose or null; overNote: the note slice under the palm, or -1;
// state: { playing, note }. Returns { state, actions }.
function noteHand(pose, overNote, state) {
  const release = state.playing ? { state: IDLE, actions: [{ type: 'release' }] } : { state, actions: [] };
  if (pose === 'palm') return release;
  if (pose !== 'fist') return { state, actions: [] };
  if (overNote < 0) return release;
  if (!state.playing) return { state: { playing: true, note: overNote }, actions: [{ type: 'press', note: overNote }] };
  if (overNote !== state.note) return { state: { playing: true, note: overNote }, actions: [{ type: 'slide', note: overNote }] };
  return { state, actions: [] };
}

// Quality hand: the quality a pose selects, or null for no change.
function qualityFor(pose) {
  return Object.prototype.hasOwnProperty.call(QUALITY_BY_POSE, pose) ? QUALITY_BY_POSE[pose] : null;
}

module.exports = { noteHand, qualityFor, QUALITY_BY_POSE, IDLE };
