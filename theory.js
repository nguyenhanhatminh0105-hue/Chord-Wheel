// ── theory.js ────────────────────────────────────────
// Music theory behind the wheel: note names, scales, chord spelling and pitch.
// Pure functions with no DOM or audio, so they run under Node for the tests.
'use strict';

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

const QUALITY_NAMES = {
  maj:'Major', min:'Minor', dim:'Diminished', aug:'Augmented',
  maj7:'Major 7th', min7:'Minor 7th', dom7:'Dominant 7th',
  sus2:'Suspended 2nd', sus4:'Suspended 4th', m7b5:'Half-Diminished',
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
const C4 = 60;
function midiToFreq(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }

// Pitch classes (0-11) of a scale built on a root.
function scaleNotes(scale, root) {
  return SCALES[scale].map(i => (root + i) % 12);
}

function noteName(idx, labelMode) {
  if (labelMode === 'flat') return NOTES_FLAT[idx];
  if (labelMode === 'solfege') return NOTES_SOLFEGE[idx];
  return NOTES_SHARP[idx];
}

function qualityFullName(q) {
  return QUALITY_NAMES[q] || q;
}

// Pitch classes (0-11) of a chord, root first.
function chordPitchClasses(rootIdx, quality) {
  return QUALITY_INTERVALS[quality].map(i => (rootIdx + i) % 12);
}

// MIDI notes the synth plays: chord tones from C4 up, plus a bass an octave below the root.
function chordMidiNotes(rootIdx, quality) {
  const root = C4 + rootIdx;
  return { bass: root - 12, tones: QUALITY_INTERVALS[quality].map(i => root + i) };
}

module.exports = {
  NOTES_SHARP, NOTES_FLAT, NOTES_SOLFEGE, QUALITIES, QUALITY_INTERVALS, SCALES, C4,
  midiToFreq, scaleNotes, noteName, qualityFullName, chordPitchClasses, chordMidiNotes,
};
