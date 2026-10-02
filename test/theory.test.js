const test = require('node:test');
const assert = require('node:assert/strict');
const theory = require('../theory');

test('A4 is 440 Hz and C4 is middle C', () => {
  assert.equal(theory.midiToFreq(69), 440);
  assert.ok(Math.abs(theory.midiToFreq(theory.C4) - 261.6256) < 1e-3);
});

test('an octave doubles the frequency', () => {
  assert.ok(Math.abs(theory.midiToFreq(81) / theory.midiToFreq(69) - 2) < 1e-12);
});

test('offers the 9 scales and 10 chord qualities the README promises', () => {
  assert.equal(Object.keys(theory.SCALES).length, 9);
  assert.equal(theory.QUALITIES.length, 10);
});

test('every chord quality starts on the root and stays within one octave', () => {
  for (const q of theory.QUALITIES) {
    const iv = theory.QUALITY_INTERVALS[q];
    assert.equal(iv[0], 0, q);
    for (let i = 1; i < iv.length; i++) assert.ok(iv[i] > iv[i - 1] && iv[i] < 12, q);
  }
});

test('scales transpose to any root', () => {
  assert.deepEqual(theory.scaleNotes('major', 0), [0, 2, 4, 5, 7, 9, 11]);
  // G major has one sharp: F# (pitch class 6).
  assert.deepEqual(theory.scaleNotes('major', 7), [7, 9, 11, 0, 2, 4, 6]);
});

test('note names follow the label mode', () => {
  assert.equal(theory.noteName(1, 'sharp'), 'C#');
  assert.equal(theory.noteName(1, 'flat'), 'Db');
  assert.equal(theory.noteName(7, 'solfege'), 'Sol');
});

test('chords are spelled from the root', () => {
  assert.deepEqual(theory.chordPitchClasses(9, 'min'), [9, 0, 4]);      // A C E
  assert.deepEqual(theory.chordPitchClasses(7, 'dom7'), [7, 11, 2, 5]); // G B D F
});

test('a chord plays its tones from C4 up plus a bass note an octave below the root', () => {
  assert.deepEqual(theory.chordMidiNotes(0, 'maj'), { bass: 48, tones: [60, 64, 67] });
  assert.deepEqual(theory.chordMidiNotes(2, 'sus4'), { bass: 50, tones: [62, 67, 69] });
});

test('chord qualities have readable names', () => {
  assert.equal(theory.qualityFullName('m7b5'), 'Half-Diminished');
  assert.equal(theory.qualityFullName('unknown'), 'unknown');
});
