const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { Recorder, KEY_LABELS } = require('../gestures/recorder');

const NOW = new Date(2026, 9, 1, 14, 5, 9);
const hand = handedness => ({
  handedness,
  landmarks: Array.from({ length: 21 }, (_, i) => ({ x: 0.1 + i / 100, y: 0.2, z: 0.3 })),
});

function fakeFs() {
  const files = {};
  const dirs = [];
  return { files, dirs, mkdirSync: d => dirs.push(d), appendFileSync: (f, s) => { files[f] = (files[f] || '') + s; } };
}

test('toggling on starts a session file named after the time', () => {
  const fs = fakeFs();
  const r = new Recorder('rec', { fs, path, now: () => NOW });
  assert.equal(r.toggle(), true);
  assert.equal(r.file, path.join('rec', '20261001-140509.jsonl'));
  assert.deepEqual(fs.dirs, ['rec']);
});

test('only labelled frames are written, one JSON line per hand', () => {
  const fs = fakeFs();
  const r = new Recorder('rec', { fs, path, now: () => NOW });
  r.toggle();
  assert.equal(r.record([hand('Right')], 640, 480), 0);
  r.setLabel('fist');
  assert.equal(r.record([hand('Right'), hand('Left')], 640, 480), 2);
  const lines = fs.files[r.file].trim().split('\n').map(l => JSON.parse(l));
  assert.equal(lines.length, 2);
  assert.deepEqual(Object.keys(lines[0]).sort(),
                   ['handedness', 'height', 'label', 'landmarks', 'session', 't', 'width']);
  assert.equal(lines[0].label, 'fist');
  assert.equal(lines[0].landmarks.length, 21);
  assert.deepEqual(lines[0].landmarks[0], [0.1, 0.2, 0.3]);
  assert.equal(lines[1].handedness, 'Left');
});

test('toggling off clears the label and stops writing', () => {
  const fs = fakeFs();
  const r = new Recorder('rec', { fs, path, now: () => NOW });
  r.toggle();
  r.setLabel('palm');
  r.toggle();
  assert.equal(r.on, false);
  assert.equal(r.label, null);
  assert.equal(r.record([hand('Right')], 640, 480), 0);
});

test('number keys map to the seven poses', () => {
  assert.deepEqual(Object.values(KEY_LABELS), ['fist', 'palm', 'one', 'peace', 'three', 'four', 'other']);
});
