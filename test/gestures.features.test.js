const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { handFeatures, NUM_FEATURES } = require('../gestures/features');

const fixture = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'ml', 'tests', 'fixtures', 'features.json'), 'utf8'));

test('matches the Python feature transform on every fixture case', () => {
  for (const [i, c] of fixture.cases.entries()) {
    const got = handFeatures(c.landmarks, c.width, c.height);
    if (c.expected === null) { assert.equal(got, null, `case ${i}`); continue; }
    assert.equal(got.length, NUM_FEATURES);
    for (let k = 0; k < NUM_FEATURES; k++) {
      assert.ok(Math.abs(got[k] - c.expected[k]) <= 1e-9,
        `case ${i} feature ${k}: ${got[k]} vs ${c.expected[k]}`);
    }
  }
});

test('accepts MediaPipe {x, y, z} landmark objects', () => {
  const c = fixture.cases.find(k => k.expected !== null);
  const objects = c.landmarks.map(([x, y, z]) => ({ x, y, z }));
  assert.deepEqual(Array.from(handFeatures(objects, c.width, c.height)),
                   Array.from(handFeatures(c.landmarks, c.width, c.height)));
});

test('a degenerate hand or a zero-sized image gives null instead of throwing', () => {
  assert.equal(handFeatures(fixture.cases[0].landmarks, 0, 0), null);
  assert.equal(handFeatures(Array(21).fill([0.5, 0.5, 0]), 1, 1), null);
});

test('rejects a landmark list of the wrong length', () => {
  assert.throws(() => handFeatures([[0, 0, 0]], 1, 1), /21 landmarks/);
});
