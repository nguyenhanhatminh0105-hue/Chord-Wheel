const test = require('node:test');
const assert = require('node:assert/strict');
const { noteHand, qualityFor, IDLE } = require('../gestures/actions');

const playing = note => ({ playing: true, note });

test('a fist over a note presses it', () => {
  assert.deepEqual(noteHand('fist', 9, IDLE), { state: playing(9), actions: [{ type: 'press', note: 9 }] });
});

test('sliding a fist to another note slides', () => {
  assert.deepEqual(noteHand('fist', 4, playing(9)), { state: playing(4), actions: [{ type: 'slide', note: 4 }] });
});

test('a fist staying on the same note does nothing more', () => {
  assert.deepEqual(noteHand('fist', 9, playing(9)), { state: playing(9), actions: [] });
});

test('a fist away from the notes releases', () => {
  assert.deepEqual(noteHand('fist', -1, playing(9)), { state: IDLE, actions: [{ type: 'release' }] });
  assert.deepEqual(noteHand('fist', -1, IDLE), { state: IDLE, actions: [] });
});

test('a palm releases a held chord and is silent otherwise', () => {
  assert.deepEqual(noteHand('palm', 9, playing(9)), { state: IDLE, actions: [{ type: 'release' }] });
  assert.deepEqual(noteHand('palm', 9, IDLE), { state: IDLE, actions: [] });
});

test('other, no pose, or a right-hand shape keeps the state', () => {
  for (const pose of ['other', null, 'peace']) {
    assert.deepEqual(noteHand(pose, 9, playing(9)), { state: playing(9), actions: [] });
  }
});

test('right-hand shapes map to qualities', () => {
  assert.deepEqual(['one', 'peace', 'three', 'four', 'palm', 'fist', 'other', null].map(qualityFor),
                   ['maj', 'min', 'dom7', 'maj7', 'min7', 'sus4', null, null]);
});

test('qualityFor ignores inherited object keys', () => {
  assert.equal(qualityFor('toString'), null);
});
