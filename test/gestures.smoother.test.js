const test = require('node:test');
const assert = require('node:assert/strict');
const { PoseSmoother } = require('../gestures/smoother');

const CLASSES = ['fist', 'palm', 'one', 'peace', 'three', 'four', 'other'];
const p = (name, conf = 0.9) => CLASSES.map(c => (c === name ? conf : (1 - conf) / 6));
const feed = (s, seq) => seq.forEach(x => s.push(x === null ? null : p(x)));

test('three confident frames of a pose make it the current pose', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'fist']);
  assert.equal(s.pose, null);
  s.push(p('fist'));
  assert.equal(s.pose, 'fist');
  assert.ok(Math.abs(s.confidence - 0.9) < 1e-12);
});

test('a pose below the confidence threshold does not take effect', () => {
  const s = new PoseSmoother(CLASSES);
  for (let i = 0; i < 5; i++) s.push(p('fist', 0.6));
  assert.equal(s.pose, null);
});

test('a single odd frame does not flip the pose', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'fist', 'fist', 'palm', 'fist']);
  assert.equal(s.pose, 'fist');
});

test('a new pose takes over once it holds the majority', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'fist', 'fist', 'palm', 'palm']);
  assert.equal(s.pose, 'fist');
  s.push(p('palm'));
  assert.equal(s.pose, 'palm');
});

test('other never changes the pose', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'fist', 'fist', 'other', 'other', 'other', 'other', 'other']);
  assert.equal(s.pose, 'fist');
});

test('tied or alternating frames change nothing', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'palm', 'fist', 'palm', 'other']);
  assert.equal(s.pose, null);
});

test('no prediction counts as other', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'fist', 'fist', null, null, null, null, null]);
  assert.equal(s.pose, 'fist');
});

test('reset forgets the history and the pose', () => {
  const s = new PoseSmoother(CLASSES);
  feed(s, ['fist', 'fist', 'fist']);
  s.reset();
  assert.equal(s.pose, null);
  s.push(p('fist'));
  assert.equal(s.pose, null);
});
