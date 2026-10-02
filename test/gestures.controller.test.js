const test = require('node:test');
const assert = require('node:assert/strict');
const { GestureController } = require('../gestures/controller');

const CLASSES = ['fist', 'palm', 'one', 'peace', 'three', 'four', 'other'];
const onehot = (name, p = 0.95) => CLASSES.map(c => (c === name ? p : (1 - p) / 6));
const HAND = Array.from({ length: 21 }, (_, i) => ({ x: 0.5 + 0.01 * i, y: 0.5 - 0.02 * i, z: 0 }));
const note = [{ role: 'note', handedness: 'Right', landmarks: HAND }];

function syncClassifier() {   // answers immediately with whatever `next` holds
  const c = { next: [], rows: [], tryClassify(rows, onResult) { c.rows.push(rows); onResult(c.next.slice(0, rows.length)); return true; } };
  return c;
}
function deferredClassifier() {  // keeps the callback so a test can answer late
  const c = { pending: null, tryClassify(rows, onResult) { c.pending = onResult; return true; } };
  return c;
}
function fakeUi() {
  const log = [];
  return {
    log,
    noteAtPoint: x => (x < 100 ? 9 : -1),
    press: n => log.push(['press', n]),
    slide: n => log.push(['slide', n]),
    release: () => log.push(['release']),
    setQuality: q => log.push(['quality', q]),
  };
}
const frames = (c, detected, notePos, qualityPos, n = 3) => {
  for (let i = 0; i < n; i++) { c.observe(detected, 640, 480); c.control(notePos, qualityPos); }
};

test('a fist held over a note presses it once, then a palm releases it', () => {
  const clf = syncClassifier(), ui = fakeUi();
  const c = new GestureController({ classifier: clf, classes: CLASSES, ui });
  clf.next = [onehot('fist')];
  frames(c, note, { x: 50, y: 50 }, null, 4);
  assert.deepEqual(ui.log, [['press', 9]]);
  clf.next = [onehot('palm')];
  frames(c, note, { x: 50, y: 50 }, null, 3);
  assert.deepEqual(ui.log, [['press', 9], ['release']]);
});

test('the right hand shape sets the quality; other leaves position control in charge', () => {
  const clf = syncClassifier(), ui = fakeUi();
  const c = new GestureController({ classifier: clf, classes: CLASSES, ui });
  const quality = [{ role: 'quality', handedness: 'Left', landmarks: HAND }];
  clf.next = [onehot('peace')];
  frames(c, quality, null, { x: 300, y: 50 });
  assert.deepEqual(ui.log.at(-1), ['quality', 'min']);
  assert.equal(c.control(null, { x: 300, y: 50 }).qualityHandled, true);
  const c2 = new GestureController({ classifier: syncClassifier(), classes: CLASSES, ui: fakeUi() });
  assert.equal(c2.control(null, { x: 300, y: 50 }).qualityHandled, false);
});

test('losing a hand forgets it: release, then a returning fist needs fresh frames', () => {
  const clf = syncClassifier(), ui = fakeUi();
  const c = new GestureController({ classifier: clf, classes: CLASSES, ui });
  clf.next = [onehot('fist')];
  frames(c, note, { x: 50, y: 50 }, null);
  c.observe([], 640, 480);
  c.control(null, null);
  assert.deepEqual(ui.log, [['press', 9], ['release']]);
  frames(c, note, { x: 50, y: 50 }, null, 2);
  assert.deepEqual(ui.log, [['press', 9], ['release']]);
  frames(c, note, { x: 50, y: 50 }, null, 1);
  assert.deepEqual(ui.log.at(-1), ['press', 9]);
});

test('late results for a lost hand are ignored', () => {
  const clf = deferredClassifier(), ui = fakeUi();
  const c = new GestureController({ classifier: clf, classes: CLASSES, ui });
  c.observe(note, 640, 480);
  const late = clf.pending;
  c.observe([], 640, 480);
  for (let i = 0; i < 3; i++) late([onehot('fist')]);
  assert.equal(c.labels().note, null);
});

test('a zero-sized video falls back to normalised coordinates', () => {
  const clf = syncClassifier();
  const c = new GestureController({ classifier: clf, classes: CLASSES, ui: fakeUi() });
  clf.next = [onehot('fist')];
  c.observe(note, 0, 0);
  assert.ok(clf.rows[0][0] instanceof Float64Array);
});

test('reset releases a held chord and clears both hands', () => {
  const clf = syncClassifier(), ui = fakeUi();
  const c = new GestureController({ classifier: clf, classes: CLASSES, ui });
  clf.next = [onehot('fist')];
  frames(c, note, { x: 50, y: 50 }, null);
  c.reset();
  assert.deepEqual(ui.log, [['press', 9], ['release']]);
  assert.deepEqual(c.labels(), { note: null, quality: null });
});
