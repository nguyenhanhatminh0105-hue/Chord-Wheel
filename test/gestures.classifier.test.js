const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { GestureClassifier } = require('../gestures/classifier');

const ROOT = path.join(__dirname, '..');
const CLASSES = ['fist', 'palm', 'one', 'peace', 'three', 'four', 'other'];

function fakeOrt(run) {
  class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
  return { Tensor, session: { run } };
}

test('classify batches the usable rows and keeps nulls in place', async () => {
  let seen;
  const ort = fakeOrt(async feeds => {
    seen = feeds.landmarks;
    const n = seen.dims[0];
    const data = new Float32Array(n * 7);
    for (let i = 0; i < n; i++) data[i * 7 + i] = 1;
    return { probs: { data } };
  });
  const c = new GestureClassifier(ort.session, CLASSES, ort);
  const row = new Float64Array(42).fill(0.5);
  const out = await c.classify([null, row, row]);
  assert.deepEqual(seen.dims, [2, 42]);
  assert.equal(out[0], null);
  assert.equal(out[1][0], 1);
  assert.equal(out[2][1], 1);
});

test('tryClassify skips frames while an inference is running', async () => {
  let finish;
  const ort = fakeOrt(() => new Promise(resolve => { finish = () => resolve({ probs: { data: new Float32Array(7) } }); }));
  const c = new GestureClassifier(ort.session, CLASSES, ort);
  const results = [];
  assert.equal(c.tryClassify([new Float64Array(42)], r => results.push(r)), true);
  assert.equal(c.tryClassify([new Float64Array(42)], r => results.push(r)), false);
  finish();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(results.length, 1);
  assert.equal(c.tryClassify([new Float64Array(42)], r => results.push(r)), true);
});

test('a failed inference reports the error and frees the classifier', async () => {
  const ort = fakeOrt(async () => { throw new Error('boom'); });
  const c = new GestureClassifier(ort.session, CLASSES, ort);
  const errors = [];
  c.tryClassify([new Float64Array(42)], () => {}, e => errors.push(e.message));
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(errors, ['boom']);
  assert.equal(c.busy, false);
});

test('loads the real model and returns 7 probabilities that sum to 1', async () => {
  const c = await GestureClassifier.load(path.join(ROOT, 'models', 'gestures.onnx'),
                                         path.join(ROOT, 'models', 'labels.json'));
  assert.deepEqual(c.labels, CLASSES);
  const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'ml', 'tests', 'fixtures', 'features.json'), 'utf8'));
  const row = Float64Array.from(fixture.cases.find(k => k.expected !== null).expected);
  const [probs] = await c.classify([row]);
  assert.equal(probs.length, 7);
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-4);
});

test('load rejects when the model file is missing', async () => {
  await assert.rejects(GestureClassifier.load(path.join(ROOT, 'models', 'missing.onnx'),
                                              path.join(ROOT, 'models', 'labels.json')));
});

test('a frame with two hands classifies in under 2 ms on average', async () => {
  const c = await GestureClassifier.load(path.join(ROOT, 'models', 'gestures.onnx'),
                                         path.join(ROOT, 'models', 'labels.json'));
  const row = new Float64Array(42).fill(0.1);
  await c.classify([row, row]);                                  // warm-up
  const runs = 200;
  const start = process.hrtime.bigint();
  for (let i = 0; i < runs; i++) await c.classify([row, row]);
  const msPerFrame = Number(process.hrtime.bigint() - start) / 1e6 / runs;
  assert.ok(msPerFrame < 2, `${msPerFrame.toFixed(3)} ms per frame`);
});
