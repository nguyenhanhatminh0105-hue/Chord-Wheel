# Gesture control: design

Chord-Wheel can already be played hands-free: MediaPipe tracks each palm, and
whichever slice of a wheel the palm hovers over is played or selected. The
weakness is that hovering is the only control. Moving the left hand across the
note wheel plays every note it passes, and the right hand must travel to the
quality wheel to change chords.

This adds a small neural network that recognises hand poses from MediaPipe's
landmarks, so a pose says what the hand means: play or just move, and which
chord quality to use.

## 1. Behaviour

**Poses.** The model outputs seven classes: `fist`, `palm`, `one`, `peace`,
`three`, `four`, and `other` (in-between or unclear poses).

**Left hand: play or move.**

| Pose | Over a note slice | Elsewhere |
|---|---|---|
| `fist` | plays that note (sliding to another slice plays the new one) | releases |
| `palm` | moves silently; releases a held chord | releases |
| `other` | keeps the previous state | keeps the previous state |

**Right hand: chord quality by shape.**

| Pose | Quality |
|---|---|
| `one` | maj |
| `peace` | min |
| `three` | dom7 |
| `four` | maj7 |
| `palm` | min7 |
| `fist` | sus4 |
| `other` | unchanged; hovering over the quality wheel still works |

The remaining qualities (dim, aug, sus2, m7b5) stay reachable by hovering or
with the mouse.

**Smoothing.** Each hand keeps its last 5 predictions. A pose takes effect
when it wins a majority of those 5 frames and its mean probability over its
winning frames is at least 0.7. `other` never changes anything. This removes
single-frame flicker and adds a natural ~100 ms of hysteresis at 30-60 fps.

**Feedback.** A small label beside each tracked hand shows the smoothed pose
and its confidence, for example `fist 96%`.

**Toggle.** A "Gestures" button turns pose control on and off. Off means
exactly today's position-only behaviour.

**Recording mode.** `R` toggles a recording mode used to build the webcam test
set. While it is on, holding a number key labels every frame with that pose
(`1` fist, `2` palm, `3` one, `4` peace, `5` three, `6` four, `7` other), and
each labelled frame is appended to `ml/data/recordings/<session>.jsonl`
(git-ignored). Record with one hand in view at a time.

## 2. Data

**HaGRID** (HAnd Gesture Recognition Image Dataset, v2) provides, for over a
million photos of tens of thousands of people, each hand's gesture label, its
21 MediaPipe landmarks as `(x, y)` image-relative coordinates, and an
anonymised person id. It does not record image sizes, or whether a hand is
left or right. Only the annotations are needed (a 719 MB
download); no images are downloaded. HaGRID is licensed under CC BY-SA 4.0
with project-specific terms; it is credited in the model card, and neither
the data nor any derived dataset is committed.

Classes used: HaGRID's `fist`, `palm`, `one`, `peace`, `three`, `four`, and,
as `other`, both `no_gesture` and four common out-of-vocabulary gestures
(`like`, `ok`, `rock`, `call`), so a pose outside the vocabulary maps to
`other` rather than to the nearest pose. HaGRID has several three-finger classes; the one
whose raised fingers are index, middle and ring is used. Before training, the
mean normalised pose of each chosen class is plotted to confirm its shape, and
the annotation field names are read from the downloaded files rather than
assumed.

**Split by person.** HaGRID's official train, validation and test sets are
already split by person id, so no person appears in two sets and the test
score measures people the model has never seen. A test checks this on the
downloaded annotations.

**Webcam test set.** About ten minutes recorded in the app's recording mode,
in two sessions on different days or lighting. Each JSON line holds
`{session, t, label, handedness, width, height, landmarks}` with all 21
landmarks as `[x, y, z]`. Only landmarks are stored, never images. The
recordings are git-ignored; the evaluation numbers derived from them are
committed.

## 3. Features

Every hand becomes 42 numbers by the same procedure in Python
(`ml/gesturenet/features.py`) and JavaScript (`gestures/features.js`):

1. **Pixel-proportional coordinates.** Multiply each `x` by the image width
   and each `y` by the image height, so one unit means the same distance in
   both directions. (MediaPipe normalises `x` and `y` by different amounts on
   a non-square image.) HaGRID records no image sizes, so for HaGRID every
   `x` is multiplied by one dataset-wide aspect factor, estimated before
   training as the factor that makes a hand's wrist-to-knuckle length,
   relative to its knuckle span, the same whether the hand is upright or
   turned sideways. Training augmentation (section 4) covers the remaining
   variation.
2. **Translate.** Subtract the wrist (landmark 0) from every point.
3. **Rotate upright.** With `v` the middle-finger knuckle (landmark 9) and
   `s = |v|`, map every point `(x, y)` to
   `((-v.y·x + v.x·y) / s, (-v.x·x - v.y·y) / s)`. This turns `v` to point
   straight up (negative `y`, image convention).
4. **Scale.** Divide every point by `s`, so the knuckle sits at `(0, -1)`.
5. **Flatten** to `[x0, y0, x1, y1, …, x20, y20]`.

If `s < 1e-6` the frame is skipped (treated as `other`). Only `x` and `y` are
used, because HaGRID has no `z`.

There is no left/right mirroring: HaGRID does not say which hand is which,
so training flips hands horizontally at random (section 4) and the model
learns both. Mirroring a hand negates every `x` feature and leaves every `y`
feature unchanged.

## 4. Model and training

- **Network:** 42 → 128 → 64 → 7, ReLU activations, dropout 0.2 after each
  hidden layer, softmax output. About 14,000 weights.
- **Loss:** cross-entropy weighted by inverse class frequency.
- **Optimiser:** AdamW, learning rate 1e-3, weight decay 1e-4, batch 512.
- **Augmentation**, applied to raw coordinates before the feature transform:
  a horizontal flip with probability 0.5, rotation by ±15°, scale by ±10 %,
  independent horizontal and vertical
  stretch by ±25 % (robustness to aspect ratio), and Gaussian noise per
  landmark with a standard deviation of 1 % of the wrist-to-knuckle distance
  `s`.
- **Stopping:** at most 60 epochs; stop when validation macro-F1 has not
  improved for 8 epochs; keep the best checkpoint.
- **Reproducibility:** fixed seeds for Python, NumPy and PyTorch; the split
  depends only on person ids and a seed.
- **Tracking:** loss, accuracy and macro-F1 per epoch are logged with Trackio
  (a local dashboard, optionally synced to a Hugging Face Space).

## 5. Export and runtime

- **ONNX** (opset 18, the minimum for PyTorch's current `torch.export`-based exporter): input `landmarks`, float32 `[N, 42]`; output `probs`,
  float32 `[N, 7]`; class names in `models/labels.json`. The export is
  checked against PyTorch on 1,000 test samples (max absolute difference
  below 1e-5).
- **In the app:** `onnxruntime-node` in the renderer (the app already runs
  with `nodeIntegration`), one session created at start-up. Each frame runs a
  single batch with one row per tracked hand. Budget: under 2 ms per frame on
  a laptop CPU.

## 6. Evaluation

Three test sets, each scored for accuracy, per-class precision and recall,
macro-F1, and a confusion matrix:

1. **HaGRID test split**: unseen people.
2. **Webcam recordings**: the real-world number for this app and camera.
3. **Rules baseline** on both sets, for comparison: a finger is "raised" when
   its tip is farther from the wrist than its middle joint (PIP); the thumb
   when its tip is farther from the index knuckle than its IP joint. Raised
   fingers map to poses (none → `fist`, all five → `palm`, index only →
   `one`, index and middle → `peace`, index to ring → `three`, index to
   little → `four`, anything else → `other`).

`ml/evaluate.py` writes `ml/reports/metrics.json` and confusion-matrix PNGs.
The README shows a model-versus-rules table on both test sets.

## 7. App integration

| Module | Responsibility |
|---|---|
| `gestures/features.js` | Section 3, the feature transform. |
| `gestures/classifier.js` | Loads `models/gestures.onnx`; `classify(hands) → [{probs}]`. |
| `gestures/smoother.js` | Per-hand 5-frame majority vote and threshold (section 1). |
| `gestures/actions.js` | Pure function: smoothed poses and palm positions to actions: play note *i*, release, set quality *q*. |
| `gestures/recorder.js` | Recording mode; appends JSON lines. |
| `gestures/controller.js` | Glue for the frame loop: features, classifier, smoother and actions, with the app's press, slide, release and set-quality functions passed in, so the whole path is testable without Electron. |

`renderer.js` changes only in `onHandResults` (which calls the controller when
gestures are on), plus the toggle button, the per-hand labels and the
recording keys.

## 8. Errors and fallbacks

- The model file is missing or fails to load: gestures are disabled, the
  toggle shows "Gestures unavailable", and the app behaves as today. Nothing
  throws into the frame loop.
- Inference throws on a frame: that frame counts as `other`.
- Degenerate landmarks (`s < 1e-6`): the frame counts as `other`.

## 9. Testing and CI

**JavaScript (`node:test`):**
- Feature parity: Python writes `ml/tests/fixtures/features.json` (raw
  landmarks and expected features); `features.js` must match within 1e-9.
- Smoother: majority, the 0.7 threshold, `other` changing nothing, and
  flicker (a single odd frame) being ignored.
- Actions: each row of the tables in section 1.

**Python (`pytest`):**
- HaGRID parsing on a small fixture file in the real annotation format.
- Feature invariance: translating, scaling or rotating a hand leaves its
  features unchanged; mirroring it negates every `x` feature.
- The split never puts one person in two sets.
- Training smoke test on synthetic data (loss decreases; runs in seconds).
- ONNX parity with PyTorch.
- Rules baseline on hand-built poses.

**End to end:** recorded landmark frames are fed through the running Electron
app's `onHandResults` over the Chrome DevTools Protocol (no camera), checking
that a fist over a note plays it, a palm releases it, and a right-hand
`peace` sets the quality to minor.

**CI:** two GitHub Actions jobs on Ubuntu, Python (pytest) and Node
(`npm test`). Both use fixtures only; CI never downloads HaGRID or trains a
full model.

## 10. Documentation and demo

- **README:** a "Gesture control" section with the pose tables, the
  model-versus-rules table and a confusion matrix.
- **`ml/MODEL_CARD.md`:** data source and licence, classes, metrics on both
  test sets, and limitations (2D landmarks only, a webcam test set from one
  person, sensitivity to lighting through MediaPipe).
- **`docs/how-it-works.md`:** the feature transform with a worked example,
  how the network and training work, ONNX export, the smoother, and why the
  split is by person (data leakage).
- **Live demo:** a Hugging Face Space (Gradio) in `ml/space/`. A visitor
  uploads a hand photo or takes a webcam snapshot; MediaPipe finds the
  landmarks, the ONNX model classifies them, and the page shows the top
  three poses with probabilities over the landmark overlay.

## 11. Layout

```
gestures/           features.js  classifier.js  smoother.js  actions.js  recorder.js
models/             gestures.onnx  labels.json
ml/                 pyproject.toml  README.md  MODEL_CARD.md
ml/gesturenet/      hagrid.py  features.py  dataset.py  model.py  train.py
                    export.py  evaluate.py  baseline.py  recordings.py
ml/scripts/         download_hagrid.py
ml/tests/           test_*.py  fixtures/
ml/reports/         metrics.json  confusion_*.png
ml/space/           app.py  requirements.txt  README.md
test/               existing tests + gestures.*.test.js
docs/               gesture-control-design.md  how-it-works.md
```

## 12. Decisions and trade-offs

- **Landmarks, not pixels.** MediaPipe already solves finding the hand; a
  classifier on 42 numbers is tiny, fast, and far less sensitive to
  lighting and background than one on images. The cost is depending on
  MediaPipe's landmark quality.
- **A small MLP.** The input is a fixed-length vector with no spatial grid or
  sequence, so convolutions or recurrence add little; per-frame
  classification plus smoothing handles time.
- **ONNX Runtime.** The standard way to ship a PyTorch model into a non-Python
  application; the same file runs in the app and in the demo.
- **Split by person.** Frames of one person are highly correlated; a random
  split would leak them across sets and overstate accuracy.
- **A rules baseline.** It shows whether learning was needed at all, and where
  it helps.
