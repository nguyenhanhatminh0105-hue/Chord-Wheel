# Gesture model: reproduce it

Everything here runs on a CPU. Commands are written for Windows from the repository root; on macOS or Linux, use `ml/.venv/bin/python` in place of `ml/.venv/Scripts/python`. Each step ends with the line shown after it (numbers from the run that produced `models/gestures.onnx`).

## 1. Environment

```bash
python -m venv ml/.venv
ml/.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ml/.venv/Scripts/python -m pip install -e "./ml[dev,track]"
ml/.venv/Scripts/python -m pytest ml/tests -q
```

```
52 passed
```

Python 3.10 or newer. The `track` extra installs Trackio for the training dashboard; leave it out if you do not need it.

## 2. Download HaGRID's annotations (719 MB, no images)

```bash
ml/.venv/Scripts/python ml/scripts/download_hagrid.py
```

```
extracted 39 files into …/ml/data/hagrid/annotations
```

## 3. Calibrate

Measures HaGRID's image aspect (it samples image sizes through the Hugging Face Dataset Viewer, which takes a few minutes), picks the three-finger class, and checks that the official splits share no person.

```bash
ml/.venv/Scripts/python ml/scripts/calibrate_hagrid.py ml/data/hagrid/annotations
```

```
aspect 0.75, three-finger class 'three', splits person-disjoint -> …/ml/gesturenet/hagrid_calibration.json
```

Optional: plot each class's mean normalised pose, a check that every class has the expected shape (writes `docs/mean-poses.png`).

```bash
ml/.venv/Scripts/python ml/scripts/plot_mean_poses.py ml/data/hagrid/annotations
```

## 4. Train

A few minutes; it stops early when validation macro-F1 stops improving. `--track` logs every epoch to Trackio (`trackio show --project chord-wheel-gestures` opens the dashboard).

```bash
ml/.venv/Scripts/python -m gesturenet.train --hagrid ml/data/hagrid/annotations --track
```

```
saved ml\artifacts\gestures.pt; best val macro-F1 0.9900
```

## 5. Export to ONNX

Writes `models/gestures.onnx` and `models/labels.json`, then checks ONNX Runtime against PyTorch on 1,000 test hands.

```bash
ml/.venv/Scripts/python -m gesturenet.export --hagrid ml/data/hagrid/annotations
```

```
wrote …\models\gestures.onnx (66356 bytes); max |onnx - torch| = 2.38e-07 on 1000 test hands
```

## 6. Evaluate

Scores the model and the finger-counting rules on HaGRID's test split; writes `ml/reports/metrics.json` and the confusion matrices. Add `--recordings ml/data/recordings` to score webcam recordings made with the app's recording mode as well.

```bash
ml/.venv/Scripts/python -m gesturenet.evaluate --hagrid ml/data/hagrid/annotations
```

```
hagrid_test  hands   59578   model acc 0.993 F1 0.991   rules acc 0.782 F1 0.828   model 'other' recall 0.993
```

## Files the app and its tests use

- `ml/scripts/make_fixtures.py` writes `ml/tests/fixtures/features.json`, the cases that prove the Python and JavaScript feature transforms agree.
- `ml/scripts/make_e2e_hands.py ml/data/hagrid/annotations` writes `test/fixtures/e2e_hands.json`, the hands `npm run e2e` feeds into the app.
