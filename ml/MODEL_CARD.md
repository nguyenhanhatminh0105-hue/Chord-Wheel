# Model card: Chord-Wheel gesture model

## Model

- **Task:** name the pose of one hand from its 21 MediaPipe landmarks, as one of seven classes: `fist`, `palm`, `one`, `peace`, `three`, `four`, `other`.
- **Input:** `landmarks`, float32 `[N, 42]`: the hand's `(x, y)` landmarks after the feature transform (translated to the wrist, turned upright, divided by the wrist-to-middle-knuckle length; see [how it works](../docs/how-it-works.md#the-feature-transform)).
- **Output:** `probs`, float32 `[N, 7]`: softmax probabilities, in the order of [`models/labels.json`](../models/labels.json).
- **Architecture:** a multilayer perceptron, 42 → 128 → 64 → 7, ReLU activations, dropout 0.2 after each hidden layer; 14,215 weights.
- **File:** [`models/gestures.onnx`](../models/gestures.onnx), 66 KB, ONNX opset 18, exported with PyTorch's `torch.export`-based exporter. On 1,000 test hands, ONNX Runtime's output differs from PyTorch's by at most 2.4e-7.
- **Runtime:** `onnxruntime-node` in the Electron app, one inference in flight at a time. A two-hand frame takes about 0.02 ms on the development laptop; a unit test fails if it averages over 2 ms.

## Training data

- **Source:** the hand-landmark annotations of [HaGRID v2](https://github.com/hukenovs/hagrid) (no images), licensed CC BY-SA 4.0 with the project's own terms. Neither the data nor anything derived from it is committed to this repository.
- **Classes:** HaGRID's `fist`, `palm`, `one`, `peace`, `three` and `four`. `other` combines HaGRID's `no_gesture` with four common gestures outside the vocabulary (`like`, `ok`, `rock`, `call`), so an unknown pose maps to `other` instead of the nearest known pose.
- **Split:** HaGRID's official train, validation and test sets, which are split by person; the calibration script checks that no person appears in two of them. 276,965 training hands, 36,194 validation hands, 59,578 test hands.
- **Aspect calibration:** HaGRID divides each `x` by the image width and each `y` by the image height but does not store the sizes. In 2,400 images sampled from HaGRID's 512-pixel release, which keeps each image's proportions, 73.8 % are portrait 3:4, so every HaGRID `x` is multiplied by 0.75 to give the hand its true shape.
- **Three-finger class:** of HaGRID's three-finger classes, `three` is the one whose raised fingers are index, middle and ring: they are in 96.5 % of its hands, against 0.1 % for `three2` and 0.2 % for `three3`.
- **Training:** cross-entropy weighted by inverse class frequency; AdamW, learning rate 1e-3, weight decay 1e-4, batch 512; augmentation by random mirroring, ±15° rotation, ±10 % scale, ±25 % independent horizontal and vertical stretch, and landmark noise of 1 % of the hand's size; early stopping on validation macro-F1, which peaked at 0.990 in epoch 15 of 23.

## Evaluation

| Test set | Hands | Model accuracy | Model macro-F1 | Rules accuracy | Rules macro-F1 |
|---|---|---|---|---|---|
| HaGRID test (people never seen in training) | 59,578 | 0.993 | 0.991 | 0.782 | 0.828 |

Recall per class on the HaGRID test split:

| | fist | palm | one | peace | three | four | other |
|---|---|---|---|---|---|---|---|
| Model | 0.992 | 1.000 | 0.989 | 0.994 | 0.988 | 0.996 | 0.993 |
| Rules | 0.901 | 0.995 | 0.752 | 0.765 | 0.905 | 0.984 | 0.681 |

The rules baseline counts raised fingers (a finger is raised when its tip is farther from the wrist than its middle joint; the thumb, when its tip is farther from the little finger's knuckle than its IP joint) and maps each set of raised fingers to a pose. Recall on `other` matters most in use: it is how often a hand that means nothing is left alone. The model's is 0.993; the rules' is 0.681. The test split holds 10,070 relaxed, non-gesturing hands (HaGRID's `no_gesture`, which includes the idle second hand in many photos), and the rules count five raised fingers on 74.7 % of them, so they become `palm`; so do 18 % of OK signs. That is also why the rules' precision on `palm` is only 0.367.

All numbers come from [`ml/reports/metrics.json`](reports/metrics.json); the confusion matrices are in [`ml/reports/`](reports/).

## Limitations

- **Two dimensions only.** HaGRID has no depth, so the model sees `x` and `y`. A pose seen edge-on, with fingers pointing at the camera, loses the information that tells it apart.
- **MediaPipe first.** The model is only as good as MediaPipe's landmarks, and those degrade in dim light, with motion blur, and when fingers hide one another.
- **HaGRID is not this camera.** The test split measures people the model never saw, photographed by HaGRID's contributors on their own phones. It does not measure the app's webcam, distance and lighting.
- **One aspect for all of HaGRID.** About one HaGRID image in four is not portrait 3:4, and its hands are trained slightly stretched or squeezed. The ±25 % stretch augmentation spans 86 % of the sampled image shapes.
- **`other` covers what its examples cover.** It is `no_gesture`, `like`, `ok`, `rock` and `call`. Other shapes outside the vocabulary may be read as the nearest of the six poses.
- **The rules were tuned once.** The thumb test was picked from five parameter-free variants by their score on the training split, then scored once on the test split.
