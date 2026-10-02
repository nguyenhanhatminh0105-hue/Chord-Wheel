---
title: Chord-Wheel Gestures
emoji: ✋
colorFrom: yellow
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Hand-pose recognition that runs in your browser
---

Recognises the hand poses that control [Chord-Wheel](https://github.com/nguyenhanhatminh0105-hue/Chord-Wheel): fist, palm, one, peace, three, four and other. Upload a photo of a hand or turn on the webcam.

Everything runs in the browser. MediaPipe's Hand Landmarker finds 21 landmarks per hand, and a network of 14,215 weights, trained on landmarks from [HaGRID](https://github.com/hukenovs/hagrid) (CC BY-SA 4.0) and exported to ONNX, classifies them with ONNX Runtime Web. The feature transform, classifier and smoother are the same files the desktop app runs. Details in the [model card](https://github.com/nguyenhanhatminh0105-hue/Chord-Wheel/blob/main/ml/MODEL_CARD.md).
