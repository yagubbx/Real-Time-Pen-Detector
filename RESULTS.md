# Results

YOLOv8n, one class (`pen`), 124 photos: 99 for training and 25 for validation.
Training: 40 epochs (passes through the data), image size 416, batch size 8, CPU, seed 42.
The first 10 backbone layers were kept fixed. The other layers were trained on this dataset.

| Validation metric | Result |
|---|---:|
| Precision | 78.9% |
| Recall | 66.7% |
| mAP@50 | 73.2% |
| mAP@50-95 | 43.1% |

Precision shows how many detections are correct. Recall shows how many real pens are found.
mAP measures detection quality at different score levels and box overlap rules.
The precision and recall above use the score level chosen by Ultralytics.
The table below uses fixed confidence settings and IoU >= 0.5 (box overlap).
Each predicted box can match only one labeled box.

| Confidence | Correct boxes (TP) | Wrong boxes (FP) | Missed pens (FN) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.15 | 29 | 10 | 13 | 74.4% | 69.0% |
| 0.45 | 25 | 2 | 17 | 92.6% | 59.5% |
| 0.75 | 16 | 1 | 26 | 94.1% | 38.1% |

Model file: `weights/best.pt`. Training logs, settings, and file hashes:
`reports/training/pen/`. Example results: [examples.jpg](reports/examples.jpg).
Final results were measured with `evaluate.py --sweep`. During training, the best model
was chosen with confidence set to 0.1. This small validation set was used to choose
the model. It is not a separate final test set.

At confidence 0.45, the model found pens on plain purple and dark backgrounds.
It missed the thin pen on a keyboard (`pen_109.jpg`), the pen pointing at the camera
(`pen_124.jpg`), and a very close view (`pen_104.jpg`). These were validation photos, not live tests.

## Live test

| Condition | Status |
|---|---|
| Bright light, plain background | Not tested yet |
| Low light | Not tested yet |
| Busy background | Not tested yet |

Run the `--test` command in `START.md`. Update this table with your real results.
`summary.json` checks if a pen is present in each frame. It does not measure box mAP.
`demo.avi` is saved at 10 FPS with timing based on real elapsed time. This is the video
playback rate, not the model's FPS. Actual frame processing times are in `frames.csv`.

Live webcam FPS and Raspberry Pi FPS have not been measured. The included dataset has
no separate photos without pens. Test an empty desk and objects that look like pens.
Results may be worse with webcam images, distant pens, or partly hidden pens.
Smoothing makes boxes more stable, but waiting for 2 frames adds a small delay.
