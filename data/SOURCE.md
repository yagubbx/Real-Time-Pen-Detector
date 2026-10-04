# Dataset Source

Source: [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html).
Class: `Pen` (`/m/0k1tl`). The 124 photos come from the validation and test sets of the
V5 box labels. This project uses its own split: 99 training images and 25 validation
images (seed 42). These are not official Open Images benchmark results.

Images are named `pen_1.jpg` to `pen_124.jpg`. Each image has a label file with
the same number, such as `pen_1.txt`. The manifest links these names to the original image IDs.

Photos with group boxes or boxes marked as drawings were left out.
The boxes were changed to YOLO format. The JPEG files downloaded from CVDF were not changed.
Random image changes during training are used only on the training set.
Before training, the script checks for duplicate files using SHA-256 file hashes.
Different photos of similar scenes may still be present. The validation set is small,
so it cannot prove that the model works well everywhere. There are no separate photos without pens.

Each photo's author, source link, license, original boxes, and file hash are in `manifest.json`.
Photos use the CC BY 2.0 license. Box labels use CC BY 4.0.
See the [Open Images license details](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).

These public images already have labels. To label new webcam photos by hand,
follow the [Roboflow](https://roboflow.com/annotate) steps in `BONUS.md`.
