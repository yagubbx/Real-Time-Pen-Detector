# Extra Features and Tests

## Smoothing and alerts

These features are ready. The app matches boxes by their overlap (IoU).
It smooths box movement and shows a box after 2 detections in a row.
After 5 detections in a row, it gives one alert. If detection stops, the count starts again.
Use `--no-smooth` to compare the results without smoothing.
With `--sound`, Windows plays a system sound. Other systems use the terminal bell.

## Train a second class

The included model has only one class: `pen`. To train a new model for `pen` and `pencil`,
first collect your own images:

```powershell
.venv\Scripts\python collect.py --session bright-desk
```

Press SPACE to save a photo. Use different rooms, lights, and angles.
New photos use the next free name, such as `pen_125.jpg`.
Draw boxes for `pen` and `pencil` in [Roboflow Annotate](https://roboflow.com/annotate).
Keep photos from the same camera session in one split: training or validation.
Collect at least 50 images for each class and at least 100 different photos in total.
Both classes must be in both splits. Export in YOLOv8 detection format and unzip into `custom-data/`.
In `data.yaml`, set `path: .`, `train: train/images`, and `val: valid/images`.

```powershell
.venv\Scripts\python train.py --data custom-data/data.yaml --name two-class --output weights/two-class.pt
.venv\Scripts\python evaluate.py --weights weights/two-class.pt --data custom-data/data.yaml
.venv\Scripts\python webcam.py --weights weights/two-class.pt
```

Precision, recall, and mAP for each class are saved in `metrics.json`.
A trained two-class model is not included.

## Raspberry Pi

Use 64-bit Raspberry Pi OS, Python 3.11, and a USB webcam. Clone the same repository.
On Linux, set up the environment with `python3 -m venv .venv`, `source .venv/bin/activate`,
and `python -m pip install -r requirements.txt`.

```bash
python webcam.py --device cpu --threads 2 --imgsz 320 --test
```

Run the same test on your laptop with the same image size and confidence setting.
Compare FPS in the two `summary.json` files.
Setup and FPS have not been tested on a real Raspberry Pi.
