# Getting Started

You need Python 3.11, Git, and a webcam. Use these commands in Windows PowerShell.
Replace `REPO_URL` with your GitHub repository link.

```powershell
git clone REPO_URL pen-detector
cd pen-detector
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python webcam.py
```

The dataset and trained model (`weights/best.pt`) are included. You do not need to train again.
If the camera does not open, add `--source 1`. Press `Q` to quit or `S` to turn smoothing on or off.
Smoothing makes boxes move less. A higher confidence setting hides boxes with low scores.

## Live test and alerts

```powershell
.venv\Scripts\python webcam.py --sound --alert-frames 5 --test
```

Test three real conditions: bright light and a plain background (`1`), low light (`2`),
and a busy background (`3`). In each condition, press `P` and show the pen for 15 seconds.
Press `U` to pause, remove the pen, then press `N` and wait 15 seconds.
Try different angles and distances. Press `Q` to finish.
The video, frame logs, and results are saved in `reports/live/`.

## Train again or check the model

```powershell
.venv\Scripts\python train.py --overwrite
.venv\Scripts\python evaluate.py --sweep
```

CPU training takes time. If your PyTorch has CUDA support, use `--device 0`.
For two classes and Raspberry Pi, read [BONUS.md](BONUS.md).
