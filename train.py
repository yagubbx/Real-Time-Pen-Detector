"""Fine-tune YOLOv8n and retain the checkpoint and training evidence."""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

from common import ROOT, check_dataset, file_hash, metric_dict, set_threads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/data.yaml')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--freeze', type=int, default=10, help='Freeze early backbone layers; 0 trains every layer')
    p.add_argument('--device', default='cpu', help='cpu, 0 (CUDA), or mps')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--name', default='pen')
    p.add_argument('--output', type=Path, default=ROOT / 'weights/best.pt')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.output.exists() and not a.overwrite:
        p.error('Weights already exist. Choose --output or explicitly use --overwrite.')
    if min(a.epochs, a.imgsz, a.batch, a.threads) < 1 or a.freeze < 0:
        p.error('epochs, imgsz, batch and threads must be positive.')
    import yaml
    import torch
    import ultralytics
    from ultralytics import YOLO

    spec, counts, fingerprint = check_dataset(a.data)
    set_threads(a.threads)
    config = ROOT / 'runs' / f'{a.name}-data.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(spec), encoding='utf-8')
    model = YOLO(str(ROOT / 'runs/yolov8n.pt'))
    start = time.perf_counter()
    model.train(data=str(config), epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, device=a.device, workers=0, seed=42,
                deterministic=True, patience=0, optimizer='AdamW', lr0=0.0005,
                nbs=a.batch, freeze=a.freeze, warmup_epochs=0, warmup_bias_lr=0,
                degrees=10, translate=0.05, scale=0.2, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
                fliplr=0.5, mosaic=0, close_mosaic=0, cache=False, conf=0.1,
                project=str(ROOT / 'runs/train'), name=a.name, plots=True)
    trainer = model.trainer
    a.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trainer.best, a.output)
    report = ROOT / 'reports/training' / a.name
    report.mkdir(parents=True, exist_ok=True)
    for name in ('results.csv', 'args.yaml', 'results.png', 'confusion_matrix.png',
                 'PR_curve.png', 'BoxPR_curve.png', 'val_batch0_pred.jpg'):
        src = trainer.save_dir / name
        if src.is_file():
            shutil.copy2(src, report / name)
    record = {
        'base_model': 'yolov8n.pt', 'classes': spec['names'], 'seed': 42,
        'requested_epochs': a.epochs, 'imgsz': a.imgsz, 'batch': a.batch, 'freeze': a.freeze,
        'device': a.device, 'threads': torch.get_num_threads(), 'platform': platform.platform(),
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'elapsed_seconds': round(time.perf_counter() - start, 2),
        'dataset': counts, 'dataset_sha256': fingerprint,
        'weights_sha256': file_hash(a.output),
        'checkpoint_selection_confidence': 0.1,
        'validation': metric_dict(trainer.validator.metrics, model.names),
    }
    (report / 'metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    print(f'Saved: {a.output}')


if __name__ == '__main__':
    main()
