"""Reproduce held-out detection metrics; never substitute these for a live test."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from common import ROOT, check_dataset, file_hash, metric_dict, set_threads
from tracking import iou


def threshold_sweep(model, spec, imgsz, device):
    import cv2
    counters = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in (0.15, 0.45, 0.75)}
    folder = Path(spec['path']) / spec['val']
    images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    for image in images:
        frame = cv2.imread(str(image))
        height, width = frame.shape[:2]
        parts = list(image.parts)
        parts[len(parts)-1-parts[::-1].index('images')] = 'labels'
        labels = Path(*parts).with_suffix('.txt')
        truth = []
        for line in labels.read_text().splitlines():
            if not line.strip():
                continue
            cls, x, y, w, h = map(float, line.split())
            truth.append((int(cls), [(x-w/2)*width, (y-h/2)*height,
                                     (x+w/2)*width, (y+h/2)*height]))
        result = model.predict(frame, conf=0.15, iou=0.45, imgsz=imgsz,
                               device=device, verbose=False)[0]
        predicted = sorted(result.boxes.data.cpu().tolist(), key=lambda b: b[4], reverse=True)
        for threshold, count in counters.items():
            matched = set()
            for box in predicted:
                if box[4] < threshold:
                    continue
                candidates = [(iou(box[:4], gt), index) for index, (cls, gt) in enumerate(truth)
                              if index not in matched and cls == int(box[5])]
                best = max(candidates, default=(0, -1))
                if best[0] >= 0.5:
                    matched.add(best[1])
                    count['tp'] += 1
                else:
                    count['fp'] += 1
            count['fn'] += len(truth)-len(matched)
    for count in counters.values():
        count['precision'] = count['tp']/(count['tp']+count['fp']) if count['tp']+count['fp'] else None
        count['recall'] = count['tp']/(count['tp']+count['fn']) if count['tp']+count['fn'] else None
    return counters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=ROOT / 'weights/best.pt')
    p.add_argument('--data', type=Path, default=ROOT / 'data/data.yaml')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--sweep', action='store_true', help='Also compare confidence 0.15, 0.45 and 0.75 at IoU 0.5')
    a = p.parse_args()
    if min(a.imgsz, a.threads) < 1:
        p.error('imgsz and threads must be positive.')
    if not a.weights.is_file():
        p.error('Custom checkpoint is missing.')
    import yaml
    from ultralytics import YOLO
    set_threads(a.threads)
    spec, counts, fingerprint = check_dataset(a.data)
    config = ROOT / 'runs/evaluation-data.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(spec), encoding='utf-8')
    model = YOLO(str(a.weights))
    if model.names != spec['names']:
        p.error('Dataset classes do not match the checkpoint.')
    report = ROOT / 'reports/validation' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    metrics = model.val(data=str(config), imgsz=a.imgsz, device=a.device,
                        conf=0.001, workers=0, batch=8, plots=True,
                        project=str(report.parent), name=report.name)
    output = report / 'metrics.json'
    record = {'weights_sha256': file_hash(a.weights), 'dataset_sha256': fingerprint,
              'dataset': counts, 'imgsz': a.imgsz, 'validation_confidence': 0.001,
              'device': a.device, 'threads': a.threads, 'metrics': metric_dict(metrics, model.names)}
    if a.sweep:
        record['threshold_sweep_iou50'] = threshold_sweep(model, spec, a.imgsz, a.device)
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
