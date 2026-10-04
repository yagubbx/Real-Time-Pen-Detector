"""Shared paths, dataset checks and validation metrics."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs' / 'settings'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs' / 'matplotlib'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_threads(count):
    import torch
    from ultralytics.utils import torch_utils
    # CPU device selection otherwise restores Ultralytics' own thread default.
    torch_utils.NUM_THREADS = count
    torch.set_num_threads(count)


def check_dataset(config):
    import math
    import cv2
    import yaml

    config = Path(config).resolve()
    spec = yaml.safe_load(config.read_text(encoding='utf-8'))
    base = (config.parent / spec.get('path', '.')).resolve()
    names = spec['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if sorted(names) != list(range(len(names))):
        raise ValueError('Class IDs must start at 0 and be consecutive.')
    counts, seen = {}, {}
    digest = hashlib.sha256()
    for split in ('train', 'val'):
        folder = base / spec[split]
        images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not images:
            raise ValueError(f'No images: {folder}')
        positives, boxes, classes = 0, 0, set()
        for image in images:
            if cv2.imread(str(image)) is None:
                raise ValueError(f'Unreadable image: {image}')
            parts = list(image.parts)
            if 'images' not in parts:
                raise ValueError('Use the YOLO images/ and labels/ folder structure.')
            parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing label: {label}; negatives need an empty .txt.')
            lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
            for fields in lines:
                if len(fields) != 5:
                    raise ValueError(f'Expected class x y width height: {label}')
                cls, x, y, w, h = map(float, fields)
                valid = all(math.isfinite(v) for v in (cls, x, y, w, h))
                valid = valid and cls.is_integer() and int(cls) in names
                valid = valid and 0 < w <= 1 and 0 < h <= 1
                valid = valid and x - w/2 >= -1e-5 and x + w/2 <= 1.00001
                valid = valid and y - h/2 >= -1e-5 and y + h/2 <= 1.00001
                if not valid:
                    raise ValueError(f'Invalid bounding box: {label}: {fields}')
                classes.add(int(cls))
            checksum = file_hash(image)
            if checksum in seen:
                raise ValueError(f'Duplicate image: {image} and {seen[checksum]}')
            seen[checksum] = str(image)
            digest.update(f'{split}/{image.name}:{checksum}:'.encode())
            digest.update(label.read_bytes())
            positives += bool(lines)
            boxes += len(lines)
        if classes != set(names):
            raise ValueError(f'{split} does not contain every class.')
        counts[split] = {'images': len(images), 'positive_images': positives, 'boxes': boxes}
    if sum(c['positive_images'] for c in counts.values()) < 100:
        raise ValueError('At least 100 distinct labeled positive images are required.')
    spec.update(path=str(base), names=names)
    spec.pop('download', None)
    return spec, counts, digest.hexdigest()


def metric_dict(metrics, names):
    return {
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'per_class': {
            names[int(cls)]: dict(zip(('precision', 'recall', 'mAP50', 'mAP50_95'),
                                     map(float, metrics.box.class_result(i))))
            for i, cls in enumerate(metrics.box.ap_class_index)
        },
    }
