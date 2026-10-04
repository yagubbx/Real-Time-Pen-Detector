"""Capture images for manual labeling in Roboflow or CVAT."""
import argparse
from pathlib import Path

from common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=int, default=0)
    p.add_argument('--session', required=True, help='e.g. bright-desk, dim-room, clutter')
    a = p.parse_args()
    if not a.session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.session):
        p.error('Use letters, numbers, - and _ for the session name.')
    import cv2
    folder = ROOT / 'captures' / a.session
    folder.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.stem[4:]) for base in (ROOT / 'data/images', ROOT / 'captures')
               for p in base.rglob('pen_*.jpg') if p.stem[4:].isdigit()]
    next_number = max(numbers, default=0) + 1
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Camera unavailable. Try --source 1.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Could not read camera frame.')
            preview = frame.copy()
            cv2.putText(preview, 'SPACE: save | Q: quit', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 230, 80), 2)
            cv2.imshow('Collect', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                path = folder / f'pen_{next_number}.jpg'
                while path.exists():
                    next_number += 1
                    path = folder / f'pen_{next_number}.jpg'
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f'Could not save {path}')
                print(path)
                next_number += 1
            if key in (ord('q'), 27) or cv2.getWindowProperty('Collect', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
