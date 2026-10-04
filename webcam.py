"""Live pen detection with measured FPS, smoothing and a scored test mode."""
import argparse
import csv
import json
import platform
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from common import ROOT, file_hash, set_threads
from tracking import Smoother

CONDITIONS = {ord('1'): 'bright', ord('2'): 'dim', ord('3'): 'clutter'}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=ROOT / 'weights/best.pt')
    p.add_argument('--source', default='0', help='Camera index or local video path')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--conf', type=float, default=0.45)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--alert-frames', type=int, default=5)
    p.add_argument('--sound', action='store_true')
    p.add_argument('--test', action='store_true', help='Record per-frame CSV and annotated clip')
    p.add_argument('--headless', action='store_true', help='Video smoke test; no display')
    p.add_argument('--max-frames', type=int, default=0)
    a = p.parse_args()
    if not a.weights.is_file():
        p.error(f'Custom weights missing: {a.weights}. See START.md.')
    if not 0.01 <= a.conf <= 1 or min(a.imgsz, a.threads, a.alert_frames) < 1 or a.max_frames < 0:
        p.error('Invalid threshold, size, thread count or frame count.')
    if a.headless and a.test:
        p.error('--test needs the window to label conditions and pen presence.')
    return a


def summarize(rows):
    summary = {}
    for condition in ('bright', 'dim', 'clutter'):
        selected = [r for r in rows if r['condition'] == condition and r['truth'] in (0, 1)]
        by_mode = {}
        for mode in ('raw', 'shown'):
            tp = sum(r['truth'] == 1 and r[mode] > 0 for r in selected)
            fp = sum(r['truth'] == 0 and r[mode] > 0 for r in selected)
            fn = sum(r['truth'] == 1 and r[mode] == 0 for r in selected)
            tn = sum(r['truth'] == 0 and r[mode] == 0 for r in selected)
            by_mode[mode] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                             'precision': tp/(tp+fp) if tp+fp else None,
                             'recall': tp/(tp+fn) if tp+fn else None}
        seconds = sum(r['loop_ms'] for r in selected)/1000
        summary[condition] = {
            'scored_frames': len(selected),
            'positive_frames': sum(r['truth'] == 1 for r in selected),
            'negative_frames': sum(r['truth'] == 0 for r in selected),
            'scored_seconds': seconds,
            'fps': len(selected)/seconds if seconds else None,
            'presence_metrics': by_mode,
        }
    return summary


def main():
    a = arguments()
    import cv2
    import numpy as np
    from ultralytics import YOLO

    set_threads(a.threads)
    model = YOLO(str(a.weights))
    names = model.names
    if not any('pen' in str(v).lower() for v in names.values()):
        raise ValueError('Load a custom pen/pencil checkpoint, not a generic COCO model.')
    model.predict(np.zeros((a.imgsz, a.imgsz, 3), dtype=np.uint8), imgsz=a.imgsz,
                  device=a.device, verbose=False)
    source = int(a.source) if a.source.isdecimal() else a.source
    if isinstance(source, str) and not Path(source).is_file():
        raise ValueError('Source must be a camera index or an existing local video.')
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Camera/video could not open. Close other camera apps or try --source 1.')
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    smoother = Smoother(alpha=1.0 if a.no_smooth else 0.65,
                        min_hits=1 if a.no_smooth else 2, alert_frames=a.alert_frames)
    history, rows = deque(maxlen=30), []
    conf, frame_number, condition, truth = a.conf, 0, 'unscored', -1
    total_loop_seconds = 0.0
    writer, csv_file, csv_writer, test_dir = None, None, None, None
    recorded_frames, recording_start = 0, None
    window = 'Pen detector'
    event_until = 0.0
    started = time.perf_counter()
    try:
        if not a.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.createTrackbar('Confidence %', window, round(conf*100), 100, lambda _: None)
        if a.test:
            test_dir = ROOT / 'reports/live' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            test_dir.mkdir(parents=True)
            csv_file = (test_dir / 'frames.csv').open('w', newline='', encoding='utf-8')
            csv_writer = csv.DictWriter(csv_file, fieldnames=(
                'frame', 'elapsed_s', 'condition', 'truth', 'threshold', 'smoothing', 'raw', 'shown', 'loop_ms'))
            csv_writer.writeheader()
        while True:
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, int):
                    raise RuntimeError('Camera stopped returning frames.')
                break
            if not a.headless:
                new_conf = max(0.01, cv2.getTrackbarPos('Confidence %', window)/100)
                if abs(new_conf-conf) > 1e-6:
                    smoother.clear()
                    event_until = 0.0
                    conf = new_conf
            result = model.predict(frame, imgsz=a.imgsz, conf=conf, iou=0.45,
                                   device=a.device, verbose=False)[0]
            detections = [(int(b[5]), float(b[4]), b[:4].tolist())
                          for b in result.boxes.data.cpu().numpy()]
            visible, events = smoother.update(detections)
            pen_events = [t for t in events if 'pen' in names[t.cls].lower()]
            if pen_events:
                event_until = time.perf_counter() + 1
                print(f'Pen detected for {a.alert_frames} consecutive frames.', flush=True)
                if a.sound:
                    if platform.system() == 'Windows':
                        import winsound
                        winsound.PlaySound('SystemAsterisk', winsound.SND_ALIAS | winsound.SND_ASYNC)
                    else:
                        print('\a', end='', flush=True)
            for track in visible:
                x1, y1, x2, y2 = map(int, track.box)
                color = (70, 220, 100) if track.cls == 0 else (40, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f'{names[track.cls]} {track.confidence:.2f}'
                cv2.putText(frame, text, (max(0, x1), max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            fps = len(history)/sum(history) if history else 0
            lines = [f'FPS {fps:.1f} | confidence {conf:.2f} | smoothing {not a.no_smooth}',
                     'Q: quit | S: smoothing']
            if a.test:
                lines += [f'1: bright  2: dim  3: clutter | {condition}',
                          f'P: pen present  N: absent  U: pause | truth {truth}']
            if time.perf_counter() < event_until:
                lines.append('PEN CONFIRMED')
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 25*len(lines)+8), (25, 25, 25), -1)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (8, 23+25*i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (240, 240, 240), 1, cv2.LINE_AA)
            if a.test:
                if writer is None:
                    writer = cv2.VideoWriter(str(test_dir / 'demo.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                             10, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Could not create the demo video.')
                    recording_start = tick
                # Resample to 10 FPS so playback follows elapsed time on slow CPUs too.
                target_frames = int((time.perf_counter()-recording_start)*10)+1
                while recorded_frames < target_frames:
                    writer.write(frame)
                    recorded_frames += 1
            key = -1
            if not a.headless:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
            # Measured after capture, inference, drawing, recording and display.
            elapsed = time.perf_counter() - tick
            total_loop_seconds += elapsed
            history.append(elapsed)
            frame_number += 1
            if csv_writer:
                row = dict(frame=frame_number, elapsed_s=round(time.perf_counter()-started, 4),
                           condition=condition, truth=truth, threshold=conf, smoothing=not a.no_smooth,
                           raw=sum('pen' in names[d[0]].lower() for d in detections),
                           shown=sum('pen' in names[t.cls].lower() for t in visible),
                           loop_ms=round(elapsed*1000, 3))
                rows.append(row)
                csv_writer.writerow(row)
            if key in (ord('q'), 27) or (a.max_frames and frame_number >= a.max_frames):
                break
            if not a.headless and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                a.no_smooth = not a.no_smooth
                smoother.alpha = 1 if a.no_smooth else 0.65
                smoother.min_hits = 1 if a.no_smooth else 2
                smoother.clear()
            if a.test:
                if key in CONDITIONS:
                    condition, truth = CONDITIONS[key], -1
                    smoother.clear()
                elif key in (ord('p'), ord('n'), ord('u')):
                    truth = {ord('p'): 1, ord('n'): 0, ord('u'): -1}[key]
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if not a.headless:
            cv2.destroyAllWindows()
        if test_dir is not None:
            summary = {'source': str(source), 'platform': platform.platform(),
                       'live_camera': isinstance(source, int),
                       'processor': platform.processor(), 'machine': platform.machine(),
                       'python': platform.python_version(),
                       'weights_sha256': file_hash(a.weights), 'imgsz': a.imgsz,
                       'device': a.device, 'threads': a.threads,
                       'metric': 'Pen presence per frame; not bounding-box accuracy.',
                       'video_playback_fps': 10, 'video_frames': recorded_frames,
                       'conditions': summarize(rows)}
            (test_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f'Test saved: {test_dir}')
    mean_fps = frame_number / total_loop_seconds if total_loop_seconds else 0
    print(f'Processed {frame_number} frames | loop FPS {mean_fps:.2f}')


if __name__ == '__main__':
    main()
