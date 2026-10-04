"""Class-aware IoU matching, box smoothing and consecutive-frame events."""
from dataclasses import dataclass


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
    return w*h / max(area-w*h, 1e-9)


@dataclass
class Track:
    id: int
    cls: int
    box: list
    raw_box: list
    confidence: float
    streak: int = 1
    missed: int = 0
    alerted: bool = False


class Smoother:
    def __init__(self, alpha=0.65, min_hits=2, max_missed=2, match_iou=0.15, alert_frames=5):
        self.alpha = alpha
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.match_iou = match_iou
        self.alert_frames = alert_frames
        self.tracks = []
        self.next_id = 1

    def clear(self):
        self.tracks.clear()

    def update(self, detections):
        # Detections are (class, confidence, xyxy); match each box at most once.
        pairs = sorted((iou(t.raw_box, d[2]), ti, di)
                       for ti, t in enumerate(self.tracks)
                       for di, d in enumerate(detections) if t.cls == d[0])
        used_t, used_d = set(), set()
        for score, ti, di in reversed(pairs):
            if score < self.match_iou or ti in used_t or di in used_d:
                continue
            t, d = self.tracks[ti], detections[di]
            t.box = [self.alpha * new + (1-self.alpha)*old for old, new in zip(t.box, d[2])]
            t.raw_box, t.confidence = list(d[2]), d[1]
            t.streak = t.streak + 1 if t.missed == 0 else 1
            t.missed = 0
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.streak = 0
                t.alerted = False
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for di, (cls, conf, box) in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(self.next_id, cls, list(box), list(box), conf))
                self.next_id += 1
        events = []
        for t in self.tracks:
            if t.missed == 0 and t.streak >= self.alert_frames and not t.alerted:
                events.append(t)
                t.alerted = True
        # Keep identities briefly, but never draw boxes from missed frames.
        visible = [t for t in self.tracks if t.missed == 0 and t.streak >= self.min_hits]
        return visible, events
