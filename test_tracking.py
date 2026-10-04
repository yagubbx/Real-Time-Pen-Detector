import unittest

from tracking import Smoother
from webcam import summarize


def detection(x=0, cls=0, conf=0.8):
    return cls, conf, [x, 0, x+20, 20]


class TrackingTests(unittest.TestCase):
    def test_confirmation_and_one_event_per_streak(self):
        tracker = Smoother(min_hits=2, alert_frames=3)
        self.assertEqual(tracker.update([detection()]), ([], []))
        self.assertEqual(len(tracker.update([detection(1)])[0]), 1)
        self.assertEqual(len(tracker.update([detection(2)])[1]), 1)
        self.assertEqual(tracker.update([detection(3)])[1], [])

    def test_missing_frame_hides_box_and_resets_streak(self):
        tracker = Smoother(min_hits=1, alert_frames=2)
        tracker.update([detection()])
        self.assertEqual(len(tracker.update([detection()])[1]), 1)
        self.assertEqual(tracker.update([]), ([], []))
        self.assertEqual(tracker.update([detection()])[1], [])
        self.assertEqual(len(tracker.update([detection()])[1]), 1)

    def test_one_to_one_and_class_matching(self):
        tracker = Smoother(min_hits=1)
        tracker.update([detection(), detection(5), detection(cls=1)])
        tracks, _ = tracker.update([detection(2)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].cls, 0)

    def test_ema_dampens_motion(self):
        tracker = Smoother(min_hits=1, alpha=0.5)
        tracker.update([detection()])
        tracks, _ = tracker.update([detection(4)])
        self.assertEqual(tracks[0].box[0], 2)

    def test_expiry_and_threshold_reset(self):
        tracker = Smoother(min_hits=1, max_missed=1)
        first = tracker.update([detection()])[0][0].id
        tracker.update([])
        tracker.update([])
        self.assertEqual(tracker.tracks, [])
        self.assertNotEqual(tracker.update([detection()])[0][0].id, first)
        tracker.clear()
        self.assertEqual(tracker.update([]), ([], []))

    def test_unscored_frames_do_not_improve_metrics(self):
        rows = [dict(condition='bright', truth=t, raw=d, shown=d, loop_ms=100)
                for t, d in [(1, 1), (1, 0), (0, 1), (0, 0), (-1, 1)]]
        result = summarize(rows)
        self.assertEqual(result['bright']['scored_frames'], 4)
        self.assertEqual(result['bright']['presence_metrics']['raw']['precision'], 0.5)
        self.assertEqual(result['bright']['presence_metrics']['raw']['recall'], 0.5)
        self.assertEqual(result['bright']['fps'], 10)
        self.assertIsNone(result['dim']['fps'])


if __name__ == '__main__':
    unittest.main()
