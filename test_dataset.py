import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from common import check_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for split in ('train', 'val'):
            (self.root / 'images' / split).mkdir(parents=True)
            (self.root / 'labels' / split).mkdir(parents=True)
        for i in range(100):
            split = 'train' if i < 80 else 'val'
            cv2.imwrite(str(self.root / 'images' / split / f'{i}.png'), np.full((12, 12, 3), i, np.uint8))
            (self.root / 'labels' / split / f'{i}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        self.config = self.root / 'data.yaml'
        self.config.write_text('path: .\ntrain: images/train\nval: images/val\nnames: [pen]\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_split_and_content_fingerprint(self):
        _, counts, before = check_dataset(self.config)
        self.assertEqual(counts['train']['images'], 80)
        self.assertEqual(counts['val']['positive_images'], 20)
        (self.root / 'labels/train/0.txt').write_text('0 0.5 0.5 0.4 0.4\n')
        self.assertNotEqual(before, check_dataset(self.config)[2])

    def test_cross_split_duplicate_is_rejected(self):
        shutil.copy2(self.root / 'images/train/0.png', self.root / 'images/val/80.png')
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            check_dataset(self.config)

    def test_invalid_box_and_missing_label_are_rejected(self):
        label = self.root / 'labels/train/0.txt'
        label.write_text('0 0.99 0.5 0.5 0.5\n')
        with self.assertRaisesRegex(ValueError, 'Invalid bounding box'):
            check_dataset(self.config)
        label.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            check_dataset(self.config)


if __name__ == '__main__':
    unittest.main()
