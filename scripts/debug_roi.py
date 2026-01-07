import sys
import os

sys.path.append(os.getcwd())

from lookma.roi import ROIFinder
import numpy as np
import cv2


# Mock ROIFinder to test logic without model
class MockROIFinder(ROIFinder):
    def __init__(self):
        self.min_lv = 0.0
        self.max_lv = 12.0
        # Skip model load

    def predict_batch(self, crops, batch_size=64):
        # Mock predictions
        N = len(crops)
        preds = np.zeros((N, 36, 3))
        scores = np.random.rand(N)
        return preds, scores


print("Initializing Mock Finder...")
finder = MockROIFinder()

# Mock Image (1080p)
h, w = 1080, 1920
img = np.zeros((h, w, 3), dtype=np.uint8)

print(f"Testing with Image: {w}x{h}")


def pbar(curr, total):
    print(f"Callback: curr={curr}, total={total}")


print("Running find_best_roi...")
finder.find_best_roi(
    img,
    min_size=256,
    stride_ratio=0.10,
    scale_factor=0.90,
    start_with_max=True,
    progress_callback=pbar,
)
