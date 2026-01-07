import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from lookma.models import BodyRoiNetwork


class ROIFinder:
    def __init__(self, model_path, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load Model
        self.model = BodyRoiNetwork(backbone_name="resnet18", pretrained=False)
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            # Sanitize keys
            new_ckpt = {}
            for k, v in ckpt.items():
                new_ckpt[k.replace("_orig_mod.", "")] = v
            self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()

        self.min_lv = 0.0
        self.max_lv = 12.0

    def preprocess_batch(self, crops):
        """Converts a list of BGR crops to a normalized tensor batch."""
        tensors = []
        for img in crops:
            # Resize
            img_resized = cv2.resize(img, (256, 256))
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            t_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            t_img = TF.normalize(
                t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            tensors.append(t_img)

        if not tensors:
            return None

        return torch.stack(tensors).to(self.device)

    def predict_batch(self, crops, batch_size=64):
        """Runs inference on a list of image crops."""
        if not crops:
            return None, None

        full_batch = self.preprocess_batch(crops)
        num_items = full_batch.shape[0]
        preds_list = []

        with torch.no_grad():
            for i in range(0, num_items, batch_size):
                b_inputs = full_batch[i : i + batch_size]
                b_preds = self.model(b_inputs)
                preds_list.append(b_preds.cpu())

        all_preds = torch.cat(preds_list, dim=0).numpy()

        # Calculate Scores
        scores = []
        for i in range(len(all_preds)):
            avg_log_var = np.mean(all_preds[i, :, 2])
            t_avg = (avg_log_var - self.min_lv) / (self.max_lv - self.min_lv)
            conf_score = 1.0 - max(0.0, min(1.0, t_avg))
            scores.append(conf_score)

        return all_preds, np.array(scores)

    def generate_windows(self, img_shape, size, stride_ratio=0.10):
        """Generates (x, y, w, h) tuples."""
        h, w = img_shape[:2]
        windows = []
        stride = int(size * stride_ratio)
        if stride < 1:
            stride = 1

        start_offset = -size // 2
        x_limit = w - (size // 2) + stride
        x_range = list(range(start_offset, x_limit, stride))
        y_limit = h - (size // 2) + stride

        for y in range(start_offset, y_limit, stride):
            for x in x_range:
                windows.append((x, y, size, size))
        return windows

    def find_best_roi(
        self,
        image,
        min_size=256,
        stride_ratio=0.10,
        scale_factor=0.75,
        start_with_max=True,
        progress_callback=None,
    ):
        """
        Scans multiscale windows to find the best ROI.

        Args:
            image: BGR numpy image
            min_size: Minimum window dimension
            stride_ratio: Stride as percentage of window size
            scale_factor: Multiplier for next window size (e.g. 0.75)
            start_with_max: If True, starts with max(h,w), else min(h,w)
            progress_callback: Optional func(current, total)

        Returns:
            (best_rect, best_score, best_landmarks, best_size_idx)
            best_rect is (x, y, w, h)
        """
        h, w = image.shape[:2]

        # Generate sizes
        available_sizes = []
        curr = float(max(h, w) if start_with_max else min(h, w))
        while curr >= min_size:
            available_sizes.append(int(curr))
            curr *= scale_factor

        if not available_sizes:
            available_sizes = [min(h, w)]

        best_score = -1.0
        best_rect = None
        best_landmarks = None
        best_size_idx = 0

        total_steps = len(available_sizes)

        for i, size in enumerate(available_sizes):
            if progress_callback:
                progress_callback(i, total_steps)

            windows = self.generate_windows((h, w), size, stride_ratio)

            # Extract Crops
            crops = []
            valid_windows = []

            for x, y, win_w, win_h in windows:
                x1, y1 = x, y
                x2, y2 = x + win_w, y + win_h
                ix1, iy1 = max(0, x1), max(0, y1)
                ix2, iy2 = min(w, x2), min(h, y2)

                if ix2 > ix1 and iy2 > iy1:
                    roi = image[iy1:iy2, ix1:ix2]
                    crop_canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8) + 128
                    cx1, cy1 = ix1 - x1, iy1 - y1
                    cx2, cy2 = cx1 + (ix2 - ix1), cy1 + (iy2 - iy1)
                    crop_canvas[cy1:cy2, cx1:cx2] = roi
                    crops.append(crop_canvas)
                    valid_windows.append((x, y, win_w, win_h))

            if not crops:
                continue

            # Predict
            preds, scores = self.predict_batch(crops)

            # Find best in this batch
            if scores.size > 0:
                batch_best_idx = np.argmax(scores)
                batch_best_score = scores[batch_best_idx]

                if batch_best_score > best_score:
                    best_score = batch_best_score
                    best_rect = valid_windows[batch_best_idx]
                    best_landmarks = preds[batch_best_idx]
                    best_size_idx = i

        return best_rect, best_score, best_landmarks, best_size_idx
