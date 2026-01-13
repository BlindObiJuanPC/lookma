import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from lookma.models import BodyRoiNetwork


import concurrent.futures


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

        # Thread pool for preprocessing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        self.min_lv = 0.0
        self.max_lv = 12.0

    def _process_single_crop(self, img):
        # Resize
        img_resized = cv2.resize(img, (256, 256))
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        t_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        t_img = TF.normalize(
            t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return t_img

    def preprocess_batch(self, crops):
        """Converts a list of BGR crops to a normalized tensor batch using threads."""
        if not crops:
            return None

        # Use thread pool to speed up resize/convert
        tensors = list(self.executor.map(self._process_single_crop, crops))

        return torch.stack(tensors).to(self.device)

    def predict_batch(self, crops, batch_size=64):
        """Runs inference on a list of image crops."""
        if not crops:
            return None, None

        # preprocess_batch now handles threading
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
        scale_factor=0.90,
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

        # 1. Flatten all windows first
        all_metadata = []  # (size_idx, win_tuple)
        for idx, size in enumerate(available_sizes):
            wins = self.generate_windows((h, w), size, stride_ratio)
            for win in wins:
                all_metadata.append((idx, win))

        total_windows = len(all_metadata)
        best_score = -1.0
        best_rect = None
        best_landmarks = None
        best_size_idx = 0

        # 2. Process in chunks
        chunk_size = 64

        for i in range(0, total_windows, chunk_size):
            if progress_callback:
                progress_callback(i, total_windows)

            chunk_meta = all_metadata[i : i + chunk_size]

            # Extract Crops for this chunk
            crops = []
            valid_indices = []  # Indices within the chunk that are valid

            for k, (s_idx, (x, y, win_w, win_h)) in enumerate(chunk_meta):
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
                    valid_indices.append(k)

            if not crops:
                continue

            # Predict
            preds, scores = self.predict_batch(crops, batch_size=chunk_size)

            # Find best in this batch
            if scores.size > 0:
                batch_best_idx = np.argmax(scores)
                batch_best_score = scores[batch_best_idx]

                if batch_best_score > best_score:
                    best_score = batch_best_score
                    # Map back to metadata
                    valid_k = valid_indices[batch_best_idx]
                    s_idx, rect = chunk_meta[valid_k]

                    best_size_idx = s_idx
                    best_rect = rect
                    best_landmarks = preds[batch_best_idx]

        return best_rect, best_score, best_landmarks, best_size_idx

    def refine_roi(
        self,
        image,
        initial_rect,
        refine_stride=0.05,
        refine_scale=0.95,
        min_refine_size=128,
        progress_callback=None,
    ):
        """
        Refines the ROI by treating the initial window as the search image.

        Args:
            image: BGR numpy image.
            initial_rect: (x, y, w, h) tuple (the "crop").
            refine_stride: Stride for internal scan (default 0.05).
            refine_scale: Scale factor for internal scan (default 0.95).
            min_refine_size: Minimum window size for internal scan.
            progress_callback: Optional func(current, total).

        Returns:
            (best_rect, best_score, best_landmarks)
            best_rect is in global coordinates.
        """
        x, y, w, h = initial_rect
        img_h, img_w = image.shape[:2]

        # Enforce bounds for crop
        cx1, cy1 = max(0, x), max(0, y)
        cx2, cy2 = min(img_w, x + w), min(img_h, y + h)

        crop = image[cy1:cy2, cx1:cx2].copy()

        # We need to find the best window *within* this crop.
        # find_best_roi scans up to max(h,w) of input.
        # Since 'crop' is the ROI, we essentially zoom in.

        sub_rect, sub_score, sub_landmarks, _ = self.find_best_roi(
            crop,
            min_size=min_refine_size,
            stride_ratio=refine_stride,
            scale_factor=refine_scale,
            start_with_max=True,  # Search largest first (fitting the crop)
            progress_callback=progress_callback,
        )

        if sub_rect is None:
            # Should not happen unless crop < min_size
            return initial_rect, -1.0, None

        # Transform sub_rect back to global coords
        sx, sy, sw, sh = sub_rect
        global_rect = (cx1 + sx, cy1 + sy, sw, sh)

        return global_rect, sub_score, sub_landmarks
