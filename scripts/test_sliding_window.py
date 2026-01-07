import argparse
import os

import ctypes
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from lookma.models import BodyRoiNetwork

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROI_CHECKPOINT = "experiments/checkpoints/roi/model_epoch_600.pth"


class WindowVisualizer:
    def __init__(self, img_path, stride_ratio=0.10):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        self.original_img = cv2.imread(img_path)
        if self.original_img is None:
            raise ValueError("Could not load image.")

        self.h, self.w = self.original_img.shape[:2]
        self.stride_ratio = stride_ratio

        # Load Model
        print(f"Loading model from {ROI_CHECKPOINT}...")
        self.model = BodyRoiNetwork(backbone_name="resnet18", pretrained=False)
        if os.path.exists(ROI_CHECKPOINT):
            ckpt = torch.load(ROI_CHECKPOINT, map_location=DEVICE)
            # Sanitize keys
            new_ckpt = {}
            for k, v in ckpt.items():
                new_ckpt[k.replace("_orig_mod.", "")] = v
            self.model.load_state_dict(new_ckpt)
        else:
            print("WARNING: Checkpoint not found! Inference will be random.")

        self.model.to(DEVICE)
        self.model.eval()

        # Display Config
        self.window_name = "Interactive Sliding Window"
        self.max_disp_size = 1200
        self.min_win_size = 256

        # Pre-calculate Window Sizes (Logarithmic: Max * 0.9^k)
        self.available_sizes = []
        curr = float(min(self.h, self.w))
        while curr >= self.min_win_size:
            self.available_sizes.append(int(curr))
            curr *= 0.90

        if not self.available_sizes:
            self.available_sizes = [min(self.h, self.w)]

        # --- AUTO SCAN FOR BEST WINDOW ---
        print("\nScanning for best initialization window...")
        best_size_idx, best_win_idx, best_score = self.scan_for_best_window()
        print(
            f"Match Found! Size Level: {best_size_idx}, Index: {best_win_idx}, Score: {best_score:.4f}\n"
        )

        # State
        self.size_idx = best_size_idx
        self.current_size = self.available_sizes[self.size_idx]
        self.current_idx = best_win_idx
        self.windows = []
        self.initialized = False

        # Update State
        self.dirty = False
        self.pending_size_idx = self.size_idx
        self.pending_stride_val = int(self.stride_ratio * 100)
        self.last_interaction_time = 0
        self.debounce_delay = 0.5

        # Mouse Detection Setup (Robust)
        self.mouse_check_func = None

        if os.name == "nt":
            try:
                user32 = ctypes.windll.user32
                if hasattr(user32, "GetKeyState"):
                    self.mouse_check_func = user32.GetKeyState
                    print("Mouse detection enabled: using GetKeyState.")
                elif hasattr(user32, "GetGetAsyncKeyState"):
                    self.mouse_check_func = user32.GetGetAsyncKeyState
                    print("Mouse detection enabled: using GetAsyncKeyState.")
                else:
                    print("Mouse detection functions not found. Using timer fallback.")
            except Exception as e:
                print(f"Failed to load user32: {e}. Using timer fallback.")

        # Grid Cache
        self.grid_img = None
        self.grid_latents = []
        self.last_grid_cols = 1
        self.current_landmarks = None
        self.current_confidences = []

        # Init Windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Create Trackbars
        cv2.createTrackbar(
            "Size Level",
            self.window_name,
            0,
            len(self.available_sizes) - 1,
            self.on_size_change,
        )
        cv2.setTrackbarPos("Size Level", self.window_name, self.size_idx)

        cv2.createTrackbar(
            "Stride %",
            self.window_name,
            int(self.stride_ratio * 100),
            100,
            self.on_stride_change,
        )
        cv2.createTrackbar("Index", self.window_name, 0, 100, self.on_index_change)

        # Mark as initialized and trigger generation manually once
        self.initialized = True
        self.process_update()

        # Force Index after first update
        if self.current_idx < len(self.windows):
            self.on_index_change(self.current_idx)
            try:
                cv2.setTrackbarPos("Index", self.window_name, self.current_idx)
            except Exception as e:
                print(f"Failed to set trackbar position: {e}")

    def scan_for_best_window(self):
        """Scans all size levels (>=64px) with 10% stride to find max confidence."""
        best_score = -1.0
        best_size_idx = 0
        best_win_idx = 0

        # Use a fixed stride for scanning as requested (10%)
        scan_stride = 0.10
        min_lv, max_lv = 0.0, 12.0

        # Filter levels >= 64
        valid_indices = [i for i, s in enumerate(self.available_sizes) if s >= 64]

        for idx in tqdm(valid_indices, desc="Scanning Levels"):
            size = self.available_sizes[idx]
            windows = self._make_windows(size, scan_stride)

            # Prepare Crops
            crop_list = []
            for x, y, w, h in windows:
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                ix1, iy1 = max(0, x1), max(0, y1)
                ix2, iy2 = min(self.w, x2), min(self.h, y2)

                if ix2 > ix1 and iy2 > iy1:
                    roi = self.original_img[iy1:iy2, ix1:ix2]
                    crop_canvas = np.zeros((h, w, 3), dtype=np.uint8) + 128
                    cx1, cy1 = ix1 - x1, iy1 - y1
                    cx2, cy2 = cx1 + (ix2 - ix1), cy1 + (iy2 - iy1)
                    crop_canvas[cy1:cy2, cx1:cx2] = roi
                    crop = crop_canvas
                else:
                    crop = np.zeros((h, w, 3), dtype=np.uint8)
                crop_list.append(crop)

            # Run Inference in Batches (to avoid OOM on huge grids)
            batch_size = 64
            num_crops = len(crop_list)

            if num_crops == 0:
                continue

            # Simple batch loop
            for b_start in range(0, num_crops, batch_size):
                b_end = min(b_start + batch_size, num_crops)
                batch_crops = crop_list[b_start:b_end]

                preds = self.run_inference(batch_crops)  # [B, 36, 3]

                # Calculate scores
                for lp, ldmks_raw in enumerate(preds):
                    # log_var is index 2
                    avg_log_var = np.mean(ldmks_raw[:, 2])

                    # Compute score based on our formula
                    t_avg = (avg_log_var - min_lv) / (max_lv - min_lv)
                    conf_score = 1.0 - max(0.0, min(1.0, t_avg))

                    if conf_score > best_score:
                        best_score = conf_score
                        best_size_idx = idx
                        best_win_idx = b_start + lp

        return best_size_idx, best_win_idx, best_score

    def _make_windows(self, size, stride_ratio):
        """Helper to generate window rects without affecting state."""
        windows = []
        stride = int(size * stride_ratio)
        if stride < 1:
            stride = 1

        start_offset = -size // 2
        x_limit = self.w - (size // 2) + stride
        x_range = list(range(start_offset, x_limit, stride))
        y_limit = self.h - (size // 2) + stride

        for y in range(start_offset, y_limit, stride):
            for x in x_range:
                windows.append((x, y, size, size))
        return windows

    def check_mouse_down(self):
        """Returns True if Left Mouse Button is down."""
        if not self.mouse_check_func:
            return False

        try:
            # 0x01 is VK_LBUTTON
            state = self.mouse_check_func(0x01)
            return (state & 0x8000) != 0
        except Exception as e:
            print(f"Failed to check mouse state: {e}")
            return False

    def generate_windows(self):
        """Generates single-scale windows based on current_size."""
        # Use the helper but update state variables
        self.windows = self._make_windows(self.current_size, self.stride_ratio)

        # Calculate cols for grid layout
        size = self.current_size
        stride = int(size * self.stride_ratio)
        if stride < 1:
            stride = 1

        start_offset = -size // 2
        x_limit = self.w - (size // 2) + stride
        x_range = list(range(start_offset, x_limit, stride))
        self.last_grid_cols = len(x_range)
        if self.last_grid_cols < 1:
            self.last_grid_cols = 1

        # Update Index Trackbar Range
        if self.initialized:
            num_windows = len(self.windows)
            new_max = max(1, num_windows - 1)
            try:
                cv2.setTrackbarMax("Index", self.window_name, new_max)
                if self.current_idx > new_max:
                    self.current_idx = 0
                cv2.setTrackbarPos("Index", self.window_name, self.current_idx)
            except cv2.error:
                pass

    def run_inference(self, crop_list):
        """Run batch inference."""
        if not crop_list:
            return None

        tensors = []
        for img in crop_list:
            # Resize
            img_resized = cv2.resize(img, (256, 256))

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            t_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            t_img = TF.normalize(
                t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            tensors.append(t_img)

        batch = torch.stack(tensors).to(DEVICE)

        with torch.no_grad():
            preds = self.model(batch)

        return preds.cpu().numpy()

    def generate_grid(self):
        """Generates the base grid image with inference results."""
        if not self.windows:
            self.grid_img = np.zeros((self.h, self.h, 3), dtype=np.uint8)
            self.grid_latents = []
            self.current_landmarks = None
            self.current_confidences = []
            return

        cell_w, cell_h = self.windows[0][2], self.windows[0][3]
        num_windows = len(self.windows)

        cols = self.last_grid_cols
        rows = int(np.ceil(num_windows / cols))

        grid_w = cols * cell_w
        grid_h = rows * cell_h

        canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        self.grid_latents = []

        # 1. Collect Crops
        crop_list = []
        meta_list = []

        for idx, (x, y, w, h) in enumerate(self.windows):
            r = idx // cols
            c = idx % cols

            x1, y1 = x, y
            x2, y2 = x + w, y + h
            ix1, iy1 = max(0, x1), max(0, y1)
            ix2, iy2 = min(self.w, x2), min(self.h, y2)

            if ix2 > ix1 and iy2 > iy1:
                roi = self.original_img[iy1:iy2, ix1:ix2]
                crop_canvas = np.zeros((h, w, 3), dtype=np.uint8) + 128
                cx1, cy1 = ix1 - x1, iy1 - y1
                cx2, cy2 = cx1 + (ix2 - ix1), cy1 + (iy2 - iy1)
                crop_canvas[cy1:cy2, cx1:cx2] = roi
                crop = crop_canvas
            else:
                crop = np.zeros((h, w, 3), dtype=np.uint8)

            crop_list.append(crop)
            meta_list.append((idx, r, c))

        # 2. Run Inference
        print(f"Running inference on {len(crop_list)} windows...")
        self.current_landmarks = self.run_inference(crop_list)
        self.current_confidences = []

        # 3. Draw on Canvas
        min_lv, max_lv = 0.0, 12.0

        for i, (idx, r, c) in enumerate(meta_list):
            crop = crop_list[i].copy()
            ldmks = self.current_landmarks[i]

            avg_log_var = np.mean(ldmks[:, 2])

            for p in range(ldmks.shape[0]):
                lx, ly, lv = ldmks[p]
                px = int(lx * cell_w)
                py = int(ly * cell_h)

                t = (lv - min_lv) / (max_lv - min_lv)
                t = max(0.0, min(1.0, t))
                color = (0, int(255 * (1 - t)), int(255 * t))

                cv2.circle(crop, (px, py), 4, color, -1)

            t_avg = (avg_log_var - min_lv) / (max_lv - min_lv)
            conf_score = 1.0 - max(0.0, min(1.0, t_avg))
            self.current_confidences.append(conf_score)

            conf_color = (0, 0, 0)
            if conf_score > 0.6:
                conf_color = (0, 255, 0)
            elif conf_score < 0.3:
                conf_color = (0, 0, 255)
            else:
                conf_color = (0, 255, 255)

            # Larger Font on Grid
            cv2.putText(
                crop,
                f"{conf_score:.2f}",
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                crop,
                f"{conf_score:.2f}",
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                conf_color,
                2,
            )

            start_y = r * cell_h
            start_x = c * cell_w
            canvas[start_y : start_y + cell_h, start_x : start_x + cell_w] = crop
            self.grid_latents.append((start_x, start_y, cell_w, cell_h))

        self.grid_img = canvas

    def update_display(self):
        """Composites the final image."""
        if not self.windows or self.grid_img is None:
            return

        min_lv, max_lv = 0.0, 12.0

        # 1. Left Image
        left_img = self.original_img.copy()
        if 0 <= self.current_idx < len(self.windows):
            wx, wy, ww, wh = self.windows[self.current_idx]
            cv2.rectangle(left_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 3)

            if self.current_confidences and self.current_idx < len(
                self.current_confidences
            ):
                c_score = self.current_confidences[self.current_idx]
                c_col = (
                    (0, 255, 0)
                    if c_score > 0.6
                    else (0, 0, 255)
                    if c_score < 0.3
                    else (0, 255, 255)
                )

                # Much Larger Font on Original Image
                txt_pos = (wx, max(40, wy - 10))
                cv2.putText(
                    left_img,
                    f"Conf: {c_score:.2f}",
                    txt_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    6,
                )
                cv2.putText(
                    left_img,
                    f"Conf: {c_score:.2f}",
                    txt_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    c_col,
                    3,
                )

            if (
                self.current_landmarks is not None
                and len(self.current_landmarks) > self.current_idx
            ):
                ldmks = self.current_landmarks[self.current_idx]
                for p in range(ldmks.shape[0]):
                    lx, ly, lv = ldmks[p]
                    gx = int(wx + lx * ww)
                    gy = int(wy + ly * wh)

                    t = (lv - min_lv) / (max_lv - min_lv)
                    t = max(0.0, min(1.0, t))
                    color = (0, int(255 * (1 - t)), int(255 * t))

                    if 0 <= gx < self.w and 0 <= gy < self.h:
                        cv2.circle(left_img, (gx, gy), 5, color, -1)

        # 2. Right Image
        right_img = self.grid_img.copy()
        if 0 <= self.current_idx < len(self.grid_latents):
            gx, gy, gw, gh = self.grid_latents[self.current_idx]

            img_h = right_img.shape[0]
            tickness = max(5, int(img_h * 0.005))

            cv2.rectangle(
                right_img, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), tickness
            )

        # 3. Resize Logic
        target_h = 800
        scale_l = target_h / left_img.shape[0]
        left_disp = cv2.resize(left_img, (0, 0), fx=scale_l, fy=scale_l)

        if right_img.shape[0] > 0:
            scale_r = target_h / right_img.shape[0]
            right_disp = cv2.resize(right_img, (0, 0), fx=scale_r, fy=scale_r)
        else:
            right_disp = np.zeros_like(left_disp)

        combined = np.hstack([left_disp, right_disp])

        if combined.shape[1] > self.max_disp_size * 2:
            sf = (self.max_disp_size * 1.5) / combined.shape[1]
            combined = cv2.resize(combined, (0, 0), fx=sf, fy=sf)

        info_text = (
            f"Win: {self.current_size}x{self.current_size} | Idx: {self.current_idx}"
        )
        cv2.putText(
            combined,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, combined)

    def process_update(self):
        """Applies pending logic changes and regenerates windows."""
        # Apply pending values
        self.size_idx = self.pending_size_idx
        self.current_size = self.available_sizes[self.size_idx]

        val = max(1, self.pending_stride_val)
        self.stride_ratio = val / 100.0

        self.generate_windows()
        self.generate_grid()
        self.update_display()

    def on_size_change(self, val):
        if not self.initialized:
            return
        self.pending_size_idx = val
        self.dirty = True
        self.last_interaction_time = time.time()

    def on_stride_change(self, val):
        if not self.initialized:
            return
        self.pending_stride_val = val
        self.dirty = True
        self.last_interaction_time = time.time()

    def on_index_change(self, val):
        if not self.initialized:
            return
        self.current_idx = val
        self.update_display()

    def run(self):
        print("Starting Interactive Visualizer (with Inference)...")
        print("Controls:")
        print(" - 'Size Level' Trackbar: 0 (Max) to N (Min)")
        print(" - 'Stride' Trackbar: Adjust overlap percentage")
        print(" - 'Index' Trackbar: Select specific window")
        print(" - Press 'q', 'ESC', or close window to exit")

        cv2.waitKey(500)

        while True:
            # Smart Update Logic
            if self.dirty:
                should_update = False

                if self.mouse_check_func:
                    is_down = self.check_mouse_down()
                    if not is_down:
                        should_update = True
                else:
                    # Timer Fallback
                    if time.time() - self.last_interaction_time > self.debounce_delay:
                        should_update = True

                if should_update:
                    print(
                        f"Updating grid (Native Check: {self.mouse_check_func is not None})..."
                    )
                    self.process_update()
                    self.dirty = False

            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord("q"):
                break
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

        cv2.destroyAllWindows()


def main(args):
    try:
        vis = WindowVisualizer(args.image, stride_ratio=args.stride)
        vis.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/jordan-pants-toe-touch.png",
        help="Path to test image",
    )
    parser.add_argument("--stride", type=float, default=0.10, help="Stride ratio")

    args = parser.parse_args()
    main(args)
