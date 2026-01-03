import argparse
import os
import cv2
import numpy as np


class WindowVisualizer:
    def __init__(self, img_path, stride_ratio=0.5):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        self.original_img = cv2.imread(img_path)
        if self.original_img is None:
            raise ValueError("Could not load image.")

        self.h, self.w = self.original_img.shape[:2]
        self.stride_ratio = stride_ratio

        # Display Config
        self.window_name = "Interactive Sliding Window"
        self.max_disp_size = 1200
        self.min_win_size = 32

        # Pre-calculate Window Sizes (Logarithmic: Max * 0.75^k)
        self.available_sizes = []
        curr = float(min(self.h, self.w))
        while curr >= self.min_win_size:
            self.available_sizes.append(int(curr))
            curr *= 0.75

        # Ensure we have at least one size
        if not self.available_sizes:
            self.available_sizes = [min(self.h, self.w)]

        # State
        self.size_idx = 0  # Index into available_sizes
        self.current_size = self.available_sizes[0]
        self.current_idx = 0
        self.windows = []
        self.initialized = False

        # Grid Cache
        self.grid_img = None
        self.grid_latents = []
        self.last_grid_cols = 1

        # Init Windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Create Trackbars
        # Size Slider: 0 to len(sizes)-1
        cv2.createTrackbar(
            "Size Level",
            self.window_name,
            0,
            len(self.available_sizes) - 1,
            self.on_size_change,
        )
        cv2.createTrackbar(
            "Stride %",
            self.window_name,
            int(self.stride_ratio * 100),
            100,
            self.on_stride_change,
        )
        cv2.createTrackbar("Index", self.window_name, 0, 100, self.on_index_change)

        # Mark as initialized and trigger generation
        self.initialized = True
        self.on_size_change(0)

    def generate_windows(self):
        """Generates single-scale windows based on current_size."""
        self.windows = []
        size = self.current_size

        stride = int(size * self.stride_ratio)
        if stride < 1:
            stride = 1

        # Expand range to allow windows to go partially out of bounds
        start_offset = -size // 2

        x_limit = self.w - (size // 2) + stride
        x_range = list(range(start_offset, x_limit, stride))
        self.last_grid_cols = len(x_range)
        if self.last_grid_cols < 1:
            self.last_grid_cols = 1

        y_limit = self.h - (size // 2) + stride
        for y in range(start_offset, y_limit, stride):
            for x in x_range:
                self.windows.append((x, y, size, size))

        # Update Index Trackbar Range
        if self.initialized:
            num_windows = len(self.windows)
            new_max = max(1, num_windows - 1)
            try:
                cv2.setTrackbarMax("Index", self.window_name, new_max)
                cv2.setTrackbarPos("Index", self.window_name, 0)
            except cv2.error:
                pass

        self.current_idx = 0

    def generate_grid(self):
        """Generates the base grid image (without selection highlights)."""
        if not self.windows:
            self.grid_img = np.zeros((self.h, self.h, 3), dtype=np.uint8)
            self.grid_latents = []
            return

        cell_w, cell_h = self.windows[0][2], self.windows[0][3]
        num_windows = len(self.windows)

        cols = self.last_grid_cols
        rows = int(np.ceil(num_windows / cols))

        grid_w = cols * cell_w
        grid_h = rows * cell_h

        canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        self.grid_latents = []

        for idx, (x, y, w, h) in enumerate(self.windows):
            r = idx // cols
            c = idx % cols

            # Safe Crop with Padding
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            ix1 = max(0, x1)
            iy1 = max(0, y1)
            ix2 = min(self.w, x2)
            iy2 = min(self.h, y2)

            if ix2 > ix1 and iy2 > iy1:
                roi = self.original_img[iy1:iy2, ix1:ix2]

                crop_canvas = np.zeros((h, w, 3), dtype=np.uint8) + 128

                cx1 = ix1 - x1
                cy1 = iy1 - y1
                cx2 = cx1 + (ix2 - ix1)
                cy2 = cy1 + (iy2 - iy1)

                crop_canvas[cy1:cy2, cx1:cx2] = roi
                crop = crop_canvas
            else:
                crop = np.zeros((h, w, 3), dtype=np.uint8)

            start_y = r * cell_h
            start_x = c * cell_w
            canvas[start_y : start_y + cell_h, start_x : start_x + cell_w] = crop

            self.grid_latents.append((start_x, start_y, cell_w, cell_h))

        self.grid_img = canvas

    def update_display(self):
        """Composites the final image."""
        if not self.windows or self.grid_img is None:
            return

        # 1. Left Image: Original + Highlight
        left_img = self.original_img.copy()
        if 0 <= self.current_idx < len(self.windows):
            wx, wy, ww, wh = self.windows[self.current_idx]
            cv2.rectangle(
                left_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 3
            )  # Green Thick

        # 2. Right Image: Grid + Highlight
        right_img = self.grid_img.copy()
        if 0 <= self.current_idx < len(self.grid_latents):
            gx, gy, gw, gh = self.grid_latents[self.current_idx]
            cv2.rectangle(
                right_img, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 5
            )  # Red Thick

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

        # Draw Info Text
        info_text = f"Window Size: {self.current_size}x{self.current_size}"
        # Draw on top left (green text with black border for readability)
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

    def on_size_change(self, val):
        if not self.initialized:
            return
        self.size_idx = val
        self.current_size = self.available_sizes[val]
        self.generate_windows()
        self.generate_grid()
        self.update_display()

    def on_stride_change(self, val):
        if not self.initialized:
            return
        val = max(1, val)
        self.stride_ratio = val / 100.0
        self.generate_windows()
        self.generate_grid()
        self.update_display()

    def on_index_change(self, val):
        if not self.initialized:
            return
        self.current_idx = val
        self.update_display()

    def run(self):
        print("Starting Interactive Visualizer...")
        print("Controls:")
        print(" - 'Size Level' Trackbar: 0 (Max) to N (Min)")
        print(" - 'Stride' Trackbar: Adjust overlap percentage")
        print(" - 'Index' Trackbar: Select specific window")
        print(" - Press 'q', 'ESC', or close window to exit")

        cv2.waitKey(500)

        while True:
            key = cv2.waitKey(100) & 0xFF
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
    parser.add_argument("--stride", type=float, default=0.5, help="Stride ratio")

    args = parser.parse_args()
    main(args)
