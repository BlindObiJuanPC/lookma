import argparse
import os
import sys

import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())

from lookma.dataset import SynthBodyDataset, SynthHandDataset


def main():
    parser = argparse.ArgumentParser(description="Visualize Dataset Augmentations")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["body", "hand"],
        default="body",
        help="Which dataset to visualize",
    )
    parser.add_argument(
        "--root_body",
        type=str,
        default="data/synth_body",
        help="Root dir for body data",
    )
    parser.add_argument(
        "--root_hand",
        type=str,
        default="data/synth_hand",
        help="Root dir for hand data",
    )
    args = parser.parse_args()

    if args.dataset == "body":
        print(f"Initializing SynthBodyDataset from {args.root_body}...")
        dataset = SynthBodyDataset(
            root_dir=args.root_body, is_train=True, return_debug_info=True
        )
    else:
        print(f"Initializing SynthHandDataset from {args.root_hand}...")
        dataset = SynthHandDataset(
            root_dir=args.root_hand, is_train=True, return_debug_info=True
        )

    print(f"Dataset length: {len(dataset)}")
    print("Press 'n' for next image, 'q' to quit.")

    # Indices to shuffle through
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    for idx in indices:
        try:
            item = dataset[idx]
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            continue

        # Processed Image (Tensor -> Numpy -> uint8)
        # Tensor is (3, H, W) float
        proc_img_tensor = item["image"]
        proc_img_np = proc_img_tensor.permute(1, 2, 0).numpy()
        proc_img_np = np.clip(proc_img_np, 0, 255).astype(np.uint8)
        proc_img_np = cv2.cvtColor(proc_img_np, cv2.COLOR_RGB2BGR)

        # Original Image is RGB (from dataset), convert to BGR for OpenCV
        orig_img = item["original_image"]
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        # Aug Params
        aug_params = item["aug_params"]

        # Visualization Layout
        h_orig, w_orig = orig_img.shape[:2]
        h_proc, w_proc = proc_img_np.shape[:2]

        # Target height for display (e.g. 512)
        disp_h = 512
        scale_orig = disp_h / h_orig
        scale_proc = disp_h / h_proc

        orig_disp = cv2.resize(orig_img, (int(w_orig * scale_orig), disp_h))
        proc_disp = cv2.resize(proc_img_np, (int(w_proc * scale_proc), disp_h))

        # Concatenate
        combined = np.hstack([orig_disp, proc_disp])

        # Text Settings
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        line_spacing = 20
        color = (0, 0, 255)  # Red
        thickness = 1

        # Start drawing text on the augmented image side
        # x_offset is width of original image
        x_start = orig_disp.shape[1] + 10
        y_start = 20

        # Draw Augmentation Params (multiline)
        lines = [
            f"Rot: {aug_params['rot']:.1f}",
            f"Scale: {aug_params['scale']:.2f}",
            f"Shift X: {aug_params['shift_x']:.2f}",
            f"Shift Y: {aug_params['shift_y']:.2f}",
        ]

        for i, line in enumerate(lines):
            cv2.putText(
                combined,
                line,
                (x_start, y_start + i * line_spacing),
                font,
                font_scale,
                color,
                thickness,
            )

        # Labels
        cv2.putText(combined, "Original", (10, disp_h - 10), font, 0.6, (0, 255, 0), 1)
        cv2.putText(
            combined, "Augmented", (x_start, disp_h - 10), font, 0.6, (0, 255, 0), 1
        )

        # Instructions (Small text)
        cv2.putText(
            combined, "n: next, q: quit", (10, 20), font, 0.4, (255, 255, 255), 1
        )

        cv2.imshow("Augmentation Visualization", combined)

        # Poll for key press or window close
        user_quit = False
        next_img = False
        while True:
            if (
                cv2.getWindowProperty(
                    "Augmentation Visualization", cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                user_quit = True
                break

            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                user_quit = True
                break
            elif key == ord("n"):
                next_img = True
                break

        if user_quit:
            break
        if next_img:
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
