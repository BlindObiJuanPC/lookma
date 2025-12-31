import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from lookma.dataset import SynthBodyDataset, SynthHandDataset
from lookma.helpers.augmentation import TrainingAugmentation
from lookma.helpers.visualize_data import (
    LDMK_CONN,
    draw_dense_landmarks,
    draw_mesh,
    draw_skeleton,
    get_smplh_vertices,
)


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
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the first visualization and exit",
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

    # Initialize GPU Augmentation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    augmentor = TrainingAugmentation().to(device)
    print(f"Using device: {device}")

    # Indices to shuffle through
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    for idx in indices:
        try:
            item = dataset[idx]
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            continue

        # item["image"] is Tensor (3, H, W) [0-255].
        # We need [0, 1] for augmentation.
        raw_img = (
            (item["image"].float() / 255.0).to(device).unsqueeze(0)
        )  # [1, 3, H, W]

        # Apply Augmentation
        aug_img_tensor, debug_info = augmentor(raw_img, return_debug_info=True)

        # Convert back to uint8 BGR for display
        # Raw Image (from Dataset, before GPU aug)
        raw_img_np = (raw_img.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        raw_img_bgr = cv2.cvtColor(raw_img_np, cv2.COLOR_RGB2BGR)

        # Augmented Image
        aug_img_np = (
            aug_img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        aug_img_bgr = cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR)

        # Original Clean (Before Dataset Crop/Aug)
        orig_clean = item["original_image"]
        orig_clean_bgr = cv2.cvtColor(orig_clean, cv2.COLOR_RGB2BGR)

        # Alignment Visualization: Mesh Mask
        align_img = raw_img_bgr.copy()
        try:
            align_vis = draw_mesh(
                align_img,
                item["betas"].numpy(),
                item["pose"].numpy(),
                item["trans"].numpy(),
                item["cam_extrinsics"].numpy(),
                item["cam_intrinsics"].numpy(),
            )
            # Also draw skeleton on top
            # landmarks_2d is tensor -> numpy
            ldmks = item["landmarks_2d"].numpy()

            # Select proper connectivity
            conn_key = "hand" if args.dataset == "hand" else "body"
            draw_skeleton(align_vis, ldmks, LDMK_CONN[conn_key], thickness=1)

            # Draw dense landmarks for hand
            if args.dataset == "hand":
                vertices = get_smplh_vertices(
                    item["betas"].numpy(),
                    item["pose"].numpy(),
                    item["trans"].numpy(),
                )
                draw_dense_landmarks(
                    align_vis,
                    vertices,
                    SynthHandDataset.DENSE_LANDMARK_IDS,
                    item["cam_extrinsics"].numpy(),
                    item["cam_intrinsics"].numpy(),
                    color=(0, 255, 255),  # Yellow
                )

        except Exception as e:
            print(f"Mesh draw failed: {e}")
            align_vis = np.zeros_like(align_img)

        # Visualization Layout
        h_disp = 512

        # Scale all to h_disp
        def resize_h(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (int(w * scale), target_h))

        disp_clean = resize_h(orig_clean_bgr, h_disp)
        disp_raw = resize_h(raw_img_bgr, h_disp)
        disp_align = resize_h(align_vis, h_disp)
        disp_aug = resize_h(aug_img_bgr, h_disp)

        # Concatenate: [Original Full] | [Dataset Crop] | [GPU Aug] | [Align Check]
        combined = np.hstack([disp_clean, disp_raw, disp_aug, disp_align])

        # Draw Text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        line_spacing = 20
        color = (0, 0, 255)  # Red

        # Determine x start positions
        x_clean = 10
        x_raw = disp_clean.shape[1] + 10
        x_aug = disp_clean.shape[1] + disp_raw.shape[1] + 10
        x_align = disp_clean.shape[1] + disp_raw.shape[1] + disp_aug.shape[1] + 10

        y_start = 30

        # Labels
        cv2.putText(
            combined, "Raw Source", (x_clean, h_disp - 10), font, 0.6, (0, 255, 0), 1
        )
        cv2.putText(
            combined, "Dataset Crop", (x_raw, h_disp - 10), font, 0.6, (0, 255, 0), 1
        )
        cv2.putText(
            combined,
            "Mesh Alignment",
            (x_align, h_disp - 10),
            font,
            0.6,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            combined,
            "GPU Augmentation",
            (x_aug, h_disp - 10),
            font,
            0.6,
            (0, 255, 0),
            1,
        )

        # Print Aug params
        # Dataset Aug Params
        ds_aug = item["aug_params"]
        ds_lines = [
            f"Rot: {ds_aug['rot']:.1f}",
            f"Scale: {ds_aug['scale']:.2f}",
            f"Shift X: {ds_aug['shift_x']:.2f}",
            f"Shift Y: {ds_aug['shift_y']:.2f}",
        ]

        for i, line in enumerate(ds_lines):
            cv2.putText(
                combined,
                line,
                (x_raw, y_start + i * line_spacing),
                font,
                font_scale,
                color,
                1,
            )

        # GPU Aug Params
        gpu_lines = []

        # Color Jitter
        cj = debug_info["color_jitter"]
        if cj["brightness"]:
            gpu_lines.append(f"Bright: {cj['brightness']:.2f}")
        if cj["contrast"]:
            gpu_lines.append(f"Contr: {cj['contrast']:.2f}")
        if cj["saturation"]:
            gpu_lines.append(f"Sat: {cj['saturation']:.2f}")
        if cj["hue"]:
            gpu_lines.append(f"Hue: {cj['hue']:.2f}")

        # ISO Noise
        iso = debug_info["iso_noise"]
        if iso["applied"]:
            gpu_lines.append("ISO Noise: ON")
            gpu_lines.append(f" Sigma: {iso['sigma_read']:.4f}")
            gpu_lines.append(f" Gain: {iso['gain']:.4f}")
        else:
            gpu_lines.append("ISO Noise: OFF")

        # Pixelate
        pix = debug_info["pixelate"]
        if pix["applied"]:
            gpu_lines.append(f"Pixelate: ON (1/{pix['downsample_factor']})")
        else:
            gpu_lines.append("Pixelate: OFF")

        # Motion Blur
        blur = debug_info["motion_blur"]
        if blur["applied"]:
            gpu_lines.append(f"Blur: ON (k={blur['kernel_size']})")
            gpu_lines.append(f" Ang: {blur['angle']:.1f}")
        else:
            gpu_lines.append("Blur: OFF")

        for i, line in enumerate(gpu_lines):
            cv2.putText(
                combined,
                line,
                (x_aug, y_start + i * line_spacing),
                font,
                font_scale,
                color,
                1,
            )

        if args.save_path:
            cv2.imwrite(args.save_path, combined)
            print(f"Saved visualization to {args.save_path}")
            sys.exit(0)

        cv2.imshow("Augmentation Visualization", combined)

        # Loop with polling to handle Window Close ('X')
        user_quit = False
        next_img = False
        while True:
            # Check if window was closed
            if (
                cv2.getWindowProperty(
                    "Augmentation Visualization", cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                user_quit = True
                break

            key = cv2.waitKey(100)
            if key == -1:
                continue

            if key == ord("q") or key == 27:  # q or Esc
                user_quit = True
                break
            elif key == ord("n") or key == 32:  # n or Space
                next_img = True
                break

        if user_quit:
            break
        if next_img:
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
