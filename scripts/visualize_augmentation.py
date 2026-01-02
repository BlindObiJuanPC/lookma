import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from lookma.dataset import SynthBodyDataset, SynthHandDataset, SynthBodyRoiDataset
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
        choices=["body", "hand", "roi"],
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
    elif args.dataset == "hand":
        print(f"Initializing SynthHandDataset from {args.root_hand}...")
        dataset = SynthHandDataset(
            root_dir=args.root_hand, is_train=True, return_debug_info=True
        )
    elif args.dataset == "roi":
        print(f"Initializing SynthBodyRoiDataset from {args.root_body}...")
        dataset = SynthBodyRoiDataset(
            root_dir=args.root_body, is_train=True, return_debug_info=True
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
        orig_clean = item.get("original_image")
        if orig_clean is not None:
            orig_clean_bgr = cv2.cvtColor(orig_clean, cv2.COLOR_RGB2BGR)
        else:
            orig_clean_bgr = np.zeros_like(raw_img_bgr)  # Fallback

        # Alignment Visualization: Mesh Mask
        align_img = aug_img_bgr.copy()
        try:
            # Check if we have necessary 3D params (ROI dataset might simulate them or not)
            if "cam_intrinsics" in item and "cam_extrinsics" in item:
                # Note: Draw mesh requires 3D params. ROI dataset updates intrinsics/extrinsics.
                align_vis = draw_mesh(
                    align_img,
                    item["betas"].numpy(),
                    item["pose"].numpy(),
                    item.get(
                        "trans", torch.zeros(3)
                    ).numpy(),  # ROI might skip trans return or it is zero centered
                    item["cam_extrinsics"].numpy(),
                    item["cam_intrinsics"].numpy(),
                )
            else:
                align_vis = align_img

            # Draw skeleton on top
            ldmks = item["landmarks_2d"].numpy()

            # Select proper connectivity
            conn_key = "hand" if args.dataset == "hand" else "body"
            # If shape doesn't match skeleton, skip or warn
            if ldmks.shape[0] > 100:  # Dense landmarks probably
                pass  # Use draw_dense
            elif ldmks.shape[0] == 21:  # Body Joints? No 2D ldmks usually 2D.
                # Hand is 21 joints, Body is...
                # ROI output is 36 points. Skeleton conn might not match.
                if args.dataset != "roi":
                    draw_skeleton(align_vis, ldmks, LDMK_CONN[conn_key], thickness=1)

            # Draw dense landmarks (if defined in dataset)
            if hasattr(dataset, "DENSE_LANDMARK_IDS"):
                # For ROI dataset, landmarks_2d ARE the dense landmarks
                if args.dataset == "roi":
                    # Draw these points
                    for i in range(ldmks.shape[0]):
                        # ldmks are normalized 0-1
                        x = int(ldmks[i, 0] * align_vis.shape[1])
                        y = int(ldmks[i, 1] * align_vis.shape[0])
                        cv2.circle(align_vis, (x, y), 2, (0, 255, 255), -1)
                else:
                    # For Body/Hand, we project specific vertices
                    vertices = get_smplh_vertices(
                        item["betas"].numpy(),
                        item["pose"].numpy(),
                        item["trans"].numpy(),
                    )
                    draw_dense_landmarks(
                        align_vis,
                        vertices,
                        dataset.DENSE_LANDMARK_IDS,
                        item["cam_extrinsics"].numpy(),
                        item["cam_intrinsics"].numpy(),
                        color=(0, 255, 255),  # Yellow
                    )

        except Exception as e:
            print(f"Mesh draw failed: {e}")
            align_vis = np.zeros_like(align_img)

        # Visualization Layout: 2x2 Grid
        h_disp = 512

        def resize_h(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (int(w * scale), target_h))

        disp_clean = resize_h(orig_clean_bgr, h_disp)
        disp_raw = resize_h(raw_img_bgr, h_disp)
        disp_aug = resize_h(aug_img_bgr, h_disp)
        disp_align = resize_h(align_vis, h_disp)

        # Grid:
        # [Clean] [Raw/Crop]
        # [Aug]   [Align]

        # Ensure widths match for vertical stacking
        # Top Row
        w_top = disp_clean.shape[1] + disp_raw.shape[1]
        top_row = np.hstack([disp_clean, disp_raw])

        # Bottom Row
        w_bot = disp_aug.shape[1] + disp_align.shape[1]
        bot_row = np.hstack([disp_aug, disp_align])

        # Pad if widths mismatch
        max_w = max(w_top, w_bot)
        if w_top < max_w:
            pad = np.zeros((h_disp, max_w - w_top, 3), dtype=np.uint8)
            top_row = np.hstack([top_row, pad])
        if w_bot < max_w:
            pad = np.zeros((h_disp, max_w - w_bot, 3), dtype=np.uint8)
            bot_row = np.hstack([bot_row, pad])

        combined = np.vstack([top_row, bot_row])

        # Draw Text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thick = 1

        # Top Left
        cv2.putText(combined, "Original Full", (10, 30), font, font_scale, color, thick)

        # Top Right
        x_tr = disp_clean.shape[1] + 10
        cv2.putText(
            combined, "Dataset Crop", (x_tr, 30), font, font_scale, color, thick
        )

        # Bottom Left
        y_bot = h_disp + 30
        cv2.putText(
            combined, "GPU Augmented", (10, y_bot), font, font_scale, color, thick
        )

        # Bottom Right
        x_br = disp_aug.shape[1] + 10
        cv2.putText(
            combined,
            "Ground Truth / Landmarks",
            (x_br, y_bot),
            font,
            font_scale,
            color,
            thick,
        )

        # Print Aug params overlays
        # Dataset Params (Top Right)
        ds_aug = item.get("aug_params", {})
        y_txt = 60
        for k, v in ds_aug.items():
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                txt = f"{k}: {v:.2f}"
            else:
                txt = f"{k}: {v}"
            cv2.putText(combined, txt, (x_tr, y_txt), font, 0.5, (0, 0, 255), 1)
            y_txt += 20

        # GPU Params (Bottom Left)
        # Reuse existing logic to format lines
        gpu_lines = []
        # ... (rest of gpu lines logic from original file, re-implemented here concisely) ...
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

        # ISO
        iso = debug_info["iso_noise"]
        if iso["applied"]:
            gpu_lines.append(f"ISO: {iso['sigma_read']:.3f}/{iso['gain']:.3f}")

        # Pixelate
        pix = debug_info["pixelate"]
        if pix["applied"]:
            gpu_lines.append(f"Pix: 1/{pix['downsample_factor']}")

        # Blur
        blur = debug_info["motion_blur"]
        if blur["applied"]:
            gpu_lines.append(f"Blur: {blur['angle']:.1f} (k={blur['kernel_size']})")

        y_txt = h_disp + 60
        for line in gpu_lines:
            cv2.putText(combined, line, (10, y_txt), font, 0.5, (0, 0, 255), 1)
            y_txt += 20

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
