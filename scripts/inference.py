import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from lookma.dataset import SynthBodyDataset
from lookma.helpers.geometry import batch_rodrigues, rotation_6d_to_matrix
from lookma.helpers.visualize_data import draw_mesh
from lookma.models import HMRBodyNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_affine_transform(meta, target_size=256):
    # Replicate logic from SynthBodyDataset.__getitem__
    # to get the affine transform matrix M
    landmarks_2d = np.array(meta["landmarks"]["2D"], dtype=np.float32)

    min_x, max_x = np.min(landmarks_2d[:, 0]), np.max(landmarks_2d[:, 0])
    min_y, max_y = np.min(landmarks_2d[:, 1]), np.max(landmarks_2d[:, 1])
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    width = max_x - min_x
    height = max_y - min_y
    base_size = max(width, height) * 1.2

    # is_train=False logic (no random augs)
    rot = 0
    scale_aug = 1.0  # Default scale logic for eval
    shift_x = 0.0
    shift_y = 0.0

    # Apply shifts (none)
    center_x += base_size * shift_x
    center_y += base_size * shift_y

    proc_size = base_size / scale_aug

    dst_size = target_size
    dst_center = dst_size / 2.0
    scale_ratio = dst_size / proc_size

    M = cv2.getRotationMatrix2D((center_x, center_y), rot, scale_ratio)
    M[0, 2] += dst_center - center_x
    M[1, 2] += dst_center - center_y

    return M


def run_inference(image_name=None):
    # Load Model
    print("Loading model...")
    model = HMRBodyNetwork(backbone_name="hrnet_w48", pretrained=False).to(DEVICE)

    # Finds the latest checkpoint
    checkpoints = glob.glob("experiments/checkpoints/model_epoch_*.pth")
    if not checkpoints:
        print("No checkpoints found in experiments/checkpoints/")
        return

    # Sort by epoch number
    latest_ckpt = sorted(
        checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )[-1]
    print(f"Loading checkpoint: {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load Dataset & Sample
    print("Loading dataset sample...")
    dataset = SynthBodyDataset(
        "data/synth_body", target_size=256, is_train=False, specific_image=image_name
    )

    # Pick a random sample if an image name is not specified
    idx = random.randint(0, len(dataset)) if image_name is None else 0
    sample = dataset[idx]

    # --- NEW: Load Original Image & Camera for Visualization ---
    # We want to show the result on the FULL image, not the crop.
    json_path = dataset.json_paths[idx]
    with open(json_path, "r") as f:
        meta = json.load(f)

    # Reconstruct path to original image
    base_name = (
        os.path.basename(json_path)
        .replace("metadata_", "img_")
        .replace(".json", ".jpg")
    )
    img_path = os.path.join(dataset.root_dir, base_name)
    original_img_bgr = cv2.imread(img_path)

    # Original Intrinsics (Uncropped)
    original_K = np.array(meta["camera"]["camera_to_image"], dtype=np.float32)
    # -----------------------------------------------------------

    raw_img = sample["image"].to(DEVICE).float() / 255.0
    norm_img = TF.normalize(
        raw_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).unsqueeze(0)  # Add batch dim [1, 3, 256, 256]

    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        p_pose, p_shape, p_ldmk = model(norm_img)

    # Visualize
    print("Visualizing...")

    # Convert data for visualization (Adapting logic from train.py)
    # We need to construct the full pose matrix by combining GT pose (for hands/face not predicted)
    # with Predicted body pose, OR just visualize what we have.
    # Training script mixes GT and Pred because the model only predicts body params (21 joints).
    # We will do the same: Hybrid Pose.

    gt_pose = sample["pose"].unsqueeze(0).to(DEVICE)  # [1, 52, 3] Axis Angle
    gt_rotmat = batch_rodrigues(gt_pose.view(1, 52, 3))  # [1, 52, 3, 3]

    p_pose_single = p_pose.float()  # [1, 21, 6]
    p_rotmat = rotation_6d_to_matrix(p_pose_single.view(1, 21, 6))  # [1, 21, 3, 3]

    full_rotmat = gt_rotmat.clone()
    full_rotmat[0, 1:22] = p_rotmat[0]  # Overwrite body joints

    # Convert back to Axis-Angle for draw_mesh
    rots_np = full_rotmat[0].cpu().numpy()
    pose_aa_flat = []
    for i in range(52):
        aa, _ = cv2.Rodrigues(rots_np[i])
        pose_aa_flat.append(aa.squeeze())
    pose_aa_flat = np.concatenate(pose_aa_flat)

    # Params for draw_mesh
    cam_ext = sample["cam_extrinsics"].unsqueeze(0).numpy()

    # Use Original Intrinsics
    K_mat = original_K[np.newaxis, ...]  # [1, 3, 3]

    # In training, we passed GT trans. For inference on training set, we also use GT trans
    # since this model doesn't seem to predict translation (based on train.py usage).
    gt_world_t = sample["trans"].unsqueeze(0).numpy()

    vis_img = draw_mesh(
        original_img_bgr,  # Pass the detailed original image
        p_shape[0].cpu().numpy(),
        pose_aa_flat,
        gt_world_t[0],
        cam_ext[0],
        K_mat[0],  # Pass the original K
    )

    # --- Landmark Visualization ---
    print("Drawing landmarks...")

    # 1. Start with original image
    lm_img = original_img_bgr.copy()

    # 2. Get Affine Transform used for cropping (Inverse needed)
    M = get_affine_transform(meta, target_size=256)

    # 3. Process Landmarks
    # p_ldmk is [1, 1378, 3] (x, y, log_var)
    # x,y are normalized [0, 1] in crop space.
    pred_ldmk_cpu = p_ldmk[0].cpu().numpy()

    pts_crop = pred_ldmk_cpu[:, :2] * 256.0  # Scale to crop pixels [N, 2]
    log_var = pred_ldmk_cpu[:, 2]

    print(
        f"Log Var Stats: Min={log_var.min():.3f}, Max={log_var.max():.3f}, Mean={log_var.mean():.3f}"
    )

    # Inverse Transform to Original Image Space
    # cv2.transform expects points as [N, 1, 2]
    # Invert M
    M_inv = cv2.invertAffineTransform(M)
    pts_orig = cv2.transform(pts_crop.reshape(-1, 1, 2), M_inv).squeeze()  # [N, 2]

    # 4. Draw Dots
    # Confidence Heuristic: High log_var = Low Cert. Low log_var = High Cert.
    # log_var generally ranges from -4 (confident) to +4 (uncertain).
    # Convert to 0-1 score where 1 is good (Green)
    # score = exp(-0.5 * exp(log_var)) ? No, simply map log_var.
    # Let's trust that smaller variance is better.
    # Map [-5, 0] to Color.

    # Fixed Range Normalization based on losses.py analysis
    # log_var is ln(sigma^2).
    # 0.0 -> sigma=1px (Good/Green)
    # 5.0 -> sigma=12px (Bad/Red)
    min_val = 0.0
    max_val = 5.0

    for i in range(pts_orig.shape[0]):
        x, y = int(pts_orig[i, 0]), int(pts_orig[i, 1])
        conf = log_var[i]

        # Normalize
        factor = (conf - min_val) / (max_val - min_val)
        factor = np.clip(factor, 0.0, 1.0)

        # BGR
        # Good (0) -> Green (0,255,0)
        # Bad (1) -> Red (0,0,255)
        # B: 0
        # G: 255 * (1 - factor)
        # R: 255 * factor
        color = (0, int(255 * (1 - factor)), int(255 * factor))

        cv2.circle(lm_img, (x, y), 2, color, -1)

    # 5. Concatenate Side-by-Side
    # Ensure same height (they match because both are from original_img_bgr)
    # Add a small black separator line?
    sep = np.zeros((vis_img.shape[0], 10, 3), dtype=np.uint8)
    combined = np.hstack([vis_img, sep, lm_img])

    print("Displaying result (Press any key to close)...")
    cv2.imshow("Inference Result (Left: Mesh, Right: Landmarks)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference(image_name="img_0000000_001.jpg")
