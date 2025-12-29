from PIL.ImagePalette import random
import glob
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from lookma.dataset import SynthBodyDataset
from lookma.helpers.geometry import batch_rodrigues, rotation_6d_to_matrix
from lookma.helpers.visualize_data import draw_mesh
from lookma.models import HMRBodyNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    print("Displaying result (Press any key to close)...")
    cv2.imshow("Inference Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference(image_name="img_0000000_001.jpg")
