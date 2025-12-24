import torch
import cv2
import numpy as np
import os
import smplx
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from lookma.dataset import SynthBodyDataset
from lookma.models import HMRBodyNetwork
from lookma.geometry import rotation_6d_to_matrix
from lookma.visualizer import MeshRenderer

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/synth_body"
MODEL_PATH = "data/smplx"

# 1. UPDATE THIS to point to your 5090 checkpoint!
CHECKPOINT_PATH = "experiments/checkpoints/model_epoch_60.pth"
OUTPUT_DIR = "experiments/inspection_results"
NUM_SAMPLES = 10


def main():
    print(f"--- INSPECTING 5090 CHECKPOINT: {os.path.basename(CHECKPOINT_PATH)} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Setup Data (Use target_size=256 and is_train=False for clean validation)
    dataset = SynthBodyDataset(DATA_PATH, target_size=256, is_train=False)

    # Pick random images to test generalization
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)

    # 3. Setup Models
    print("Loading HMR Network...")
    model = HMRBodyNetwork(backbone_name="hrnet_w48").to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Loading SMPL Layer...")
    viz_smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", use_pca=False, num_betas=10
    ).to(DEVICE)
    # Ensure internal pose mean is disabled
    viz_smpl.pose_mean = torch.tensor([0.0], device=DEVICE)

    print(f"Running inference on {NUM_SAMPLES} samples...")

    for i, batch in enumerate(loader):
        # A. Prepare Image
        raw_image = batch["image"].to(DEVICE) / 255.0
        norm_image = TF.normalize(
            raw_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # B. Get Camera Matrices from Dataset
        K = batch["cam_intrinsics"].to(DEVICE)
        cam_ext = batch["cam_extrinsics"].to(DEVICE)
        R_ext = cam_ext[0, :3, :3]

        # C. Inference
        with torch.no_grad():
            # Network predicts 4 outputs now
            pred_pose, pred_shape, pred_cam, pred_ldmk = model(norm_image)

            # Convert 6D -> Matrices
            p_rotmat = rotation_6d_to_matrix(pred_pose.view(1, 52, 6))

            # D. Generate 3D Mesh in Local Space
            smpl_out = viz_smpl(
                betas=pred_shape[:, :10],
                global_orient=p_rotmat[:, 0:1],
                body_pose=p_rotmat[:, 1:22],
                left_hand_pose=p_rotmat[:, 22:37],
                right_hand_pose=p_rotmat[:, 37:52],
                pose2rot=False,
            )

            # E. SYNCED COORDINATE MATH (The Truth)
            # 1. Take Local Vertices [N, 3]
            verts_local = smpl_out.vertices[0]
            # 2. Rotate to Camera Frame: R_ext * Verts
            verts_rotated = torch.matmul(R_ext, verts_local.transpose(0, 1)).transpose(
                0, 1
            )
            # 3. Add Predicted Translation (which is in Camera Space)
            verts_cam = verts_rotated + pred_cam[0]

        # F. RENDER
        # Convert raw tensor back to uint8 BGR for visualizer
        img_np = (raw_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        renderer = MeshRenderer(width=256, height=256)
        # MeshRenderer expects RGB input for blending
        res_rgb = renderer.render_mesh(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            verts_cam.cpu().numpy(),
            viz_smpl.faces,
            K=K[0].cpu().numpy(),
        )
        renderer.delete()

        # Save Result
        out_path = os.path.join(OUTPUT_DIR, f"inspect_{i}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    main()
