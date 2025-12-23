import cv2
import numpy as np
import smplx
import torch

from lookma.dataset import SynthBodyDataset
from lookma.geometry import batch_rodrigues
from lookma.visualizer import MeshRenderer

# --- CONFIG ---
DATA_PATH = "data/synth_body"
MODEL_PATH = "data/smplx"
TARGET_IMAGE = "img_0000020_003.jpg"
DEVICE = "cuda"


def main():
    print("--- TRUTH SERUM TEST ---")

    # 1. Load Data
    dataset = SynthBodyDataset(
        DATA_PATH, specific_image=TARGET_IMAGE, target_size=256, is_train=False
    )
    data = dataset[0]  # Get sample

    # 2. Setup SMPL
    smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", num_betas=10, use_pca=False
    ).to(DEVICE)
    mean_pose = smpl.pose_mean.clone().detach()  # [156]
    smpl.pose_mean = torch.tensor([0.0], device=DEVICE)  # Disable internal

    # 3. Prepare Inputs
    pose_raw = data["pose"].to(DEVICE)  # [156]
    betas = data["betas"].unsqueeze(0).to(DEVICE)
    trans = data["trans"].unsqueeze(0).to(DEVICE)

    # Load Image for background
    img = data["image"].permute(1, 2, 0).numpy().astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # --- TEST A: RAW POSE (What we use now) ---
    print("Generating Test A (Raw)...")
    rot_A = batch_rodrigues(pose_raw.view(1, 52, 3))
    run_render(smpl, rot_A, betas, trans, data, img_rgb, "verify_GT_A_Raw.jpg")

    # --- TEST B: POSE + MEAN (What we suspected) ---
    print("Generating Test B (Raw + Mean)...")
    pose_B = pose_raw + mean_pose
    rot_B = batch_rodrigues(pose_B.view(1, 52, 3))
    run_render(smpl, rot_B, betas, trans, data, img_rgb, "verify_GT_B_Mean.jpg")


def run_render(smpl, rotmats, betas, trans, data, img_rgb, filename):
    # Run SMPL
    output = smpl(
        betas=betas[:, :10],
        global_orient=rotmats[:, 0:1],
        body_pose=rotmats[:, 1:22],
        left_hand_pose=rotmats[:, 22:37],
        right_hand_pose=rotmats[:, 37:52],
        pose2rot=False,
    )

    # World -> Camera
    verts_world = output.vertices + trans.unsqueeze(1)
    ext = data["cam_extrinsics"].unsqueeze(0).to(DEVICE)

    ones = torch.ones(1, verts_world.shape[1], 1, device=DEVICE)
    verts_homo = torch.cat([verts_world, ones], dim=-1)
    verts_cam = torch.matmul(ext, verts_homo.transpose(1, 2)).transpose(1, 2)[..., :3]

    # Render
    K = data["cam_intrinsics"].numpy()
    renderer = MeshRenderer(width=256, height=256)
    res = renderer.render_mesh(
        img_rgb, verts_cam[0].cpu().detach().numpy(), smpl.faces, K=K
    )

    # Save
    cv2.imwrite(filename, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    print(f"Saved {filename}")
    renderer.delete()


if __name__ == "__main__":
    main()
