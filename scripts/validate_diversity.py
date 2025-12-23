import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
import smplx
import cv2
import numpy as np
import os
from lookma.dataset import SynthBodyDataset
from lookma.models import HMRBodyNetwork
from lookma.losses import HMRLoss
from lookma.geometry import rotation_6d_to_matrix
from lookma.visualizer import MeshRenderer

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/synth_body"
MODEL_PATH = "data/smplx"
TARGET_ID = "0000020"  # Same person, two frames


def main():
    print(f"🚀 Running Diversity Test on ID {TARGET_ID}...")

    # 1. Load TWO frames for the same person
    dataset = SynthBodyDataset(DATA_PATH, is_train=False)
    indices = [i for i, path in enumerate(dataset.json_paths) if TARGET_ID in path][:2]
    print(f"🎯 Training on indices: {indices}")

    subset = Subset(dataset, indices)
    # Batch size 2: Both images processed at once
    loader = DataLoader(subset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # 2. Prepare Data
    raw_images = batch["image"].to(DEVICE) / 255.0
    norm_images = TF.normalize(
        raw_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    gt_pose = batch["pose"].to(DEVICE)
    gt_shape = batch["betas"].to(DEVICE)
    gt_ldmks = batch["landmarks_2d"].to(DEVICE)
    K = batch["cam_intrinsics"].to(DEVICE)
    ext = batch["cam_extrinsics"].to(DEVICE)

    # Camera Space Translation logic
    gt_world_t = batch["trans"].to(DEVICE)
    ones = torch.ones(2, 1, 1, device=DEVICE)
    gt_cam_t = torch.matmul(ext, torch.cat([gt_world_t.unsqueeze(-1), ones], dim=1))[
        :, :3, 0
    ]

    # 3. Setup Model & Loss
    model = HMRBodyNetwork(backbone_name="hrnet_w48").to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = HMRLoss(MODEL_PATH, device=DEVICE)

    # SMPL for visualization
    viz_smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", num_betas=10, use_pca=False
    ).to(DEVICE)
    viz_smpl.pose_mean = torch.tensor([0.0], device=DEVICE)

    os.makedirs("experiments/diversity", exist_ok=True)

    for epoch in range(1, 501):
        optimizer.zero_grad()
        p_pose, p_shape, p_cam, p_ldmk = model(norm_images)

        # Loss on both images simultaneously
        loss, comps, _ = criterion(
            p_pose,
            p_shape,
            p_cam,
            p_ldmk,
            gt_pose,
            gt_shape,
            gt_ldmks,
            gt_cam_t,
            K,
            ext,
        )

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

            with torch.no_grad():
                # Render BOTH frames to see if they untwist together
                for f in range(2):
                    pred_rotmat = rotation_6d_to_matrix(
                        p_pose[f : f + 1].view(1, 52, 6)
                    )
                    smpl_out = viz_smpl(
                        betas=p_shape[f : f + 1, :10],
                        global_orient=pred_rotmat[:, 0:1],
                        body_pose=pred_rotmat[:, 1:22],
                        left_hand_pose=pred_rotmat[:, 22:37],
                        right_hand_pose=pred_rotmat[:, 37:52],
                        pose2rot=False,
                    )

                    # Same rendering logic that worked in your validate scripts
                    R_ext = ext[f, :3, :3]
                    verts_rotated = torch.matmul(
                        R_ext, smpl_out.vertices[0].transpose(0, 1)
                    ).transpose(0, 1)
                    verts_cam = verts_rotated + p_cam[f]

                    img_np = (
                        raw_images[f].permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    renderer = MeshRenderer(width=256, height=256)
                    vis_img = renderer.render_mesh(
                        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                        verts_cam.cpu().numpy(),
                        viz_smpl.faces,
                        K=K[f].cpu().numpy(),
                    )
                    cv2.imwrite(
                        f"experiments/diversity/epoch_{epoch}_frame_{f}.jpg", vis_img
                    )
                    renderer.delete()


if __name__ == "__main__":
    main()
