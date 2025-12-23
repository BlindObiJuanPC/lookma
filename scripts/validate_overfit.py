import torch
import cv2
import numpy as np
import smplx
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from lookma.dataset import SynthBodyDataset
from lookma.models import HMRBodyNetwork
from lookma.losses import HMRLoss
from lookma.geometry import rotation_6d_to_matrix

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/synth_body"
MODEL_PATH = "data/smplx"
TARGET_IMAGE = "img_0000020_003.jpg"

try:
    from lookma.visualizer import MeshRenderer

    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False


def main():
    print(f"🚀 Synchronized Overfit Test on {DEVICE}")

    # 1. Load Data
    dataset = SynthBodyDataset(
        DATA_PATH, specific_image=TARGET_IMAGE, target_size=256, is_train=False
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    raw_images = batch["image"].to(DEVICE) / 255.0
    norm_images = TF.normalize(
        raw_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    gt_pose = batch["pose"].to(DEVICE)
    gt_shape = batch["betas"].to(DEVICE)
    gt_ldmks = batch["landmarks_2d"].to(DEVICE)
    K = batch["cam_intrinsics"].to(DEVICE)
    cam_extrinsics = batch["cam_extrinsics"].to(DEVICE)

    # Pre-calculate Ground Truth Camera-Space Translation
    gt_trans_world = batch["trans"].to(DEVICE)
    ones = torch.ones(1, 1, 1, device=DEVICE)
    gt_trans_homo = torch.cat([gt_trans_world.unsqueeze(-1), ones], dim=1)
    gt_trans_cam = torch.matmul(cam_extrinsics, gt_trans_homo)[:, :3, 0]

    # 2. Setup Model
    model = HMRBodyNetwork(backbone_name="hrnet_w48").to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = HMRLoss(MODEL_PATH, device=DEVICE)

    viz_smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", num_betas=10, use_pca=False
    ).to(DEVICE)
    viz_smpl.pose_mean = torch.tensor([0.0], device=DEVICE)

    for epoch in range(1, 1001):
        optimizer.zero_grad()
        # Network outputs
        pred_pose, pred_shape, pred_cam, pred_ldmk = model(norm_images)

        # Loss (Pass cam_extrinsics correctly)
        loss, components, _ = criterion(
            pred_pose,
            pred_shape,
            pred_cam,
            pred_ldmk,
            gt_pose,
            gt_shape,
            gt_ldmks,
            gt_trans_cam,
            K,
            cam_extrinsics,
        )

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

            with torch.no_grad():
                # 3. Generate Prediction for Visualization
                pred_rotmat = rotation_6d_to_matrix(pred_pose.view(1, 52, 6))
                smpl_out = viz_smpl(
                    betas=pred_shape[:, :10],
                    global_orient=pred_rotmat[:, 0:1],
                    body_pose=pred_rotmat[:, 1:22],
                    left_hand_pose=pred_rotmat[:, 22:37],
                    right_hand_pose=pred_rotmat[:, 37:52],
                    pose2rot=False,
                )

                # --- SYNCED VISUALIZATION MATH ---
                # This matches your working validate_ground_truth.py logic exactly.
                # 1. Rotate Local Vertices to Camera Frame
                R_ext = cam_extrinsics[0, :3, :3]
                verts_local = smpl_out.vertices[0]
                verts_rotated = torch.matmul(
                    R_ext, verts_local.transpose(0, 1)
                ).transpose(0, 1)

                # 2. Add Predicted Camera-Space Translation
                # pred_cam is already in camera space
                verts_cam = verts_rotated + pred_cam[0]

                # 4. Render
                img_np = (raw_images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
                if HAS_RENDERER:
                    renderer = MeshRenderer(width=256, height=256)
                    vis_img = renderer.render_mesh(
                        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                        verts_cam.cpu().numpy(),
                        viz_smpl.faces,
                        K=K[0].cpu().numpy(),
                    )
                    cv2.imwrite(f"debug_epoch_{epoch}.jpg", vis_img)
                    renderer.delete()


if __name__ == "__main__":
    main()
