import cv2
import numpy as np
import smplx
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

# Core lookma imports
from lookma.dataset import SynthBodyDataset
from lookma.geometry import (
    batch_rodrigues,
    perspective_projection,
    rotation_6d_to_matrix,
)
from lookma.losses import HMRLoss
from lookma.models import HMRBodyNetwork

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/synth_body"
MODEL_PATH = "data/smplx"
TARGET_IMAGE = "img_0000020_003.jpg"
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4


def save_debug_image(
    image_tensor, pred_verts, pred_faces, pred_ldmks, gt_ldmks_dense, K_matrix, epoch
):
    img = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Render Mesh
    from lookma.visualizer import MeshRenderer

    renderer = MeshRenderer(width=256, height=256)
    mesh_rgb = renderer.render_mesh(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        pred_verts[0].detach().cpu().numpy(),
        pred_faces,
        K=K_matrix,
    )
    renderer.delete()
    vis_img = cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR)

    # 2. Draw Error Vectors & Confidence
    gt_np = gt_ldmks_dense[0].cpu().numpy()
    pred_np = pred_ldmks[0].detach().cpu().numpy()

    sigmas = []

    for i in range(0, len(pred_np), 2):
        gx, gy = int(gt_np[i, 0]), int(gt_np[i, 1])
        px, py = int(pred_np[i, 0] * 256), int(pred_np[i, 1] * 256)

        log_var = pred_np[i, 2]
        sigma = np.sqrt(np.exp(np.clip(log_var, -10, 10)))
        sigmas.append(sigma)

        # --- SHARPER COLOR GRADIENT ---
        # 0.0 - 1.0 px = Pure Green
        # 1.0 - 3.0 px = Yellow
        # 3.0+ px = Pure Red
        u = np.clip((sigma - 0.5) / 2.5, 0, 1)  # Linear scale from 0.5 to 3.0
        color = (0, int(255 * (1 - u)), int(255 * u))

        if 0 <= gx < 256 and 0 <= gy < 256:
            # Draw prediction dot (larger)
            cv2.circle(vis_img, (px, py), 2, color, -1)
            # Draw target dot (tiny red center)
            cv2.circle(vis_img, (gx, gy), 1, (0, 0, 255), -1)

    print(
        f"📊 Epoch {epoch} Sigma Range: Min={min(sigmas):.2f}, Max={max(sigmas):.2f}, Avg={np.mean(sigmas):.2f}"
    )
    cv2.imwrite(f"debug_epoch_{epoch}.jpg", vis_img)


def main():
    print(f"🚀 Density Check: Overfitting {TARGET_IMAGE} on {DEVICE}")

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
    gt_ldmks_sparse = batch["landmarks_2d"].to(DEVICE)
    K = batch["cam_intrinsics"].to(DEVICE)
    cam_extrinsics = batch["cam_extrinsics"].to(DEVICE)

    # GT Camera-Space Translation
    gt_world_t = batch["trans"].to(DEVICE)
    ones = torch.ones(1, 1, 1, device=DEVICE)
    gt_cam_t = torch.matmul(
        cam_extrinsics, torch.cat([gt_world_t.unsqueeze(-1), ones], dim=1)
    )[:, :3, 0]

    # 2. Setup Model
    model = HMRBodyNetwork(backbone_name="hrnet_w48").to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = HMRLoss(MODEL_PATH, device=DEVICE)

    viz_smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", use_pca=False, num_betas=10
    ).to(DEVICE)
    viz_smpl.pose_mean = torch.tensor([0.0], device=DEVICE)

    # 3. Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        p_pose, p_shape, p_cam, p_ldmk = model(norm_images)

        loss, comps, _ = criterion(
            p_pose,
            p_shape,
            p_cam,
            p_ldmk,
            gt_pose,
            gt_shape,
            gt_ldmks_sparse,
            gt_cam_t,
            K,
            cam_extrinsics,
        )

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

            with torch.no_grad():
                # Prediction Viz
                p_rotmat = rotation_6d_to_matrix(p_pose.view(1, 52, 6))
                p_smpl_out = viz_smpl(
                    betas=p_shape[:, :10],
                    global_orient=p_rotmat[:, 0:1],
                    body_pose=p_rotmat[:, 1:22],
                    left_hand_pose=p_rotmat[:, 22:37],
                    right_hand_pose=p_rotmat[:, 37:52],
                    pose2rot=False,
                )
                R_ext = cam_extrinsics[0, :3, :3]
                p_verts_cam = (
                    torch.matmul(
                        R_ext, p_smpl_out.vertices[0].transpose(0, 1)
                    ).transpose(0, 1)
                    + p_cam[0]
                )

                # Ground Truth Viz (Target for red dots)
                gt_rotmat_raw = batch_rodrigues(gt_pose.view(1, 52, 3))
                gt_smpl_out = viz_smpl(
                    betas=gt_shape[:, :10],
                    body_pose=gt_rotmat_raw[:, 1:22],
                    global_orient=gt_rotmat_raw[:, 0:1],
                    left_hand_pose=gt_rotmat_raw[:, 22:37],
                    right_hand_pose=gt_rotmat_raw[:, 37:52],
                    pose2rot=False,
                )
                v_gt_rot = torch.matmul(
                    R_ext, gt_smpl_out.vertices[0].transpose(0, 1)
                ).transpose(0, 1)
                verts_cam_gt = v_gt_rot + gt_cam_t[0]

                # Full density GT (1378 points)
                gt_dense_3d = verts_cam_gt[::5]
                gt_dense_2d = perspective_projection(
                    gt_dense_3d.unsqueeze(0), torch.zeros(1, 3, device=DEVICE), K
                )

                save_debug_image(
                    raw_images,
                    p_verts_cam.unsqueeze(0),
                    viz_smpl.faces,
                    p_ldmk,
                    gt_dense_2d,
                    K[0].cpu().numpy(),
                    epoch,
                )


if __name__ == "__main__":
    main()
