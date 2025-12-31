import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lookma.dataset import SynthBodyDataset
from lookma.helpers.augmentation import TrainingAugmentation
from lookma.helpers.geometry import batch_rodrigues, rotation_6d_to_matrix
from lookma.helpers.visualize_data import draw_mesh
from lookma.losses import BodyLoss
from lookma.models import BodyNetwork

# --- FINAL PRODUCTION CONFIG (RTX 5090) ---
BATCH_SIZE = 128  # Calculated from a VRAM stress test
ACCUMULATION_STEPS = 2  # 128 * 2 = 256 Effective Batch (Official Paper Spec)
LEARNING_RATE = 1e-4  # Official Paper Spec
NUM_EPOCHS = 60  # Long-form convergence
SAVE_INTERVAL = 1
VIS_INTERVAL = 500
NUM_WORKERS = 12  # 5090 needs fast CPU feeding to stay busy
NUM_IMAGES = None  # ALL 95,000 IMAGES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- AUGMENTATION ---


def save_debug_image(
    image_tensor,
    pred_pose,
    pred_shape,
    gt_pose,
    gt_world_t,
    cam_ext,
    K_mat,
    epoch,
    batch_idx,
):
    with torch.no_grad():
        # --- FIX: Cast to .float() to prevent Half/Float mismatch in Mixed Precision ---
        p_pose_single = pred_pose[0:1].float()
        p_shape_single = pred_shape[0:1].float()
        p_trans_single = gt_world_t[0:1].float()

        # Prediction to Matrices [1, 21, 3, 3] (Body Only)
        p_rotmat = rotation_6d_to_matrix(p_pose_single.view(1, 21, 6))

        # Hybrid Pose Construction (Match Loss Logic)
        # GT Pose is [B, 52, 3] (Axis Angle) -> [B, 52, 3, 3] (RotMat) for easy blending
        gt_rotmat = batch_rodrigues(gt_pose[0:1].view(1, 52, 3))

        # Clone GT and overwrite Body joints (1-21) with Prediction
        full_rotmat = gt_rotmat.clone()
        full_rotmat[0, 1:22] = p_rotmat[0]

        # Convert Hybrid Matrices to Axis-Angle for draw_mesh
        rots_np = full_rotmat[0].cpu().numpy()  # [52, 3, 3]
        pose_aa_flat = []
        for i in range(52):
            aa, _ = cv2.Rodrigues(rots_np[i])
            pose_aa_flat.append(aa.squeeze())
        pose_aa_flat = np.concatenate(pose_aa_flat)  # [156]

        img_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Call Visualizer
        # We pass GT World Translation + Camera Extrinsics
        # draw_mesh will run: Verts = SMPL(Pose, Trans); Render(Verts, Ext)
        vis_img = draw_mesh(
            img_bgr,
            p_shape_single[0].cpu().numpy(),
            pose_aa_flat,
            p_trans_single[0].cpu().numpy(),
            cam_ext[0].cpu().numpy(),
            K_mat[0].cpu().numpy(),
        )

        os.makedirs("experiments/vis", exist_ok=True)
        cv2.imwrite(f"experiments/vis/train_e{epoch}_b{batch_idx}.jpg", vis_img)


def train():
    print(
        f"PRODUCTION RUN STARTING | Batch: {BATCH_SIZE} | Eff: {BATCH_SIZE * ACCUMULATION_STEPS}"
    )
    os.makedirs("experiments/checkpoints", exist_ok=True)

    dataset = SynthBodyDataset("data/synth_body", target_size=256, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = BodyNetwork(backbone_name="hrnet_w48").to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = BodyLoss("data/smplx", device=DEVICE)
    scaler = GradScaler("cuda")

    gpu_aug = TrainingAugmentation().to(DEVICE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        if epoch == 1:  # Warmup heads
            for param in model.backbone.parameters():
                param.requires_grad = False
        if epoch == 2:  # Full train
            print("Unfreezing Backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Re-init optimizer so it sees the new parameters
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - 1
            )

        progress_bar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            raw_imgs = batch["image"].to(DEVICE, non_blocking=True) / 255.0
            if epoch >= 2:
                raw_imgs = gpu_aug(raw_imgs)

            norm_imgs = TF.normalize(
                raw_imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            ext = batch["cam_extrinsics"].to(DEVICE, non_blocking=True)
            gt_world_t = batch["trans"].to(DEVICE, non_blocking=True)
            curr_batch = raw_imgs.shape[0]
            ones = torch.ones(curr_batch, 1, 1, device=DEVICE)
            gt_cam_t = torch.matmul(
                ext, torch.cat([gt_world_t.unsqueeze(-1), ones], dim=1)
            )[:, :3, 0]

            with autocast("cuda"):
                p_pose, p_shape, p_ldmk = model(norm_imgs)
                loss, comps, _ = criterion(
                    p_pose,
                    p_shape,
                    p_ldmk,
                    batch["pose"].to(DEVICE),
                    batch["betas"].to(DEVICE),
                    gt_cam_t,
                    batch["cam_intrinsics"].to(DEVICE),
                    ext,
                )
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item() * ACCUMULATION_STEPS:.2f}",
                    "J3D": f"{comps['loss_joint_t']:.3f}",
                }
            )

            if batch_idx % VIS_INTERVAL == 0:
                save_debug_image(
                    raw_imgs,
                    p_pose,
                    p_shape,
                    batch["pose"].to(DEVICE),
                    gt_world_t,
                    ext,
                    batch["cam_intrinsics"],
                    epoch,
                    batch_idx,
                )

        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]
        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Avg Loss: {average_loss:.4f} | LR: {learning_rate:.6f}")
        torch.save(
            model.state_dict(), f"experiments/checkpoints/model_epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    train()
