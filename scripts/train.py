import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lookma.dataset import SynthBodyDataset
from lookma.losses import HMRLoss
from lookma.models import HMRBodyNetwork

# --- PRODUCTION CONFIG FOR RTX 5090 ---
BATCH_SIZE = 128  # Optimized for 256x256 resolution on 5090
ACCUMULATION_STEPS = 2  # Effective Batch = 256 (Matches the Paper exactly)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 60
SAVE_INTERVAL = 1
NUM_WORKERS = 12  # Boosted for the 5090's CPU power
NUM_IMAGES = None  # USE ALL 95,000 IMAGES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Appearance Augmentations
color_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)


class AddISONoise(nn.Module):
    def forward(self, img):
        if torch.rand(1) > 0.5:
            return img
        sigma_read = torch.rand(1, device=img.device) * 0.05
        gain = torch.rand(1, device=img.device) * 0.05
        shot_std = torch.sqrt(torch.clamp(img, min=1e-5)) * gain
        return torch.clamp(
            img + torch.randn_like(img) * sigma_read + torch.randn_like(img) * shot_std,
            0,
            1,
        )


class Pixelate(nn.Module):
    def forward(self, img):
        if torch.rand(1) < 0.3:
            B, C, H, W = img.shape
            factor = np.random.randint(2, 5)
            small = torch.nn.functional.interpolate(
                img, scale_factor=1 / factor, mode="nearest"
            )
            return torch.nn.functional.interpolate(small, size=(H, W), mode="nearest")
        return img


def train():
    print(f"🚀 STARTING PRODUCTION RUN | Eff Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    os.makedirs("experiments/checkpoints", exist_ok=True)

    # 1. Dataset & Loader
    dataset = SynthBodyDataset("data/synth_body", target_size=256, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Model, Optimizer, Scheduler
    model = HMRBodyNetwork(backbone_name="hrnet_w48").to(DEVICE)

    # Optimizer starts with everything, but we freeze backbone in loop for Epoch 1
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    criterion = HMRLoss("data/smplx", device=DEVICE)
    scaler = GradScaler("cuda")

    gpu_iso = AddISONoise().to(DEVICE)
    gpu_pixel = Pixelate().to(DEVICE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        # Step 1: Freeze Backbone for the first epoch (Warmup)
        if epoch == 1:
            for param in model.backbone.parameters():
                param.requires_grad = False
        if epoch == 2:
            print("🔓 Unfreezing Backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        progress_bar = tqdm(loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Load Data
            raw_imgs = batch["image"].to(DEVICE, non_blocking=True) / 255.0
            gt_pose = batch["pose"].to(DEVICE, non_blocking=True)
            gt_shape = batch["betas"].to(DEVICE, non_blocking=True)
            gt_ldmks = batch["landmarks_2d"].to(DEVICE, non_blocking=True)
            cam_intrinsics = batch["cam_intrinsics"].to(DEVICE, non_blocking=True)
            cam_extrinsics = batch["cam_extrinsics"].to(DEVICE, non_blocking=True)
            gt_world_t = batch["trans"].to(DEVICE, non_blocking=True)

            # --- Augmentation (Post-Epoch 1) ---
            if epoch >= 2:
                raw_imgs = color_aug(raw_imgs)
                raw_imgs = gpu_iso(raw_imgs)
                raw_imgs = gpu_pixel(raw_imgs)

            # --- Pre-calculate Camera Space Translation ---
            # Using current batch_size (might be last batch)
            curr_batch = raw_imgs.shape[0]
            ones = torch.ones(curr_batch, 1, 1, device=DEVICE)
            gt_cam_t = torch.matmul(
                cam_extrinsics, torch.cat([gt_world_t.unsqueeze(-1), ones], dim=1)
            )[:, :3, 0]

            # --- Normalization ---
            norm_imgs = TF.normalize(
                raw_imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            with autocast("cuda"):
                # Forward Pass
                p_pose, p_shape, p_cam, p_ldmk = model(norm_imgs)

                # Loss Calculation
                loss, comps, _ = criterion(
                    p_pose,
                    p_shape,
                    p_cam,
                    p_ldmk,
                    gt_pose,
                    gt_shape,
                    gt_ldmks,
                    gt_cam_t,
                    cam_intrinsics,
                    cam_extrinsics,
                )
                loss = loss / ACCUMULATION_STEPS

            # Backward
            scaler.scale(loss).backward()

            # Step
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item()*ACCUMULATION_STEPS:.2f}",
                    "J3D": f"{comps['loss_joint_t']:.3f}",
                }
            )

        scheduler.step()
        print(f"Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")

        # Save every epoch
        torch.save(
            model.state_dict(), f"experiments/checkpoints/model_epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    train()
