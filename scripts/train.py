import os
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lookma.dataset import SynthBodyDataset, SynthHandDataset
from lookma.helpers.augmentation import TrainingAugmentation
from lookma.helpers.geometry import batch_rodrigues, rotation_6d_to_matrix
from lookma.helpers.visualize_data import draw_mesh
from lookma.losses import BodyLoss, HandLoss
from lookma.models import BodyNetwork, HandNetwork

# --- CONFIGURATION (RTX 5090) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Performance Optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CONFIGS = {
    "body": {
        "batch_size": 128,
        "acc_steps": 2,  # Effective 256
        "lr": 1e-4,
        "epochs": 600,
        "target_size": 256,
        "dataset_cls": SynthBodyDataset,
        "data_path": "data/synth_body",
        "model_cls": BodyNetwork,
        "backbone": "hrnet_w48",
        "loss_cls": BodyLoss,
        "num_workers": 12,
        "has_shape": True,
    },
    "hand": {
        "batch_size": 256,  # Smaller image size (128) allows larger batch
        "acc_steps": 1,  # Effective 256
        "lr": 1e-4,
        "epochs": 600,
        "target_size": 128,  # ROI for hand is smaller
        "dataset_cls": SynthHandDataset,
        "data_path": "data/synth_hand",
        "model_cls": HandNetwork,
        "backbone": "hrnet_w18",  # Lighter backbone
        "loss_cls": HandLoss,
        "num_workers": 12,
        "has_shape": False,
    },
}


def save_debug_image(
    image_tensor,
    pred_pose,
    pred_shape,  # Can be None
    batch,  # Contains all GT data
    epoch,
    batch_idx,
    is_hand=False,
):
    with torch.no_grad():
        # Extact GT data from batch
        gt_pose = batch["pose"].to(DEVICE)
        gt_betas = batch["betas"].to(DEVICE)
        gt_world_t = batch["trans"].to(DEVICE)
        cam_ext = batch["cam_extrinsics"].to(DEVICE)
        K_mat = batch["cam_intrinsics"].to(DEVICE)

        # --- FIX: Cast to .float() to prevent Half/Float mismatch in Mixed Precision ---
        p_pose_single = pred_pose[0:1].float()  # [1, 90 or 126]

        if pred_shape is not None:
            p_shape_single = pred_shape[0:1].float()
        else:
            # Fallback to GT shape if prediction is missing (e.g. for Hand network)
            p_shape_single = gt_betas[0:1].float()

        p_trans_single = gt_world_t[0:1].float()

        # Hybrid Pose Construction (Match Loss Logic)
        # GT Pose is [B, 52, 3] (Axis Angle)
        gt_rotmat = batch_rodrigues(gt_pose[0:1].view(1, 52, 3))

        # Clone GT and overwrite
        full_rotmat = gt_rotmat.clone()

        if not is_hand:
            # Body: Pred Pose is [1, 126] (21 joints * 6D)
            # Overwrite indices 1-21
            p_rotmat = rotation_6d_to_matrix(p_pose_single.view(1, 21, 6))
            full_rotmat[0, 1:22] = p_rotmat[0]
        else:
            # Hand: Pred Pose is [1, 90] (15 joints * 6D)
            # Overwrite indices 22-37 (Left Hand)
            p_rotmat = rotation_6d_to_matrix(p_pose_single.view(1, 15, 6))
            full_rotmat[0, 22:37] = p_rotmat[0]

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
        vis_img = draw_mesh(
            img_bgr,
            p_shape_single[0].cpu().numpy(),
            pose_aa_flat,
            p_trans_single[0].cpu().numpy(),
            cam_ext[0].cpu().numpy(),
            K_mat[0].cpu().numpy(),
        )

        folder_name = "train_hand" if is_hand else "train_body"
        os.makedirs(f"experiments/vis/{folder_name}", exist_ok=True)
        cv2.imwrite(f"experiments/vis/{folder_name}/e{epoch}_b{batch_idx}.jpg", vis_img)


def train(args):
    cfg = CONFIGS[args.type]

    if not args.explain:
        print(
            f"RUNNING {args.type.upper()} TRAINING | "
            f"Batch: {cfg['batch_size']} | "
            f"Eff: {cfg['batch_size'] * cfg['acc_steps']}"
        )

    # Checkpoint Dir
    ckpt_dir = f"experiments/checkpoints/{args.type}"
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = cfg["dataset_cls"](
        cfg["data_path"], target_size=cfg["target_size"], is_train=True
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    model = cfg["model_cls"](backbone_name=cfg["backbone"]).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=1e-4, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    criterion = cfg["loss_cls"]("data/smplx", device=DEVICE)
    # GradScaler not needed for BF16

    if args.compile and not args.explain:
        print("Compiling model (mode='max-autotune')... (First step will be slow)")
        model = torch.compile(model, mode="max-autotune")

    gpu_aug = TrainingAugmentation().to(DEVICE)

    save_interval = 1
    vis_interval = 500

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        if epoch == 1:  # Warmup heads
            for param in model.backbone.parameters():
                param.requires_grad = False
        if epoch == 2:  # Full train
            print("Unfreezing Backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Re-init optimizer so it sees the new parameters
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=1e-4, fused=True
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - 1
            )

        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", disable=args.explain)
        for batch_idx, batch in enumerate(progress_bar):
            raw_imgs = batch["image"].to(DEVICE, non_blocking=True) / 255.0
            if epoch >= 2:
                raw_imgs = gpu_aug(raw_imgs)

            norm_imgs = TF.normalize(
                raw_imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ).to(memory_format=torch.channels_last)

            ext = batch["cam_extrinsics"].to(DEVICE, non_blocking=True)
            gt_world_t = batch["trans"].to(DEVICE, non_blocking=True)
            curr_batch = raw_imgs.shape[0]
            ones = torch.ones(curr_batch, 1, 1, device=DEVICE)
            gt_cam_t = torch.matmul(
                ext, torch.cat([gt_world_t.unsqueeze(-1), ones], dim=1)
            )[:, :3, 0]

            if args.explain:
                import torch._dynamo as _dynamo

                print("Analyzing model for graph breaks (torch._dynamo.explain)...")
                # We explain the 'model' callable with its inputs
                explanation = _dynamo.explain(model)(norm_imgs)
                print(f"\n{explanation.compile_times}")
                print(f"\nGraph Count: {explanation.graph_count}")
                print(f"Graph Breaks: {explanation.graph_break_count}")
                if explanation.break_reasons:
                    print("Break Reasons:")
                    for reason in explanation.break_reasons:
                        print(f"  - {reason.reason}")
                print(f"Op Count: {explanation.op_count}\n")
                return

            with autocast("cuda", dtype=torch.bfloat16):
                # Forward Pass Switch
                if cfg["has_shape"]:
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
                else:
                    p_pose, p_ldmk = model(norm_imgs)
                    p_shape = None
                    loss, comps, _ = criterion(
                        p_pose,
                        p_ldmk,
                        batch["pose"].to(DEVICE),
                        gt_cam_t,
                        batch["cam_intrinsics"].to(DEVICE),
                        ext,
                        batch["betas"].to(DEVICE),  # gt_shape required for HandLoss
                    )

                loss = loss / cfg["acc_steps"]

            loss.backward()
            if (batch_idx + 1) % cfg["acc_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * cfg["acc_steps"]
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item() * cfg['acc_steps']:.2f}",
                    "J3D": f"{comps['loss_joint_t']:.3f}",
                    "Dense": f"{comps['loss_dense']:.3f}",
                }
            )

            if batch_idx % vis_interval == 0:
                save_debug_image(
                    raw_imgs,
                    p_pose,
                    p_shape,
                    batch,
                    epoch,
                    batch_idx,
                    is_hand=(not cfg["has_shape"]),
                )

            if args.dry_run:
                print("Dry run complete (1 batch)")
                break

        if args.dry_run:
            break

        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]
        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Avg Loss: {average_loss:.4f} | LR: {learning_rate:.6f}")

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"{ckpt_dir}/model_epoch_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="body",
        choices=["body", "hand"],
        help="Training type: body or hand",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model using torch.compile (faster)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run torch._dynamo.explain to find graph breaks",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Run a single batch for debugging"
    )
    args = parser.parse_args()

    train(args)
