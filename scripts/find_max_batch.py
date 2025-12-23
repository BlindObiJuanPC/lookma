import gc
import time

import torch
import torchvision.transforms.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from lookma.dataset import SynthBodyDataset
from lookma.losses import HMRLoss
from lookma.models import HMRBodyNetwork

DEVICE = "cuda"


def test_and_hold(batch_size, dataset):
    print(f"\n🚀 Testing Batch Size: {batch_size}")

    # 1. Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. Setup
    model = HMRBodyNetwork(backbone_name="hrnet_w48").to(DEVICE)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = HMRLoss("data/smplx", device=DEVICE)
    scaler = GradScaler("cuda")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    # 3. Data Prep
    raw_images = batch["image"].to(DEVICE) / 255.0
    gt_pose = batch["pose"].to(DEVICE)
    gt_shape = batch["betas"].to(DEVICE)
    gt_ldmks = batch["landmarks_2d"].to(DEVICE)
    cam_intrinsics = batch["cam_intrinsics"].to(DEVICE)
    ext = batch["cam_extrinsics"].to(DEVICE)
    gt_world_t = batch["trans"].to(DEVICE)

    ones = torch.ones(batch_size, 1, 1, device=DEVICE)
    gt_cam_t = torch.matmul(ext, torch.cat([gt_world_t.unsqueeze(-1), ones], dim=1))[
        :, :3, 0
    ]
    norm_images = TF.normalize(
        raw_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # 4. Run Training Step
    print("   ...Running Forward/Backward pass...")
    optimizer.zero_grad()
    with autocast("cuda"):
        p_pose, p_shape, p_cam, p_ldmk = model(norm_images)
        loss, _, _ = criterion(
            p_pose,
            p_shape,
            p_cam,
            p_ldmk,
            gt_pose,
            gt_shape,
            gt_ldmks,
            gt_cam_t,
            cam_intrinsics,
            ext,
        )

    scaler.scale(loss).backward()
    # We don't need to step the optimizer for a memory test, backward() is the peak.

    # 5. Report and Hold
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    peak_gb = peak_bytes / (1024**3)

    print(f"📊 PYTORCH PEAK VRAM: {peak_gb:.2f} GB")
    print("👀 CHECK TASK MANAGER NOW (Holding for 10 seconds...)")

    # Wait here so you can see Task Manager stay at the peak
    time.sleep(10)

    # 6. Final Cleanup for next run
    del model, optimizer, criterion, batch, raw_images, norm_images, loss
    print("   ...Cleaning up...")


def main():
    print("--- RTX 5090 VRAM SOAKER ---")
    dataset = SynthBodyDataset("data/synth_body", target_size=256, is_train=True)

    # You can manually set the list of batch sizes you want to observe
    # Based on your previous run, let's check the transition around 128-192
    test_list = [64, 128, 160, 192, 224]

    for b in test_list:
        try:
            test_and_hold(b, dataset)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch Size {b} CRASHED with Hard OOM.")
                break
            else:
                raise e


if __name__ == "__main__":
    main()
