import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import torchvision.transforms as T
from lookma.dataset import SynthBodyDataset

# --- CONFIG ---
NUM_SAMPLES = 20
OUTPUT_DIR = "experiments/verify_aug"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- COPY OF GPU AUGMENTATIONS (From train.py) ---
class AddGaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, img):
        if torch.rand(1) < 0.5:
            return img
        return img + torch.randn_like(img) * self.std


class AddISONoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if torch.rand(1) > 0.5:
            return img
        sigma_read = torch.rand(1, device=img.device) * 0.05
        noise_read = torch.randn_like(img) * sigma_read
        gain = torch.rand(1, device=img.device) * 0.05
        shot_std = torch.sqrt(torch.clamp(img, min=1e-5)) * gain
        noise_shot = torch.randn_like(img) * shot_std
        return torch.clamp(img + noise_read + noise_shot, 0, 1)


class Pixelate(nn.Module):
    def forward(self, img):
        if torch.rand(1) < 0.3:
            B, C, H, W = img.shape
            factor = np.random.randint(2, 6)
            small = torch.nn.functional.interpolate(
                img, scale_factor=1 / factor, mode="nearest"
            )
            return torch.nn.functional.interpolate(small, size=(H, W), mode="nearest")
        return img


# Standard Torchvision Color Augs
color_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)


def main():
    print("--- VERIFYING AUGMENTATION PIPELINE ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize Dataset (Train Mode = Augmentation ON)
    dataset = SynthBodyDataset("data/synth_body", target_size=256, is_train=True)

    # 2. Initialize GPU Augments
    gpu_iso = AddISONoise().to(DEVICE)
    gpu_pixel = Pixelate().to(DEVICE)

    print(f"Generating {NUM_SAMPLES} samples...")

    for i in range(NUM_SAMPLES):
        # Load random sample
        # We manually index to simulate a dataloader
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]

        # Image comes as [3, H, W] in 0-255 range (float)
        img_tensor = data["image"].unsqueeze(0).to(DEVICE) / 255.0
        landmarks = data["landmarks_2d"].numpy()

        # --- APPLY TRAINING AUGMENTATIONS ---
        # 1. Color
        img_tensor = color_aug(img_tensor)
        # 2. ISO Noise
        img_tensor = gpu_iso(img_tensor)
        # 3. Pixelate
        img_tensor = gpu_pixel(img_tensor)

        # --- VISUALIZE ---
        # Convert back to CPU Numpy for drawing
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        # These landmarks were transformed in dataset.py to match the crop
        for x, y in landmarks:
            if 0 <= x < 256 and 0 <= y < 256:
                cv2.circle(img_np, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Draw Crop info
        cv2.putText(
            img_np,
            f"Sample {i}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        save_path = os.path.join(OUTPUT_DIR, f"aug_check_{i}.jpg")
        cv2.imwrite(save_path, img_np)
        print(f"Saved {save_path}")

    print("Done. Check the images!")


if __name__ == "__main__":
    main()
