import cv2
import numpy as np
import torch
from lookma.dataset import SynthBodyRoiDataset
from lookma.models import ROINetwork


def verify():
    print("Initializing Dataset...")
    dataset = SynthBodyRoiDataset(
        root_dir="data/synth_body",
        target_size=256,
        is_train=True,
        return_debug_info=True,
    )

    print(f"Dataset Length: {len(dataset)}")

    # Check Model
    print("Initializing Model...")
    model = ROINetwork(backbone_name="resnet18").cpu()
    print("Model initialized.")

    # Visualize first 5 samples
    for i in range(5):
        print(f"Processing Sample {i}...")
        sample = dataset[i]

        # Unpack
        img_tensor = sample["image"]  # [3, 256, 256]
        landmarks_2d = sample["landmarks_2d"]  # [36, 2]

        # Run Model Check
        with torch.no_grad():
            input_tensor = img_tensor.unsqueeze(0).cpu()
            pred = model(input_tensor)
            print(f"Model Output Shape: {pred.shape}")

        # Visual Debug
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        h, w = img_bgr.shape[:2]
        for ldmk in landmarks_2d:
            x = int(ldmk[0] * w)
            y = int(ldmk[1] * h)
            cv2.circle(img_bgr, (x, y), 3, (0, 0, 255), -1)

        cv2.imwrite(f"verify_roi_crop_{i}.jpg", img_bgr)
        print(f"Saved verify_roi_crop_{i}.jpg")

        # Also save original with bbox debug
        if "original_image" in sample:
            orig = sample["original_image"].copy()
            bbox = sample["debug_bbox"]
            # Note: bbox is in rotated space, so drawing it on original image is hard without rotating original image first.
            # But we can verify the crop looks good.
            pass


if __name__ == "__main__":
    verify()
