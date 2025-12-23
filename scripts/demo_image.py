"""
Example:

    python scripts/demo_image.py --image data/synth_body/img_0000020_003.jpg --out output.jpg

"""

import argparse
import os

import cv2
import numpy as np
import smplx
import torch
import torchvision.transforms.functional as TF

from lookma.geometry import rotation_6d_to_matrix

# Import your custom modules
# Make sure lookma/visualizer.py exists!
from lookma.models import HMRBodyNetwork

try:
    from lookma.visualizer import MeshRenderer

    HAS_RENDERER = True
except ImportError:
    print("⚠️ Could not import MeshRenderer (pyrender). Fallback to skeleton drawing.")
    HAS_RENDERER = False

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "data/smplx"
# Update this to your best checkpoint (e.g. epoch 30 or 40)
CHECKPOINT = "experiments/checkpoints/model_epoch_30.pth"

# Skeleton connections (Fallback visualization)
SKELETON_BONES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (4, 7),
    (5, 8),
    (7, 10),
    (8, 11),
    (3, 6),
    (6, 9),
    (9, 12),
    (9, 13),
    (9, 14),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
]


def preprocess_image(image_path, target_size=256):
    """
    Loads image, pads to square, resizes to target_size (256), and normalizes.
    Returns: tensor, original_canvas, scale_info
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 1. Pad to Square
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))

    # Create black canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Center image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = img_resized

    # 2. Normalize for Network (ImageNet Stats)
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    norm_tensor = TF.normalize(
        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Return
    # norm_tensor: Input to network
    # canvas: The image we draw on (256x256)
    return norm_tensor.unsqueeze(0), canvas, (scale, x_offset, y_offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--out", type=str, default="demo_result.jpg", help="Output filename"
    )
    args = parser.parse_args()

    print(f"🚀 Running Inference on {args.image}...")

    # 1. Load Model
    print("Loading Neural Network...")
    model = HMRBodyNetwork(backbone_name="hrnet_w48")

    if os.path.exists(CHECKPOINT):
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from {CHECKPOINT}")
    else:
        print(f"⚠️ WARNING: Checkpoint not found at {CHECKPOINT}. Using random weights.")

    model.to(DEVICE)
    model.eval()

    # 2. Setup SMPL (Critical for converting Rotation -> Vertices)
    print("Loading SMPL...")
    smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", use_pca=False, num_betas=10
    ).to(DEVICE)
    # Disable internal mean pose addition (matches our training logic)
    smpl.pose_mean = torch.tensor([0.0], device=DEVICE)

    # 3. Preprocess Image
    input_tensor, canvas_img, _ = preprocess_image(args.image, target_size=256)
    input_tensor = input_tensor.to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        pred_pose, pred_shape, pred_cam = model(input_tensor)

    # 5. Decode Output (6D -> Matrices)
    pred_rotmat = rotation_6d_to_matrix(pred_pose.view(1, 52, 6))

    # 6. Run SMPL Forward Pass
    print("Generating Mesh...")
    smpl_output = smpl(
        betas=pred_shape[:, :10],
        global_orient=pred_rotmat[:, 0:1],
        body_pose=pred_rotmat[:, 1:22],
        left_hand_pose=pred_rotmat[:, 22:37],
        right_hand_pose=pred_rotmat[:, 37:52],
        pose2rot=False,
    )

    # 7. Get Vertices in Camera Space
    # Add the predicted translation (pred_cam) to the vertices
    pred_vertices = smpl_output.vertices  # [1, 6890, 3]
    pred_translation = pred_cam.unsqueeze(1)  # [1, 1, 3]

    # Combine (Vertices + Translation)
    final_verts_batch = pred_vertices + pred_translation

    # Convert to CPU Numpy for rendering
    verts_np = final_verts_batch[0].cpu().numpy()
    faces_np = smpl.faces

    # 8. Render
    final_vis = canvas_img.copy()  # RGB numpy array

    if HAS_RENDERER:
        print("Rendering 3D Overlay...")
        renderer = MeshRenderer(
            width=256, height=256, focal_length=256.0, device=DEVICE
        )
        final_vis = renderer.render_mesh(final_vis, verts_np, faces_np)
        renderer.delete()
    else:
        # Fallback: Project joints and draw skeleton
        # (Only runs if pyrender fails to import)
        print("Rendering Skeleton Lines (Fallback)...")
        # Simple weak perspective projection for fallback
        # (x, y) + trans_xy * scale
        # This is rough approximation for visualization if renderer fails
        joints = (smpl_output.joints + pred_translation)[0].cpu().numpy()
        # Assume fx=256, cx=128
        fx = 256.0
        cx, cy = 128.0, 128.0
        for i in range(len(joints)):
            x, y, z = joints[i]
            u = int((x * fx / z) + cx)
            v = int((y * fx / z) + cy)
            joints[i] = [u, v, z]  # Store 2D

        vis_bgr = cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR)
        for idx_a, idx_b in SKELETON_BONES:
            if idx_a < len(joints) and idx_b < len(joints):
                pt_a = (int(joints[idx_a][0]), int(joints[idx_a][1]))
                pt_b = (int(joints[idx_b][0]), int(joints[idx_b][1]))
                cv2.line(vis_bgr, pt_a, pt_b, (0, 255, 0), 2)
        final_vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # 9. Save
    # Convert RGB -> BGR for OpenCV
    save_img = cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, save_img)
    print(f"✅ Saved result to {args.out}")


if __name__ == "__main__":
    main()
