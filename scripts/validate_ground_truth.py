import torch
import cv2
import numpy as np
import smplx
from lookma.dataset import SynthBodyDataset
from lookma.geometry import batch_rodrigues, perspective_projection
from lookma.visualizer import MeshRenderer

# --- CONFIG ---
DATA_PATH = "data/synth_body"
MODEL_PATH = "data/smplx"
IS_TRAIN = False  # True applies data augmentation
TARGET_IMAGE = "img_0000020_003.jpg"

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


def draw_skeleton(img, joints_2d):
    vis_img = img.copy()
    for idx_a, idx_b in SKELETON_BONES:
        if idx_a >= len(joints_2d) or idx_b >= len(joints_2d):
            continue
        pt_a = tuple(joints_2d[idx_a].astype(int))
        pt_b = tuple(joints_2d[idx_b].astype(int))
        color = (0, 255, 0)
        if idx_a in [1, 4, 7, 10, 13, 16, 18, 20]:
            color = (0, 0, 255)
        if idx_a in [2, 5, 8, 11, 14, 17, 19, 21]:
            color = (255, 0, 0)
        cv2.line(vis_img, pt_a, pt_b, color, 2)
    return vis_img


def main():
    print("--- VERIFYING GROUND TRUTH MESH ---")

    # 1. Load Dataset
    dataset = SynthBodyDataset(
        DATA_PATH, specific_image=TARGET_IMAGE, target_size=256, is_train=IS_TRAIN
    )
    data = dataset[0]

    image_tensor = data["image"]
    image_np = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    vis_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)  # For pyrender

    # 2. Setup SMPL
    smpl = smplx.create(
        MODEL_PATH, model_type="smplh", gender="neutral", num_betas=10, use_pca=False
    )
    smpl.pose_mean = torch.tensor([0.0])

    # 3. Generate 3D Geometry
    pose_abs = data["pose"]  # Raw GT (No mean added, per our finding)
    rotmats = batch_rodrigues(pose_abs.unsqueeze(0).view(-1, 52, 3))

    output = smpl(
        betas=data["betas"].unsqueeze(0)[:, :10],
        global_orient=rotmats[:, 0:1],
        body_pose=rotmats[:, 1:22],
        left_hand_pose=rotmats[:, 22:37],
        right_hand_pose=rotmats[:, 37:52],
        pose2rot=False,
    )

    # Get Vertices & Joints in World Space
    # (Note: We use World Space here because we will pass Extrinsics+Intrinsics logic implicitly via K/T or manually project)
    # Actually, for MeshRenderer, it's easier to pass Vertices in CAMERA SPACE.

    # Apply World -> Camera Transformation
    verts_world = output.vertices + data["trans"].unsqueeze(0).unsqueeze(1)  # [1, N, 3]
    joints_world = output.joints + data["trans"].unsqueeze(0).unsqueeze(1)

    ext = data["cam_extrinsics"].unsqueeze(0)  # [1, 4, 4]

    # Transform Vertices
    ones_v = torch.ones(1, verts_world.shape[1], 1)
    verts_homo = torch.cat([verts_world, ones_v], dim=-1)
    verts_cam = torch.matmul(ext, verts_homo.transpose(1, 2)).transpose(1, 2)[..., :3]

    # Transform Joints
    ones_j = torch.ones(1, joints_world.shape[1], 1)
    joints_homo = torch.cat([joints_world, ones_j], dim=-1)
    joints_cam = torch.matmul(ext, joints_homo.transpose(1, 2)).transpose(1, 2)[..., :3]

    # 4. Project Joints (For Skeleton overlay)
    K = data["cam_intrinsics"].unsqueeze(0)
    joints_2d = perspective_projection(joints_cam, torch.zeros(1, 3), K)

    # 5. Render Mesh
    print("Rendering Mesh...")
    renderer = MeshRenderer(width=256, height=256)

    # Pass the calculated Camera Space vertices and the specific K for this crop
    mesh_vis = renderer.render_mesh(
        img_rgb, verts_cam[0].detach().numpy(), smpl.faces, K=K[0].numpy()
    )

    # 6. Draw Skeleton on top
    # Convert back to BGR for OpenCV
    final_bgr = cv2.cvtColor(mesh_vis, cv2.COLOR_RGB2BGR)
    final_combined = draw_skeleton(final_bgr, joints_2d[0].detach().numpy())

    cv2.imwrite("validate_crop.jpg", final_combined)
    print("✅ Saved validate_crop.jpg")
    renderer.delete()


if __name__ == "__main__":
    main()
