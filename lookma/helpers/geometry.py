import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".

    Args:
        d6: [Batch, Joints, 6]
    Returns:
        matrix: [Batch, Joints, 3, 3]
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    # Normalize the first vector
    b1 = F.normalize(a1, dim=-1)

    # Project a2 to be orthogonal to b1
    # b2 = a2 - (b1 . a2) * b1
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)

    # b3 is the cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack them to form the matrix
    return torch.stack((b1, b2, b3), dim=-1)


def batch_rodrigues(
    axisang: torch.Tensor,
) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix.
    """
    shape = axisang.shape
    axisang = axisang.reshape(-1, 3)

    angle = torch.norm(axisang + 1e-8, p=2, dim=1).unsqueeze(-1)
    axis = axisang / angle

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)

    # --- BUG FIX HERE: Unsqueeze to keep dims matching 'cos' [N, 1] ---
    x = axis[:, 0].unsqueeze(-1)
    y = axis[:, 1].unsqueeze(-1)
    z = axis[:, 2].unsqueeze(-1)

    # Rodrigues formula
    # K = [ 0  -z   y]
    #     [ z   0  -x]
    #     [-y   x   0]

    # R = I + sin(theta)K + (1-cos(theta))K^2

    rot_mat = torch.stack(
        [
            one * cos + x * x * (1 - cos),
            x * y * (1 - cos) - z * sin,
            x * z * (1 - cos) + y * sin,
            y * x * (1 - cos) + z * sin,
            one * cos + y * y * (1 - cos),
            y * z * (1 - cos) - x * sin,
            z * x * (1 - cos) - y * sin,
            z * y * (1 - cos) + x * sin,
            one * cos + z * z * (1 - cos),
        ],
        dim=1,
    )

    rot_mat = rot_mat.view(-1, 3, 3)

    if len(shape) > 2:
        rot_mat = rot_mat.view(shape[0], shape[1], 3, 3)

    return rot_mat


def perspective_projection(
    points_3d: torch.Tensor,
    translation: torch.Tensor,
    camera_intrinsics: torch.Tensor,
) -> torch.Tensor:
    """
    Project 3D points to 2D image plane using Full Perspective Projection.

    Args:
        points_3d: [Batch, N_Points, 3] (The skeleton joints)
        translation: [Batch, 3] (The global body position)
        camera_intrinsics: [Batch, 3, 3] (The 'K' matrix)

    Returns:
        points_2d: [Batch, N_Points, 2]
    """
    # 1. Apply Translation
    # X_cam = X_body + T
    # points_3d: [B, N, 3], translation: [B, 1, 3]
    points_cam = points_3d + translation.unsqueeze(1)

    # 2. Apply Intrinsics (Matrix Multiplication)
    # x_homo = K * X_cam
    # We transpose for matmul: (K @ P.T).T
    # K: [B, 3, 3], points_cam.mT: [B, 3, N] -> [B, 3, N]
    projected_homo = torch.matmul(camera_intrinsics, points_cam.transpose(1, 2))
    projected_homo = projected_homo.transpose(1, 2)  # Back to [B, N, 3]

    # 3. Perspective Divide
    # u = x / z, v = y / z
    # Avoid division by zero with eps
    z = torch.clamp(projected_homo[..., 2:], min=0.1)
    xy = projected_homo[..., :2]

    points_2d = xy / z

    return points_2d


def batch_get_global_rotation(
    local_rotmats: torch.Tensor, parents: torch.Tensor
) -> torch.Tensor:
    """
    Compute global rotation matrices from local rotation matrices using the kinematic tree.

    Args:
        local_rotmats: [Batch, N_Joints, 3, 3]
        parents: [N_Joints] (Kinematic tree indices)

    Returns:
        global_rotmats: [Batch, N_Joints, 3, 3]
    """
    batch_size, n_joints, _, _ = local_rotmats.shape
    global_rotmats = [None] * n_joints

    # Iterate through joints (SMPL parents are sorted, so we can iterate in order)
    for i in range(n_joints):
        parent_idx = parents[i]
        if parent_idx < 0:
            # Root joint
            global_rotmats[i] = local_rotmats[:, i]
        else:
            # Child joint: R_global = R_parent_global @ R_local
            global_rotmats[i] = torch.matmul(
                global_rotmats[parent_idx], local_rotmats[:, i]
            )

    return torch.stack(global_rotmats, dim=1)
