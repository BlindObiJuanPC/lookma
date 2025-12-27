import smplx
import torch
import torch.nn as nn

from lookma.helpers.geometry import (
    batch_get_global_rotation,
    batch_rodrigues,
    perspective_projection,
    rotation_6d_to_matrix,
)


def geodesic_loss(R1, R2):
    R_diff = torch.matmul(R1, R2.transpose(-1, -2))
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    val = torch.clamp((trace - 1) / 2, -0.9999, 0.9999)
    return torch.acos(val).mean()


class HMRLoss(nn.Module):
    def __init__(self, smpl_model_path, device="cuda"):
        super().__init__()
        self.smpl = smplx.create(
            smpl_model_path,
            model_type="smplh",
            gender="neutral",
            use_pca=False,
            num_betas=10,
        ).to(device)
        self.smpl.pose_mean = torch.tensor([0.0], device=device)

        self.w_dense = 10.0
        self.w_pose = 1.0
        self.w_shape = 1.0
        self.w_joint_t = 5.0
        self.w_joint_r = 1.0

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_pose_6d,
        pred_shape,
        pred_ldmk,
        gt_pose_aa,
        gt_shape,
        gt_translation_cam,
        cam_intrinsics,
        cam_extrinsics,
    ):
        batch_size = pred_pose_6d.shape[0]
        device = pred_pose_6d.device

        # 1. PREPARE PARAMETERS
        # Converting 21 body joints to Rotation Matrices
        pred_body_rotmat = rotation_6d_to_matrix(pred_pose_6d.view(batch_size, 21, 6))
        gt_rotmat = batch_rodrigues(gt_pose_aa.view(batch_size, 52, 3))

        # Reconstruct Full Pose: GT Root (0) + Pred Body (1-21) + GT Hands (22-51)
        # We clone GT first to fill in 0 and 22-51 automatically
        full_pred_rotmat = gt_rotmat.clone()
        full_pred_rotmat[:, 1:22] = pred_body_rotmat

        # 2. RUN KINEMATICS (Local Space)
        pred_output = self.smpl(
            betas=pred_shape[:, :10],
            global_orient=full_pred_rotmat[:, 0:1],
            body_pose=full_pred_rotmat[:, 1:22],
            left_hand_pose=full_pred_rotmat[:, 22:37],
            right_hand_pose=full_pred_rotmat[:, 37:52],
            pose2rot=False,
        )

        with torch.no_grad():
            gt_output = self.smpl(
                betas=gt_shape[:, :10],
                body_pose=gt_rotmat[:, 1:22],
                global_orient=gt_rotmat[:, 0:1],
                left_hand_pose=gt_rotmat[:, 22:37],
                right_hand_pose=gt_rotmat[:, 37:52],
                pose2rot=False,
            )

        # 4. LOSSES
        # Pose/Rot (Local): Indices 1-21 (Body)
        loss_pose = self.l1(pred_body_rotmat, gt_rotmat[:, 1:22])

        # Pose/Rot (Global/World): Indices 1-21 (Body)
        # 1. Compute Global Rotations via FK
        pred_global_rotmat = batch_get_global_rotation(
            full_pred_rotmat, self.smpl.parents
        )
        gt_global_rotmat = batch_get_global_rotation(gt_rotmat, self.smpl.parents)
        # 2. Compare Global Rotations (Geodesic)
        loss_joint_r = geodesic_loss(
            pred_global_rotmat[:, 1:22], gt_global_rotmat[:, 1:22]
        )

        loss_shape = self.l1(pred_shape, gt_shape[:, :10])

        # Joint Position (World): Indices 0-21 (Body + Pelvis)
        # Comparing SMPL output joints directly (conceptually + T_gt on both sides)
        loss_joint_t = self.l1(pred_output.joints[:, :22], gt_output.joints[:, :22])

        # DENSE LANDMARKS
        with torch.no_grad():
            R_ext = cam_extrinsics[:, :3, :3]
            gt_verts_rotated = torch.matmul(
                R_ext, gt_output.vertices[:, ::5].transpose(1, 2)
            ).transpose(1, 2)
            gt_verts_cam = gt_verts_rotated + gt_translation_cam.unsqueeze(1)
            gt_dense_2d = perspective_projection(
                gt_verts_cam, torch.zeros(batch_size, 3, device=device), cam_intrinsics
            )

        # Scale predicted 0-1 coords to pixel space (256)
        pred_xy = pred_ldmk[..., :2] * 256.0
        pred_log_var = torch.clamp(pred_ldmk[..., 2], -10.0, 10.0)

        dist_sq = (pred_xy - gt_dense_2d).pow(2).sum(dim=-1)

        loss_dense = (torch.exp(-pred_log_var) * dist_sq + pred_log_var) * 0.5
        loss_dense = loss_dense.mean()

        total_loss = (
            (self.w_pose * loss_pose)
            + (self.w_shape * loss_shape)
            + (self.w_joint_t * loss_joint_t)
            + (self.w_joint_r * loss_joint_r)
            + (self.w_dense * loss_dense)
        )

        # Re-Project for debug visualization
        # Use GT Camera parameters since we don't predict them anymore
        # 1. Rotate to Cam Frame
        pred_joints_rotated = torch.matmul(
            R_ext, pred_output.joints.transpose(1, 2)
        ).transpose(1, 2)
        # 2. Add GT Translation
        pred_joints_cam = pred_joints_rotated + gt_translation_cam.unsqueeze(1)
        # 3. Project
        pred_joints_2d = perspective_projection(
            pred_joints_cam, torch.zeros(batch_size, 3, device=device), cam_intrinsics
        )

        return (
            total_loss,
            {
                "loss_pose": loss_pose.item(),
                "loss_joint_t": loss_joint_t.item(),
                "loss_dense": loss_dense.item(),
            },
            pred_joints_2d,
        )
