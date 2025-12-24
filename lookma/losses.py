import torch
import torch.nn as nn
import smplx
from .geometry import batch_rodrigues, rotation_6d_to_matrix, perspective_projection


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
        self.w_trans = 5.0

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_pose_6d,
        pred_shape,
        pred_cam,
        pred_ldmk,
        gt_pose_aa,
        gt_shape,
        gt_ldmks_2d,
        gt_translation_world,
        cam_intrinsics,
        cam_extrinsics,
    ):
        batch_size = pred_pose_6d.shape[0]
        device = pred_pose_6d.device

        # 1. PARAMETERS
        pred_rotmat = rotation_6d_to_matrix(pred_pose_6d.view(batch_size, 52, 6))
        gt_rotmat = batch_rodrigues(gt_pose_aa.view(batch_size, 52, 3))

        # 2. GENERATE 3D BODIES (World Space)
        # We add the predicted/GT translation to the mesh BEFORE camera transform
        p_out = self.smpl(
            betas=pred_shape[:, :10],
            global_orient=pred_rotmat[:, 0:1],
            body_pose=pred_rotmat[:, 1:22],
            left_hand_pose=pred_rotmat[:, 22:37],
            right_hand_pose=pred_rotmat[:, 37:52],
            pose2rot=False,
        )
        p_joints_world = p_out.joints + pred_cam.unsqueeze(1)

        with torch.no_grad():
            g_out = self.smpl(
                betas=gt_shape[:, :10],
                body_pose=gt_rotmat[:, 1:22],
                global_orient=gt_rotmat[:, 0:1],
                left_hand_pose=gt_rotmat[:, 22:37],
                right_hand_pose=gt_rotmat[:, 37:52],
                pose2rot=False,
            )
            g_joints_world = g_out.joints + gt_translation_world.unsqueeze(1)

        # 3. TRANSFORM TO CAMERA SPACE (Using 4x4 Extrinsics)
        # This is the 'Golden Logic' from validate_ground_truth
        def to_cam(joints_world, ext_mat):
            ones = torch.ones(batch_size, joints_world.shape[1], 1, device=device)
            joints_homo = torch.cat([joints_world, ones], dim=-1)
            joints_cam = torch.matmul(ext_mat, joints_homo.transpose(1, 2)).transpose(
                1, 2
            )
            return joints_cam[..., :3]

        p_joints_cam = to_cam(p_joints_world, cam_extrinsics)
        g_joints_cam = to_cam(g_joints_world, cam_extrinsics)

        # 4. LOSSES
        loss_pose = self.l1(pred_rotmat, gt_rotmat)
        loss_shape = self.l1(pred_shape, gt_shape)
        loss_joint_t = self.l1(p_joints_cam, g_joints_cam)
        loss_joint_r = geodesic_loss(pred_rotmat, gt_rotmat)
        loss_trans = self.l1(pred_cam, gt_translation_world)  # Supervise world pos

        # DENSE LANDMARKS
        with torch.no_grad():
            g_verts_world = g_out.vertices[:, ::5] + gt_translation_world.unsqueeze(1)
            g_verts_cam = to_cam(g_verts_world, cam_extrinsics)
            gt_dense_2d = perspective_projection(
                g_verts_cam, torch.zeros(batch_size, 3, device=device), cam_intrinsics
            )

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
            + (self.w_trans * loss_trans)
        )

        pred_joints_2d = perspective_projection(
            p_joints_cam, torch.zeros(batch_size, 3, device=device), cam_intrinsics
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
