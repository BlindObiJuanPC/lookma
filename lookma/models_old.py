import timm
import torch
import torch.nn as nn


class HMRBodyNetwork(nn.Module):
    def __init__(self, backbone_name="hrnet_w48", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features

        # Paper Spec: Two FC layers, Hidden 512, Leaky ReLU
        def make_head(output_dim):
            return nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(512, output_dim),
            )

        self.pose_head = make_head(52 * 6)  # 312
        self.shape_head = make_head(16)  # 16
        self.cam_head = make_head(3)  # 3

        self.num_dense_landmarks = 1378  # SMPL-H has 6890 vertices divided by 5 = 1378
        self.ldmk_head = make_head(self.num_dense_landmarks * 3)  # (x, y, confidence)

        # Initialize Camera Z to ~3.0m (Safe start)
        nn.init.constant_(self.cam_head[-1].weight, 0)
        nn.init.constant_(self.cam_head[-1].bias, 0)
        self.cam_head[-1].bias.data[2] = 3.0

        # Initialize Pose to Mean (0.0)
        nn.init.normal_(self.pose_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.pose_head[-1].bias, 0)

    def forward(self, x):
        features = self.backbone(x)

        pred_pose = self.pose_head(features)
        pred_shape = self.shape_head(features)

        raw_cam = self.cam_head(features)
        pred_cam = torch.stack(
            [raw_cam[:, 0], raw_cam[:, 1], torch.abs(raw_cam[:, 2]) + 1.0], dim=-1
        )

        # --- FIX: LANDMARK SCALING ---
        raw_ldmk = self.ldmk_head(features).view(x.shape[0], self.n_dense, 3)

        # 1. Force X and Y to be between 0 and 1
        # This prevents the "Top-Left Cluster" bug
        xy = torch.sigmoid(raw_ldmk[..., :2])

        # 2. Keep log-variance as a raw number
        log_var = raw_ldmk[..., 2:3]

        pred_ldmk = torch.cat([xy, log_var], dim=-1)

        return pred_pose, pred_shape, pred_cam, pred_ldmk
