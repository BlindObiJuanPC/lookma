import torch
import torch.nn as nn
import timm


class HMRBodyNetwork(nn.Module):
    def __init__(self, backbone_name="hrnet_w48", pretrained=True):
        super().__init__()

        # 1. Load Backbone
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features

        # 2. Define Landmark Count (Every 5th vertex of 6890 SMPL mesh)
        self.n_dense = 1378

        # 3. Define Heads (Paper Spec: 2 FC layers, Hidden 512, Leaky ReLU)
        def make_head(output_dim):
            return nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(512, output_dim),
            )

        self.pose_head = make_head(52 * 6)  # 312
        self.shape_head = make_head(16)  # 16
        self.cam_head = make_head(3)  # 3
        self.ldmk_head = make_head(self.n_dense * 3)  # 1378 * 3

        # 4. Initialize Weights
        # Camera Z init to 5.0m
        nn.init.constant_(self.cam_head[-1].weight, 0)
        nn.init.constant_(self.cam_head[-1].bias, 0)
        self.cam_head[-1].bias.data[2] = 5.0

        # Pose Head init to small noise (starts near Mean Pose)
        nn.init.normal_(self.pose_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.pose_head[-1].bias, 0)

        # Landmark Head init (starts near center of image)
        nn.init.constant_(self.ldmk_head[-1].weight, 0)
        nn.init.constant_(self.ldmk_head[-1].bias, 0)

    def forward(self, x):
        """
        Returns:
            pred_pose: [B, 312]
            pred_shape: [B, 16]
            pred_cam: [B, 3]
            pred_ldmk: [B, 1378, 3] (x, y, log_var)
        """
        features = self.backbone(x)

        # 1. Pose and Shape
        pred_pose = self.pose_head(features)
        pred_shape = self.shape_head(features)

        # 2. Camera (with positive depth enforcement)
        raw_cam = self.cam_head(features)
        pred_cam = torch.stack(
            [raw_cam[:, 0], raw_cam[:, 1], torch.abs(raw_cam[:, 2]) + 1.0], dim=-1
        )

        # 3. Landmarks (with 0-1 scaling for X, Y)
        # Reshape flat output to [Batch, N, 3]
        raw_ldmk = self.ldmk_head(features).view(x.shape[0], self.n_dense, 3)

        # x, y: Use sigmoid so they stay within image bounds [0, 1]
        xy = torch.sigmoid(raw_ldmk[..., :2])

        # log_var: Leave as raw number (Loss handles exp)
        log_var = raw_ldmk[..., 2:3]

        pred_ldmk = torch.cat([xy, log_var], dim=-1)

        return pred_pose, pred_shape, pred_cam, pred_ldmk
