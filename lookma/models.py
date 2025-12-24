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
    features = self.backbone(x)
    pred_pose = self.pose_head(features)
    pred_shape = self.shape_head(features)

    # --- FIX: Return raw translation (World Space) ---
    # No more 'abs' or '+1' - let the network learn the real coordinates
    pred_cam = self.cam_head(features)

    raw_ldmk = self.ldmk_head(features).view(x.shape[0], self.n_dense, 3)
    xy = torch.sigmoid(raw_ldmk[..., :2])
    log_var = raw_ldmk[..., 2:3]
    pred_ldmk = torch.cat([xy, log_var], dim=-1)

    return pred_pose, pred_shape, pred_cam, pred_ldmk
