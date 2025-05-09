import torch
import torch.nn as nn
import torchvision.ops as ops


class GraspAndClassHead(nn.Module):
    def __init__(self, in_channels=576, num_classes=15):
        super().__init__()
        self.shared_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [N, 576, 1, 1]
            nn.Flatten(),  # [N, 576]
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
        )

        # Grasp box prediction branch
        self.grasp_box_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),  # ReLU(inplace=True)
            nn.Dropout(p=0.2),
            nn.Linear(256, 4),  # (cx, cy, w, h)
        )

        # Grasp angle prediction branch
        self.grasp_angle_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),  # angle
        )

        # Object classification branch
        self.classification_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),  # class logits
        )

    def forward(self, x):
        feat = self.shared_feature(x)

        box = torch.sigmoid(self.grasp_box_branch(feat))  # [N, 4]
        angle = torch.tanh(self.grasp_angle_branch(feat))  # [N, 1]
        class_logits = self.classification_branch(feat)  # [N, num_classes]

        return box, angle, class_logits


class GraspHeadROI(nn.Module):
    def __init__(self, device="cuda", in_channels=576, num_classes=15):
        super().__init__()
        self.device = device
        self.roi_align = ops.RoIAlign(
            output_size=(7, 7), spatial_scale=1.0 / 32.0, sampling_ratio=2
        )
        self.centroid_head = GraspAndClassHead(
            in_channels=in_channels, num_classes=num_classes
        )

    def forward(self, feats, boxes):
        # box size filtering
        w = boxes[:, 3] - boxes[:, 1]
        h = boxes[:, 4] - boxes[:, 2]
        valid = (w > 1) & (h > 1)
        if valid.sum() == 0:
            raise RuntimeError("No valid boxes in batch!")

        boxes = (
            boxes[valid].type(feats.dtype).to(feats.device)
        )  # [N, num_of_detected, 4(xyxy)]

        rois = boxes
        crops = self.roi_align(feats, rois)  # [N, 576, 7, 7]
        return self.centroid_head(crops)  # box, angle, class_logits
