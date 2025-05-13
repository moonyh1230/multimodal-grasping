import torch
import torch.nn as nn
import torchvision.ops as ops

from utils.metrics import expand_bbox_xyxy_tensor


class GraspAndClassHead(nn.Module):
    def __init__(self, in_channels=576, in_class_channels=576, num_classes=15):
        super().__init__()
        self.shared_feature = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),  # (b, 256, 7, 7)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # (b, 128, 7, 7)
            nn.ReLU(inplace=True),
            nn.Flatten(),  # (b, 128 * 7 * 7)
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [N, 576, 1, 1]
            nn.Flatten(),  # [N, 576]
            nn.Linear(in_class_channels, 512),
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
            nn.Linear(256, 2),  # sin(theta), cos(theta)
        )

        # Object classification branch
        self.classification_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),  # class logits
        )

    def forward(self, x):
        feat = self.shared_feature(x[0])
        class_feat = self.avg_pool(x[1])

        box = torch.sigmoid(self.grasp_box_branch(feat))  # [N, 4]
        angle = torch.tanh(self.grasp_angle_branch(feat))  # [N, 1]
        class_logits = self.classification_branch(class_feat)  # [N, num_classes]

        return box, angle, class_logits


class GraspHeadROI(nn.Module):
    def __init__(
        self,
        device="cuda",
        in_channels=192,
        in_class_channels=576,
        num_classes=15,
        feat_size=80,
    ):

        super().__init__()
        self.device = device
        self.roi_align = ops.RoIAlign(
            output_size=(7, 7), spatial_scale=feat_size / 640, sampling_ratio=2
        )
        self.roi_align_class = ops.RoIAlign(
            output_size=(7, 7), spatial_scale=(feat_size / 4) / 640, sampling_ratio=2
        )
        self.centroid_head = GraspAndClassHead(
            in_channels=in_channels,
            in_class_channels=in_class_channels,
            num_classes=num_classes,
        )

    def forward(self, feats, boxes):
        # box size filtering
        w = boxes[:, 3] - boxes[:, 1]
        h = boxes[:, 4] - boxes[:, 2]
        valid = (w > 0) & (h > 0)
        if valid.sum() == 0:
            raise RuntimeError("No valid boxes in batch!")

        boxes_class = (
            boxes[valid].type(feats[0].dtype).to(feats[0].device)
        )  # [N, num_of_detected, 4(xyxy)]
        boxes_grasp = expand_bbox_xyxy_tensor(
            boxes_class.type(feats[0].dtype).to(feats[0].device),
            scale=1.3,
            image_size=(640, 640),
        )

        rois = boxes_grasp
        rois_class = boxes_class
        crops_p3 = self.roi_align(feats[0], rois)  # [N, 576, 7, 7]
        crops_p5 = self.roi_align_class(feats[2], rois_class)
        crops = [crops_p3, crops_p5]
        return self.centroid_head(crops)  # box, angle, class_logits
