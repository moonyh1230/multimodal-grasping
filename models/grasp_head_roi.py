import torch
import torch.nn as nn
import torchvision.ops as ops

from utils.metrics import expand_bbox_xyxy_tensor


class GraspAndClassHead(nn.Module):
    """
    RoIAlign으로 추출된 피처맵으로부터 Grasp과 Class를 예측하는 헤드.
    공간 정보를 최대한 보존하기 위해 Conv 레이어를 먼저 통과한 후,
    AdaptiveAvgPool2d를 사용하여 최종 예측을 수행.
    """
    def __init__(self, grasp_in_channels=192, class_in_channels=576, num_classes=15):
        super().__init__()

        # --- Grasp 예측 브랜치 ---
        self.grasp_conv = nn.Sequential(
            nn.Conv2d(grasp_in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.grasp_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.grasp_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
        )
        self.grasp_box_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(256, 4),  # (cx, cy, w, h)
        )
        self.grasp_angle_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),  # sin(theta), cos(theta)
        )

        # --- Class 예측 브랜치 ---
        self.class_conv = nn.Sequential(
            nn.Conv2d(class_in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.class_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, crops):
        grasp_crop, class_crop = crops

        # Grasp 예측
        g_feat = self.grasp_conv(grasp_crop)
        g_feat = self.grasp_pool(g_feat)
        g_feat = torch.flatten(g_feat, 1)
        g_feat = self.grasp_fc(g_feat)

        box = torch.sigmoid(self.grasp_box_branch(g_feat))
        angle = torch.tanh(self.grasp_angle_branch(g_feat))

        # Class 예측
        c_feat = self.class_conv(class_crop)
        c_feat = self.class_pool(c_feat)
        c_feat = torch.flatten(c_feat, 1)
        class_logits = self.class_fc(c_feat)

        return box, angle, class_logits


class GraspHeadROI(nn.Module):
    def __init__(
        self,
        grasp_feat_spec: tuple[int, int],  # (channels, size) for grasp features (e.g., feats[0])
        class_feat_spec: tuple[int, int],  # (channels, size) for class features (e.g., feats[2])
        num_classes=15,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        grasp_in_channels, grasp_feat_size = grasp_feat_spec
        class_in_channels, class_feat_size = class_feat_spec

        # RoIAlign 해상도를 14x14로 증가시켜 공간적 정밀도 향상
        self.roi_align = ops.RoIAlign(
            output_size=(14, 14), spatial_scale=grasp_feat_size / 640, sampling_ratio=2
        )
        self.roi_align_class = ops.RoIAlign(
            output_size=(14, 14), spatial_scale=class_feat_size / 640, sampling_ratio=2
        )

        # 새로운 Head 구조로 초기화
        self.centroid_head = GraspAndClassHead(
            grasp_in_channels=grasp_in_channels,
            class_in_channels=class_in_channels,
            num_classes=num_classes,
        )

    def forward(self, feats, boxes):
        # box size filtering
        w = boxes[:, 3] - boxes[:, 1]
        h = boxes[:, 4] - boxes[:, 2]
        valid = (w > 0) & (h > 0)
        if valid.sum() == 0:
            # 유효한 박스가 없는 경우, 빈 텐서를 반환하거나 에러 처리
            # 여기서는 간단히 빈 텐서들을 반환하여 이후 로직에서 처리되도록 함
            return torch.empty(0, 4), torch.empty(0, 2), torch.empty(0, 15)

        # 유효한 박스만 필터링
        valid_boxes = boxes[valid]

        # RoI Align을 위한 box 형식 [batch_idx, x1, y1, x2, y2]
        # LitGrasp에서 넘어온 boxes는 이미 [batch_idx, x1, y1, x2, y2] 형식이므로 그대로 사용

        boxes_class = valid_boxes.type(feats[0].dtype).to(feats[0].device)
        boxes_grasp = expand_bbox_xyxy_tensor(
            boxes_class.clone(), # clone() to avoid in-place modification issues
            scale=1.3,
            image_size=(640, 640),
        )

        # RoIAlign 수행
        crops_p3 = self.roi_align(feats[0], boxes_grasp)
        crops_p5 = self.roi_align_class(feats[2], boxes_class)

        crops = [crops_p3, crops_p5]
        return self.centroid_head(crops)
