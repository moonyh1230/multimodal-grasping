"""
YOLOv8-seg 인스턴스 세그멘테이션 백본 래퍼.
Ultralytics 가중치를 바로 불러온 뒤 inference() 메서드로
boxes, masks, img_indices 를 반환한다.
"""

from ultralytics import YOLO
import torch
import torch.nn as nn


class SegBackbone:
    def __init__(self, model_path="sg_15class_0429.pt", device="cuda"):
        self.model = YOLO(model_path)
        self.device = device

        # ✅ 안전하게 P5 feature map 추출용 submodule 구성
        # self.feature_extractor = (
        #     nn.Sequential(
        #         *list(self.model.model.model[:10])
        #     )  # P5 추출까지 /older: [:10]
        #     .to(device)
        #     .eval()
        # )
        cutting_model = self.model
        cutting_model.model.model = cutting_model.model.model[:-1]
        self.feature_extractor = cutting_model.to(device)

    @torch.no_grad()
    def __call__(self, imgs, target_classes=None, conf=0.3):
        """
        imgs  : List[np.ndarray] or torch.Tensor (B,H,W,3)
        return: boxes (N,4), idxs (N,), masks (N,H,W)
        """
        res = self.model(imgs, device=self.device, verbose=False, conf=conf)
        boxes, idxs, masks = [], [], []
        for i, r in enumerate(res):
            for j, c in enumerate(r.boxes.cls):
                if (target_classes is None) or (int(c.item()) in target_classes):
                    boxes.append(r.boxes.xyxy[j].cpu())
                    idxs.append(torch.tensor([i]))
                    masks.append(r.masks.data[j].cpu())
        if len(boxes):
            boxes = torch.stack(boxes)
            idxs = torch.cat(idxs)
        else:  # 빈 결과
            boxes = torch.zeros((0, 4))
            idxs = torch.zeros((0,), dtype=torch.long)
        return boxes, idxs, masks

    def extract_p5_feature(self, imgs):
        with torch.no_grad():
            feats = self.feature_extractor.model(
                imgs.to(self.device)
            )  # [B, 576, 20, 20] 예상
        return feats
