"""
YOLOv8-seg 인스턴스 세그멘테이션 백본 래퍼.
Ultralytics 가중치를 바로 불러온 뒤 inference() 메서드로
boxes, masks, img_indices 를 반환한다.
"""

from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel, SegmentationModel


class YOLOv8DetectionAndFeatureExtractorModel(SegmentationModel):
    def __init__(
        self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True
    ):  # model, input channels, number of classes
        super().__init__(cfg, ch, nc, verbose)

    def custom_forward(self, x, target_classes=None):
        """
        This is a modified version of the original _forward_once() method in BaseModel,
        found in ultralytics/nn/tasks.py.
        The original method returns only the detection output, while this method returns
        both the detection output and the features extracted by the last convolutional layer.
        """
        y, feature_maps = [], []  # outputs

        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            if i == 22:
                m.eval()

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in the original YOLOV8 implementation.
            """
            if i in {
                # 15,
                # 18,
                21,
            }:  # if current layer is one of the feature extraction layers
                feature_maps.append(x)

        return x[0], feature_maps


def create_yolov8_model(model_name_or_path, nc, class_names):
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.cfg import get_cfg

    ckpt = None
    if str(model_name_or_path).endswith(".pt"):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt["model"].yaml
    else:
        cfg = model_name_or_path
    model = YOLOv8DetectionAndFeatureExtractorModel(cfg, nc=nc, verbose=False)
    if weights:
        model.load(weights)
    model.nc = nc
    model.names = class_names  # attach class names to model
    args = get_cfg(overrides={"model": model_name_or_path})
    model.args = args  # attach hyperparameters to model
    return model


class SegBackbone(nn.Module):
    def __init__(self, model_path="sg_15class_0429.pt", device="cuda"):
        super().__init__()
        seg_model = YOLO(model_path)
        self.model = seg_model.model.model
        self.device = device

    def forward(self, x, target_classes=None, embed=False):
        y, feature_maps = [], []  # outputs
        boxes, idxs, masks = [], [], []

        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in the original YOLOV8 implementation.
            """
            if embed:
                if i in {
                    # 15,
                    # 18,
                    21,
                }:  # if current layer is one of the feature extraction layers
                    feature_maps.append(x)

        if embed:
            return feature_maps
        else:
            for i, r in enumerate(x):
                for j, c in enumerate(r.boxes.cls):
                    if (target_classes is None) or (int(c.item()) in target_classes):
                        boxes.append(r.boxes.xyxy[j].cpu())
                        idxs.append(torch.tensor([i]))

            if len(boxes):
                boxes = torch.stack(boxes)
                idxs = torch.cat(idxs)
            else:  # 빈 결과
                boxes = torch.zeros((0, 4))
                idxs = torch.zeros((0,), dtype=torch.long)

            return boxes, idxs
