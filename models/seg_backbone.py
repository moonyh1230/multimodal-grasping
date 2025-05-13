"""
YOLOv8-seg 인스턴스 세그멘테이션 백본 래퍼.
Ultralytics 가중치를 바로 불러온 뒤 inference() 메서드로
boxes, masks, img_indices 를 반환한다.
"""

from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics.utils import ops
from utils.loss import v8SegmentationLoss


class YOLOv8DetectionAndFeatureExtractorModel(SegmentationModel):
    def __init__(
        self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=False, imgsz=(640, 640)
    ):  # model, input channels, number of classes
        super().__init__(cfg, ch, nc, verbose)
        self.v8segloss = None
        self.imgsz = imgsz

    def postprocess(self, pred, max_det=1):
        preds = ops.non_max_suppression(
            pred,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            # max_det=self.args.max_det,
            max_det=max_det,
            nc=len(self.names),
            end2end=getattr(self, "end2end", False),
            rotated=self.args.task == "obb",
        )
        return preds

    def custom_forward(self, x):
        if isinstance(x, dict):
            pred, feats = self._custom_forward(x["img"])
            return self.v8segloss(pred, x), feats
        else:
            pred, feats = self._custom_forward(x)
            return self.postprocess(pred[0]), feats, pred

    def _custom_forward(self, x):
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
                15,
                18,
                21,
            }:  # if current layer is one of the feature extraction layers
                feature_maps.append(x)

        return x, feature_maps


def create_yolov8_model(model_name_or_path):
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.cfg import get_cfg

    ckpt = None
    if str(model_name_or_path).endswith(".pt"):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt["model"].yaml
    else:
        cfg = model_name_or_path

    model = YOLOv8DetectionAndFeatureExtractorModel(
        cfg, verbose=False, ch=cfg["ch"], nc=cfg["nc"]
    )

    if weights:
        model.load(weights)
    if cfg:
        model.cfg = cfg

    args = get_cfg(
        overrides={
            "model": model_name_or_path,
            "conf": 0.20,
            "iou": 0.30,
            "save": False,
            "rect": True,
            "max_det": 20,
            "nms": True,
        }
    )
    model.args = args  # attach hyperparameters to model
    if torch.cuda.is_available():
        model.to("cuda")

    model.v8segloss = v8SegmentationLoss(model)

    # custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True}
    # pred_args = {**model.overrides, **custom}
    # model.predictor = model._smart_load("predictor")(
    #     overrides=pred_args, _callbacks=model.callbacks
    # )
    model.train()
    return model
