import torch
import cv2
import numpy as np
from torchvision import transforms
from models.seg_backbone import SegBackbone
from models.grasp_head_roi import GraspHeadROI
from engine.trainer import LitGrasp


# ---------------------------
# 1. 모델 로드 함수
# ---------------------------
def load_model(checkpoint_path, seg_model_path):
    seg = SegBackbone(model_path=seg_model_path)
    grasp = GraspHeadROI(in_channels=576, num_classes=15)

    model = LitGrasp.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        seg=seg,
        grasp=grasp,
        lr=1e-4,
        freeze_seg=False,
    )
    model.eval()
    model.cuda()
    return model


# ---------------------------
# 2. 이미지 로드 및 전처리
# ---------------------------
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()  # [1, 3, 640, 640]
    return img, img_tensor


# ---------------------------
# 3. Inference
# ---------------------------
def inference(model, img_tensor):
    with torch.no_grad():
        boxes, idxs, _ = model.seg(img_tensor)
        feats = model.seg.extract_p5_feature(img_tensor)
        pred_box, pred_angle, pred_class = model.grasp(feats, boxes, idxs)

    return boxes, pred_box, pred_angle, pred_class


# ---------------------------
# 4. 결과 시각화
# ---------------------------
def visualize(img, boxes, pred_box, pred_angle, pred_class):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred_class = pred_class.argmax(dim=1)

    for i in range(boxes.shape[0]):
        cx, cy, w, h = pred_box[i]
        theta = pred_angle[i].item() * 90.0  # [-1, 1] → [-90°, 90°]
        cls = pred_class[i].item()

        # 복원
        cx = int(cx * 640)
        cy = int(cy * 640)
        w = int(w * 640)
        h = int(h * 640)

        rect = cv2.boxPoints(((cx, cy), (w, h), theta))
        rect = np.int0(rect)
        img = cv2.drawContours(img, [rect], 0, (0, 255, 0), 2)

        cv2.putText(
            img,
            f"Class: {cls}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    return img


# ---------------------------
# 5. Main 실행
# ---------------------------
if __name__ == "__main__":
    checkpoint_path = "checkpoints/20250429_133335/epoch=021-val_loss=2.9478-best.ckpt"
    seg_model_path = "sg_15class_0429.pt"
    img_path = "img_test/output_2024-04-29_12-57-25_mp4-0092_jpg.rf.ce83209b018fd12d20b8b17a0d32aeb7.jpg"

    model = load_model(checkpoint_path, seg_model_path)
    img_orig, img_tensor = preprocess(img_path)

    boxes, pred_box, pred_angle, pred_class = inference(model, img_tensor)

    vis_img = visualize(img_orig, boxes, pred_box, pred_angle, pred_class)

    cv2.imshow("Grasp Detection Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
