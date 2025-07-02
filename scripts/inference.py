import torch
import cv2
import numpy as np
from torchvision import transforms
from models.seg_backbone import create_yolov8_model
from models.grasp_head_roi import GraspHeadROI
from engine.trainer import LitGrasp


# ---------------------------
# 1. 모델 로드 함수
# ---------------------------
# def load_model(checkpoint_path, seg_model_path):
#     seg = SegBackbone(model_path=seg_model_path)
#     grasp = GraspHeadROI(in_channels=576, num_classes=15)

#     model = LitGrasp.load_from_checkpoint(
#         checkpoint_path=checkpoint_path,
#         seg=seg,
#         grasp=grasp,
#         lr=1e-4,
#         freeze_seg=False,
#     )
#     model.eval()
#     model.cuda()
#     return model


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
def visualize_grasp(imgs, pred_box, pred_angle, pred_class, boxes, classes_name):
    imgs = imgs.detach().cpu()
    pred_box = pred_box.detach().cpu()
    pred_angle = pred_angle.detach().cpu()
    pred_class = pred_class.argmax(dim=1).detach().cpu()
    boxes = boxes.detach().cpu()

    drawn_imgs = []
    for i in range(min(4, imgs.size(0))):  # 최대 4개까지만
        img = imgs[i]
        img = img.permute(1, 2, 0).cpu().numpy()  # CHW → HWC + NumPy 변환
        img = (img * 255).astype(np.uint8)
        _, x1, y1, x2, y2 = boxes[i].int().tolist()
        img = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (255, 0, 0), 2)

        cx, cy, w, h = pred_box[i]

        pred_angle_rad = torch.atan2(pred_angle[i, 0], pred_angle[i, 1])
        theta = torch.rad2deg(pred_angle_rad) % 360
        # theta = math.degrees(math.asin(pred_angle[i].item()))

        flg = 1
        if theta < 0:
            flg = -1
        theta = 90 * flg - theta

        cx = int(cx * 640)
        cy = int(cy * 640)
        w = int(w * 640)
        h = int(h * 640)

        rect = cv2.boxPoints(((cx, cy), (w, h), theta.item()))
        rect = np.int0(rect)
        img_1 = cv2.drawContours(img.copy(), [rect], -1, (0, 255, 0), 2)

        cls_id = pred_class[i].item()
        cv2.putText(
            img_1,
            f"Class: {classes_name[cls_id]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )

    return img_1


# ---------------------------
# 5. Main 실행
# ---------------------------
if __name__ == "__main__":
    seg_model_path = "sg_15class_0429.pt"
    img_path = "img_test/output_2024-04-29_12-57-25_mp4-0092_jpg.rf.ce83209b018fd12d20b8b17a0d32aeb7.jpg"
    torch.cuda.set_device(0)

    feat_dim = [20, 40, 80]  # YOLOv8m-seg에서 사용한 feature map 크기
    feat_ch = [576, 384, 192]  # YOLOv8m-seg에서 사용한 feature map 채널 수
    feat_args = 2

    seg = create_yolov8_model("sg_15class_0429.pt")  # YOLOv8m-seg fine-tuned 모델

    grasp = GraspHeadROI(
        in_channels=feat_ch[feat_args], num_classes=15, feat_size=feat_dim[feat_args]
    )

    classes = {
        0: "clamp_Aillis",
        1: "clamp_kelly",
        2: "clamp_mosqutio",
        3: "clamp_sponge",
        4: "forceps_long",
        5: "forceps_wide",
        6: "needle_holder_14",
        7: "needle_holder_20",
        8: "punch",
        9: "retractor_army",
        10: "retractor_senn_b",
        11: "retractor_senn_s",
        12: "scissor_mayo",
        13: "scissor_metzenbaum",
        14: "scissor_operating",
    }

    lit = LitGrasp(
        seg=seg,
        grasp=grasp,
        classes_name=classes,
        freeze_seg=False,
        visualize=False,
    )

    lit.load_state_dict(
        torch.load(
            "checkpoints/20250616_165546_stage_2/lightning_logs/version_0/checkpoints/epoch=014-val_Dacc=0.3992-best.ckpt",
        )["state_dict"],
    )
    lit.eval()

    img_orig, img_tensor = preprocess(img_path)

    boxes, pred_box, pred_angle, pred_class = inference(lit, img_tensor)

    vis_img = visualize_grasp(
        img_orig, boxes, pred_box, pred_angle, pred_class, classes
    )

    cv2.imshow("Grasp Detection Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
