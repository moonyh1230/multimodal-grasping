import cv2
import torch
import math
import numpy as np
import pyrealsense2 as rs
from engine.trainer import LitGrasp
from models.seg_backbone import create_yolov8_model
from models.grasp_head_roi import GraspHeadROI

# --- 설정 ---
MODEL_CHECKPOINT = "checkpoints/20250709_220531/lightning_logs/version_0/checkpoints/epoch=164-val_Dacc=0.9454-best.ckpt"
# MODEL_CHECKPOINT = "checkpoints/20250710_105140_stage_2/lightning_logs/version_0/checkpoints/epoch=010-val_Dacc=0.9076-best.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 640

# --- 클래스 이름 정의 ---
CLASSES = {
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


def load_model(checkpoint_path):
    """학습된 LitGrasp 모델을 로드합니다."""
    yolo_feature_specs = {
        "p3": (192, 80),
        "p4": (384, 40),
        "p5": (576, 20),
    }
    seg_model = create_yolov8_model("sg_15class_0429.pt", is_inference=True)
    grasp_head = GraspHeadROI(
        grasp_feat_spec=yolo_feature_specs["p3"],
        class_feat_spec=yolo_feature_specs["p5"],
        num_classes=len(CLASSES),
    )
    model = LitGrasp.load_from_checkpoint(
        checkpoint_path,
        seg=seg_model,
        grasp=grasp_head,
        classes_name=CLASSES,
        map_location=DEVICE,
    )
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model


def preprocess_frame(frame, device):
    """프레임을 모델 입력에 맞게 전처리하고, 복원을 위한 정보를 반환합니다."""
    original_h, original_w, _ = frame.shape
    target_size = IMG_SIZE

    # 비율 계산
    ratio = min(target_size / original_w, target_size / original_h)
    new_w, new_h = int(original_w * ratio), int(original_h * ratio)

    # 리사이즈
    resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 패딩 계산
    pad_w, pad_h = (target_size - new_w) // 2, (target_size - new_h) // 2

    # 패딩 추가
    padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_img

    # 텐서로 변환
    img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0

    return img_tensor.permute(2, 0, 1).unsqueeze(0), ratio, pad_w, pad_h


def visualize_results(frame, det_boxes, grasp_boxes, grasp_angles, class_logits):
    """추론 결과를 원본 프레임에 시각화합니다."""
    if det_boxes is None or len(det_boxes) == 0:
        return frame

    class_ids = torch.argmax(class_logits, dim=1)

    for i in range(len(det_boxes)):
        x1, y1, x2, y2 = map(int, det_boxes[i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        class_id = class_ids[i].item()
        class_name = CLASSES.get(class_id, "Unknown")
        label = f"{class_name}"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
        )

        cx_norm, cy_norm, w_norm, h_norm = grasp_boxes[i]
        box_w, box_h = x2 - x1, y2 - y1
        cx = int(x1 + cx_norm * box_w)
        cy = int(y1 + cy_norm * box_h)
        w = int(w_norm * box_w)
        h = int(h_norm * box_h)

        angle_rad = math.atan2(grasp_angles[i, 0], grasp_angles[i, 1])
        theta_deg = np.degrees(angle_rad)

        rect = ((cx, cy), (w, h), theta_deg)
        box_points = cv2.boxPoints(rect)
        cv2.drawContours(frame, [np.intp(box_points)], 0, (0, 255, 0), 2)

    return frame


def main():
    model = load_model(MODEL_CHECKPOINT)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("Starting RealSense stream...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            original_frame = np.asanyarray(color_frame.get_data())
            img_tensor, ratio, pad_w, pad_h = preprocess_frame(original_frame, DEVICE)

            with torch.no_grad():
                preds = model(img_tensor)

                if preds is not None:
                    det_boxes_raw, grasp_preds = preds
                    pred_grasp_box, pred_angle, pred_class = grasp_preds

                    # 좌표 복원
                    boxes = det_boxes_raw[:, 1:5]
                    boxes[:, [0, 2]] -= pad_w  # x좌표에서 패딩 제거
                    boxes[:, [1, 3]] -= pad_h  # y좌표에서 패딩 제거
                    boxes[:, :] /= ratio  # 비율로 스케일링

                    # 원본 이미지 경계에 맞게 좌표 클리핑
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
                        0, original_frame.shape[1]
                    )
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
                        0, original_frame.shape[0]
                    )

                    det_boxes_scaled = boxes.round()

                    det_boxes_cpu = det_boxes_scaled.cpu()
                    grasp_boxes_cpu = pred_grasp_box.cpu()
                    grasp_angles_cpu = pred_angle.cpu()
                    class_logits_cpu = pred_class.cpu()

                    frame_to_show = visualize_results(
                        original_frame.copy(),  # 원본을 복사해서 사용
                        det_boxes_cpu,
                        grasp_boxes_cpu,
                        grasp_angles_cpu,
                        class_logits_cpu,
                    )
                else:
                    frame_to_show = original_frame

            cv2.imshow("RealSense Inference", frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Stopping RealSense stream.")
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
