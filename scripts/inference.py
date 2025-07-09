import cv2
import torch
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics.utils.ops import scale_boxes
from engine.trainer import LitGrasp
from models.seg_backbone import create_yolov8_model
from models.grasp_head_roi import GraspHeadROI

# --- 설정 ---
MODEL_CHECKPOINT = "checkpoints/20250709_171609/lightning_logs/version_0/checkpoints/epoch=016-val_Dacc=0.0882-best.ckpt"
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
    seg_model = create_yolov8_model("sg_15class_0429.pt")
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
    """프레임을 모델 입력에 맞게 전처리합니다."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 원본 비율 유지를 위한 패딩 추가 고려 (현재는 640x640으로 리사이즈)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    return img_tensor.permute(2, 0, 1).unsqueeze(0)


def visualize_results(frame, det_boxes, grasp_boxes, grasp_angles, class_logits):
    """추론 결과를 원본 프레임에 시각화합니다."""
    if det_boxes is None or len(det_boxes) == 0:
        return frame

    # 클래스 로짓에서 가장 높은 점수의 클래스 ID를 찾음
    class_ids = torch.argmax(class_logits, dim=1)

    for i in range(len(det_boxes)):
        # 1. Detection Bounding Box 그리기 (파란색)
        x1, y1, x2, y2 = map(int, det_boxes[i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 2. 클래스 이름 표시
        class_id = class_ids[i].item()
        class_name = CLASSES.get(class_id, "Unknown")
        label = f"{class_name}"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
        )

        # 3. Grasp Rectangle 그리기 (초록색)
        # Grasp 좌표를 BBox 기준으로 계산하는 원래 로직으로 복원
        cx_norm, cy_norm, w_norm, h_norm = grasp_boxes[i]

        box_w = x2 - x1
        box_h = y2 - y1
        cx = int(x1 + cx_norm * box_w)
        cy = int(y1 + cy_norm * box_h)
        w = int(w_norm * box_w)
        h = int(h_norm * box_h)

        # 각도 계산
        angle_rad = math.atan2(grasp_angles[i, 0], grasp_angles[i, 1])
        theta_deg = np.degrees(angle_rad)

        # 회전된 사각형 계산
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
            img_tensor = preprocess_frame(original_frame, DEVICE)

            with torch.no_grad():
                # forward()는 (det_boxes, [grasp_boxes, grasp_angles, class_logits])를 반환
                preds = model(img_tensor)

                if preds is not None:
                    det_boxes_raw, grasp_preds = preds
                    pred_grasp_box, pred_angle, pred_class = grasp_preds

                    # BBox 좌표를 원본 이미지 크기에 맞게 스케일링
                    # det_boxes_raw는 [batch_idx, x1, y1, x2, y2] 형태이므로, 좌표 부분만 사용
                    det_boxes_scaled = scale_boxes(
                        img_tensor.shape[2:],
                        det_boxes_raw[:, 1:5],
                        original_frame.shape[:2],
                    ).round()

                    # 시각화를 위해 모든 텐서를 CPU로 이동
                    det_boxes_cpu = det_boxes_scaled.cpu()
                    grasp_boxes_cpu = pred_grasp_box.cpu()
                    grasp_angles_cpu = pred_angle.cpu()
                    class_logits_cpu = pred_class.cpu()

                    frame_to_show = visualize_results(
                        original_frame,
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
