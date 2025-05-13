import math
import torch
import cv2
import numpy as np
from shapely.geometry import Polygon


def expand_bbox_xyxy_tensor(boxes, scale=1.2, image_size=(640, 640)):
    """
    Args:
        boxes (Tensor): shape (N, 4), format: (x1, y1, x2, y2)
        scale (float): expansion scale factor
        image_size (tuple): (W, H)

    Returns:
        expanded_boxes (Tensor): shape (N, 4), format: (x1, y1, x2, y2)
    """
    cls, x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    new_w = w * scale
    new_h = h * scale

    new_x1 = (cx - new_w / 2).clamp(min=0, max=image_size[0])
    new_y1 = (cy - new_h / 2).clamp(min=0, max=image_size[1])
    new_x2 = (cx + new_w / 2).clamp(min=0, max=image_size[0])
    new_y2 = (cy + new_h / 2).clamp(min=0, max=image_size[1])

    return torch.stack([cls, new_x1, new_y1, new_x2, new_y2], dim=-1)


def grasp_iou(pred_box, gt_box, img_size=640):
    """
    pred_box, gt_box: (cx, cy, w, h, sinθ), normalized 0~1
    """
    pred_cx, pred_cy, pred_w, pred_h, pred_sin_theta, pred_cos_theta = pred_box
    gt_cx, gt_cy, gt_w, gt_h, gt_sin_theta = gt_box

    pred_angle_rad = math.atan2(pred_sin_theta, pred_cos_theta)
    pred_theta = math.degrees(pred_angle_rad) % 360

    # pred_theta = math.degrees(math.asin(pred_sin_theta))
    gt_theta = math.degrees(math.asin(gt_sin_theta))

    pred_theta = (90 * (1 if pred_theta >= 0 else -1)) - pred_theta
    gt_theta = (90 * (1 if gt_theta >= 0 else -1)) - gt_theta

    pred_rect = (
        (pred_cx * img_size, pred_cy * img_size),
        (pred_w * img_size, pred_h * img_size),
        -pred_theta,
    )
    gt_rect = (
        (gt_cx * img_size, gt_cy * img_size),
        (gt_w * img_size, gt_h * img_size),
        -gt_theta,
    )

    pred_pts = cv2.boxPoints(pred_rect)
    gt_pts = cv2.boxPoints(gt_rect)

    pred_poly = Polygon(pred_pts)
    gt_poly = Polygon(gt_pts)

    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0

    inter = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area

    if union == 0:
        return 0.0
    return inter / union


def evaluate_grasp(pred_box, gt_box, img_size=640):
    """
    pred_box, gt_box: (cx, cy, w, h, sinθ)
    """
    iou = grasp_iou(pred_box, gt_box, img_size=img_size)
    pred_angle_rad = math.atan2(pred_box[-2], pred_box[-1])
    pred_theta = math.degrees(pred_angle_rad) % 360
    # pred_theta = math.degrees(math.asin(pred_box[-1]))
    gt_theta = math.degrees(math.asin(gt_box[-1]))

    angle_diff = abs(pred_theta - gt_theta)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return (iou > 0.25) and (angle_diff < 30)


def compute_metrics(pred_class_logits, pred_boxes, gt_classes, gt_boxes, img_size=640):
    """
    Args:
        pred_class_logits: [N, num_classes] (raw logits)
        pred_boxes: [N, 5] (normalized outputs)
        gt_classes: [N]
        gt_boxes: [N, 5]
    Returns:
        Cacc, Lacc, Dacc
    """
    pred_classes = pred_class_logits.argmax(dim=1)

    Cacc = 0
    Lacc = 0
    Dacc = 0

    total = pred_classes.shape[0]

    for i in range(total):
        correct_class = pred_classes[i].item() == gt_classes[i].item()
        correct_grasp = evaluate_grasp(
            pred_boxes[i].cpu().numpy(), gt_boxes[i].cpu().numpy(), img_size=img_size
        )

        if correct_class:
            Cacc += 1
        if correct_grasp:
            Lacc += 1
        if correct_class and correct_grasp:
            Dacc += 1

    Cacc /= total
    Lacc /= total
    Dacc /= total

    return Cacc, Lacc, Dacc
