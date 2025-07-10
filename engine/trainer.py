import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from utils.metrics import compute_metrics, expand_bbox_xyxy_tensor
from torch.optim import AdamW
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from utils.loss import GraspBBoxLoss
import math
import cv2


def angle_loss(pred_angle, gt_angle_rad):
    target = torch.stack([torch.sin(gt_angle_rad), torch.cos(gt_angle_rad)], dim=-1)
    loss = F.mse_loss(pred_angle, target.squeeze(dim=1))
    return loss


class LitGrasp(pl.LightningModule):
    def __init__(
        self,
        seg,
        grasp,
        classes_name,
        lr=0.001,
        freeze_seg=False,
        img_size=640,
        optim="AdamW",
        scheduler="StepLR",
        visualize=True,
    ):
        super().__init__()
        self.seg = seg
        self.grasp = grasp
        self.classes_name = classes_name
        self.lr = lr
        self.freeze_seg = freeze_seg
        self.alpha = 2
        self.beta = 1
        self.gamma = 1
        self.val_outputs = []
        self.img_size = img_size
        self.optim = optim
        self.scheduler = scheduler
        self.visualize = visualize

        # Epoch level metrics
        self.backbone_loss = None
        self.seg_loss = None

        self.grasp_loss = GraspBBoxLoss(
            alpha=0,
            beta=1,
        )

        always_freeze_names = [".dfl"]
        for k, v in self.seg.named_parameters():
            if any(x in k for x in always_freeze_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                v.requires_grad = True

        if self.freeze_seg:
            for p in self.seg.parameters():
                p.requires_grad = False

        self.save_hyperparameters(ignore=["seg", "grasp", "classes_name"])

    def forward(self, imgs):
        pred_res, feats, _ = self.seg.custom_forward(imgs)
        boxes = []
        if pred_res[0].shape[0] > 0:
            for det_n, rt in enumerate(pred_res[0]):
                if rt.sum() != 0:
                    boxes.append(
                        torch.cat(
                            [
                                torch.zeros(1).to(self.device),
                                rt[:4],
                            ],
                            dim=0,
                        )
                    )

        if len(boxes) == 0:
            return None
        elif len(boxes) == 1:
            boxes = boxes[0].type(feats[0].dtype).to(self.device)[None, :]
        else:
            boxes = torch.stack(boxes).type(feats[0].dtype).to(self.device)

        pred_grasp_box, pred_angle, pred_class = self.grasp(feats, boxes)

        return boxes, [pred_grasp_box, pred_angle, pred_class]

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["cls"]

        backbone_losses, feats, backbone_bboxes = self.seg.custom_forward(batch)

        boxes = []
        batch_det_count = []
        fails = []
        for bn, rt in enumerate(backbone_bboxes):
            box = []
            for det_n, rt_n in enumerate(rt if rt.sum() != 0 else torch.zeros(1, 1)):
                if rt_n.sum() != 0:
                    box.append(
                        torch.cat(
                            [
                                (torch.ones(1) * bn).to(self.device),
                                rt_n[:4],
                            ],
                            dim=0,
                        )
                    )
                else:
                    box.append(
                        torch.cat(
                            [
                                (torch.ones(1) * bn).to(self.device),
                                torch.zeros(4).to(self.device),
                            ],
                            dim=0,
                        )
                    )
                    fails.append((bn, det_n))
            batch_det_count.append(det_n + 1)
            boxes.append(torch.cat(box, dim=0)[None, :])

        boxes = torch.cat(boxes, dim=0).type(feats[0].dtype).to(self.device)
        pred_grasp_box, pred_angle, pred_class = self.grasp(feats, boxes)

        fails_count = len(fails)
        if fails_count > 0:
            fails_embed = []
            for bn, det_n in fails:
                for i in range(batch_det_count[bn]):
                    if i == det_n:
                        fails_embed.append(sum(batch_det_count[:bn]) + det_n)

            fails_embed_tensor = torch.tensor(fails_embed, device=self.device)

            # Create a mask to keep successful detections
            keep_mask = ~torch.isin(
                torch.arange(grasps_gt.size(0), device=self.device),
                fails_embed_tensor.to("cuda"),
            )

            # Apply the mask to ground truth and boxes, predictions are already filtered
            grasps_gt = grasps_gt[keep_mask]
            classes_gt = classes_gt[keep_mask]
            boxes = boxes[keep_mask]

        self.backbone_loss = backbone_losses[0].sum()
        self.loss_items = backbone_losses[1]

        # Convert absolute GT grasps to be relative to the ROI boxes
        # Note: `boxes` here still contains failed detections, but they should be filtered out
        # by the existing `fails_count` logic before loss calculation.
        # We perform this conversion before filtering to align with the predictions.
        roi_boxes = boxes
        roi_x1, roi_y1, roi_x2, roi_y2 = (
            roi_boxes[:, 1],
            roi_boxes[:, 2],
            roi_boxes[:, 3],
            roi_boxes[:, 4],
        )
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        # Avoid division by zero
        epsilon = 1e-6
        roi_w = roi_w.clamp(min=epsilon)
        roi_h = roi_h.clamp(min=epsilon)

        gt_cx_abs, gt_cy_abs, gt_w_abs, gt_h_abs = (
            grasps_gt[:, 0] * self.img_size,
            grasps_gt[:, 1] * self.img_size,
            grasps_gt[:, 2] * self.img_size,
            grasps_gt[:, 3] * self.img_size,
        )

        gt_cx_rel = (gt_cx_abs - roi_x1) / roi_w
        gt_cy_rel = (gt_cy_abs - roi_y1) / roi_h
        gt_w_rel = gt_w_abs / roi_w
        gt_h_rel = gt_h_abs / roi_h

        grasps_gt_rel = torch.stack([gt_cx_rel, gt_cy_rel, gt_w_rel, gt_h_rel], dim=1)

        loss_grasp_box = self.grasp_loss(pred_grasp_box, grasps_gt_rel)
        # loss_grasp_box = F.mse_loss(pred_grasp_box, grasps_gt[:, :4])
        loss_angle = angle_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = loss_grasp_box + self.alpha * loss_angle

        if self.freeze_seg:
            total_loss = self.beta * mseloss + loss_class
        else:
            total_loss = (
                self.beta * mseloss + loss_class + self.backbone_loss * self.gamma
            )

        self.log("train_loss_box", loss_grasp_box)
        self.log("train_loss_angle", loss_angle)
        self.log("train_loss_class", loss_class)
        self.log("train_loss", total_loss)

        if not self.freeze_seg:
            self.log_dict(
                {
                    "train_loss_backbone_tot": self.backbone_loss,
                    "train_loss_bbox": self.loss_items[0],
                    "train_loss_seg": self.loss_items[1],
                    "train_loss_cls": self.loss_items[2],
                    "train_loss_dfl": self.loss_items[3],
                }
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["cls"]
        batch_num = batch["batch_idx"]
        total_boxes = batch_num.shape[0]

        pred_res, feats, preds = self.seg.custom_forward(imgs)
        val_seg_loss = self.seg.v8segloss(preds[1], batch)

        val_seg_loss_item = val_seg_loss[1]
        val_backbone_loss = val_seg_loss[0].sum()

        dtype = feats[0].dtype
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.seg.stride[0]
        )  # image size (h,w)

        scale_tensor = imgsz[[1, 0, 1, 0]]

        boxes = []
        batch_det_count = []
        fails = []
        for bn, rt in enumerate(pred_res):
            box = []
            for det_n, rt_n in enumerate(rt if rt.sum() != 0 else torch.zeros(1, 1)):
                if rt_n.sum() != 0:
                    box.append(
                        torch.cat(
                            [
                                (torch.ones(1) * bn).to(self.device),
                                rt_n[:4],
                            ],
                            dim=0,
                        )
                    )
                else:
                    box.append(
                        torch.cat(
                            [
                                (torch.ones(1) * bn).to(self.device),
                                torch.zeros(4).to(self.device),
                            ],
                            dim=0,
                        )
                    )
                    fails.append((bn, det_n))
            batch_det_count.append(det_n + 1)
            boxes.append(torch.cat(box, dim=0)[None, :])

        boxes = torch.cat(boxes, dim=0).type(feats[0].dtype).to(self.device)
        pred_grasp_box, pred_angle, pred_class = self.grasp(feats, boxes)

        fails_count = len(fails)
        if fails_count > 0:
            fails_embed = []
            for bn, det_n in fails:
                for i in range(batch_det_count[bn]):
                    if i == det_n:
                        fails_embed.append(sum(batch_det_count[:bn]) + det_n)

            fails_embed_tensor = torch.tensor(fails_embed, device=self.device)

            # Create a mask to keep successful detections
            keep_mask = ~torch.isin(
                torch.arange(grasps_gt.size(0), device=self.device),
                fails_embed_tensor.to("cuda"),
            )

            # Apply the mask to ground truth and boxes, predictions are already filtered
            grasps_gt = grasps_gt[keep_mask]
            classes_gt = classes_gt[keep_mask]
            boxes = boxes[keep_mask]

        # Convert absolute GT grasps to be relative to the ROI boxes
        roi_boxes = boxes
        roi_x1, roi_y1, roi_x2, roi_y2 = (
            roi_boxes[:, 1],
            roi_boxes[:, 2],
            roi_boxes[:, 3],
            roi_boxes[:, 4],
        )
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        epsilon = 1e-6
        roi_w = roi_w.clamp(min=epsilon)
        roi_h = roi_h.clamp(min=epsilon)

        gt_cx_abs, gt_cy_abs, gt_w_abs, gt_h_abs = (
            grasps_gt[:, 0] * self.img_size,
            grasps_gt[:, 1] * self.img_size,
            grasps_gt[:, 2] * self.img_size,
            grasps_gt[:, 3] * self.img_size,
        )

        gt_cx_rel = (gt_cx_abs - roi_x1) / roi_w
        gt_cy_rel = (gt_cy_abs - roi_y1) / roi_h
        gt_w_rel = gt_w_abs / roi_w
        gt_h_rel = gt_h_abs / roi_h

        grasps_gt_rel = torch.stack([gt_cx_rel, gt_cy_rel, gt_w_rel, gt_h_rel], dim=1)

        loss_grasp_box = self.grasp_loss(pred_grasp_box, grasps_gt_rel)
        loss_angle = angle_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = loss_grasp_box + self.alpha * loss_angle

        if self.freeze_seg:
            val_loss = self.beta * mseloss + loss_class
        else:
            val_loss = self.beta * mseloss + loss_class + val_backbone_loss * self.gamma

        # For metrics, we need to convert predicted relative grasps back to absolute
        pred_cx_rel, pred_cy_rel, pred_w_rel, pred_h_rel = (
            pred_grasp_box[:, 0],
            pred_grasp_box[:, 1],
            pred_grasp_box[:, 2],
            pred_grasp_box[:, 3],
        )
        pred_cx_abs = pred_cx_rel * roi_w + roi_x1
        pred_cy_abs = pred_cy_rel * roi_h + roi_y1
        pred_w_abs = pred_w_rel * roi_w
        pred_h_abs = pred_h_rel * roi_h
        pred_grasp_box_abs = torch.stack(
            [pred_cx_abs, pred_cy_abs, pred_w_abs, pred_h_abs], dim=1
        )

        pred_combined = torch.cat([pred_grasp_box_abs, pred_angle], dim=1)
        gt_combined = torch.cat([grasps_gt[:, :4], grasps_gt[:, 4:5]], dim=1)

        self.log("val_loss_box", loss_grasp_box, on_step=False, on_epoch=True)
        self.log("val_loss_angle", loss_angle, on_step=False, on_epoch=True)
        self.log("val_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

        if not self.freeze_seg:
            self.log_dict(
                {
                    "val_loss_backbone_tot": val_backbone_loss,
                    "val_loss_bbox": val_seg_loss_item[0],
                    "val_loss_seg": val_seg_loss_item[1],
                    "val_loss_cls": val_seg_loss_item[2],
                    "val_loss_dfl": val_seg_loss_item[3],
                },
                on_step=False,
                on_epoch=True,
            )

        self.val_outputs.append(
            {
                "pred_class": pred_class.detach().cpu(),
                "pred_box": pred_combined.detach().cpu(),
                "gt_class": classes_gt.detach().cpu(),
                "gt_box": gt_combined.detach().cpu(),
                "total_boxes": total_boxes,
                "failed_boxes": fails_count,
            }
        )

        if batch_idx % 5 == 0 and self.visualize:
            if total_boxes - fails_count > 3:
                self.visualize_grasp(
                    imgs, pred_grasp_box, pred_angle, pred_class, batch_idx, boxes
                )

        return val_loss

    def on_validation_epoch_end(self):
        pred_classes = torch.cat([o["pred_class"] for o in self.val_outputs], dim=0)
        pred_boxes = torch.cat([o["pred_box"] for o in self.val_outputs], dim=0)
        gt_classes = torch.cat([o["gt_class"] for o in self.val_outputs], dim=0)
        gt_boxes = torch.cat([o["gt_box"] for o in self.val_outputs], dim=0)
        total_boxes = sum([o["total_boxes"] for o in self.val_outputs])
        fails_count = sum([o["failed_boxes"] for o in self.val_outputs])

        Cacc, Lacc, Dacc, suc_rate = compute_metrics(
            pred_classes, pred_boxes, gt_classes, gt_boxes, total_boxes, fails_count
        )

        self.log("val_Cacc", Cacc, prog_bar=False)
        self.log("val_Lacc", Lacc, prog_bar=False)
        self.log("val_Dacc", Dacc, prog_bar=False)
        self.log("val_suc_rate", suc_rate, prog_bar=False)

        self.val_outputs.clear()

    def configure_optimizers(self):
        batch_num = self.trainer.num_training_batches
        n_epochs = self.trainer.max_epochs

        num_warmup_steps = batch_num * 2
        num_total_steps = batch_num * n_epochs

        optimizer = (
            torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
            )
            if self.optim == "AdamW"
            else torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                momentum=0.9,
            )
        )
        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5, last_epoch=-1
            )
            if self.scheduler == "StepLR"
            else torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=0.00001
            )
            if self.scheduler == "CALR"
            else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=20,
                T_mult=1,
                eta_min=0.00001,
            )
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def visualize_grasp(self, imgs, pred_box, pred_angle, pred_class, batch_idx, boxes):
        imgs = imgs.detach().cpu()
        pred_box = pred_box.detach().cpu()
        pred_angle = pred_angle.detach().cpu()
        pred_class = pred_class.argmax(dim=1).detach().cpu()
        boxes = boxes.detach().cpu()

        drawn_imgs = []
        # Ensure we have predictions to visualize
        num_preds_to_viz = min(pred_box.size(0), 4)

        for i in range(num_preds_to_viz):
            # Find the original image in the batch this prediction corresponds to
            batch_idx_for_pred = boxes[i, 0].int().item()
            img = imgs[batch_idx_for_pred]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (
                (img * 255).astype(np.uint8).copy()
            )  # Use copy to avoid issues with read-only data

            # Draw detection box
            x1, y1, x2, y2 = boxes[i, 1:].int().tolist()
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Get relative grasp prediction
            cx_rel, cy_rel, w_rel, h_rel = pred_box[i]

            # Convert relative grasp to absolute coordinates based on its ROI
            box_w = x2 - x1
            box_h = y2 - y1
            cx_abs = int(x1 + cx_rel * box_w)
            cy_abs = int(y1 + cy_rel * box_h)
            w_abs = int(w_rel * box_w)
            h_abs = int(h_rel * box_h)

            # Get angle
            pred_angle_rad = torch.atan2(pred_angle[i, 0], pred_angle[i, 1])
            theta = torch.rad2deg(pred_angle_rad).item()

            # Draw grasp rectangle
            rect = cv2.boxPoints(((cx_abs, cy_abs), (w_abs, h_abs), theta))
            rect = np.int0(rect)
            img = cv2.drawContours(img, [rect], -1, (0, 255, 0), 2)

            # Put class text
            cls_id = pred_class[i].item()
            cv2.putText(
                img,
                f"Class: {self.classes_name.get(cls_id, 'Unknown')}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

            drawn_imgs.append(torch.tensor(img).permute(2, 0, 1) / 255.0)

        if drawn_imgs:
            grid = vutils.make_grid(drawn_imgs, nrow=2)
            self.logger.experiment.add_image(
                "Grasp Visualization", grid, self.global_step
            )
