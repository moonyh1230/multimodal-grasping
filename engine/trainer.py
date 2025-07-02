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

    def forward(self, batch):
        seg_loss, feats = self.seg.custom_forward(batch["img"])
        pred_res, feats, preds = self.seg.custom_forward(imgs)

        boxes = batch["bboxes"]

        return self.grasp(feats[0], boxes)

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["cls"]

        backbone_losses, feats, backbone_bboxes = self.seg.custom_forward(batch)

        dtype = feats[0].dtype
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.seg.stride[0]
        )  # image size (h,w)

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
            grasps_gt = grasps_gt[
                ~torch.isin(
                    torch.arange(grasps_gt.size(0), device=self.device),
                    fails_embed_tensor.to("cuda"),
                )
            ]
            classes_gt = classes_gt[
                ~torch.isin(
                    torch.arange(classes_gt.size(0), device=self.device),
                    fails_embed_tensor.to("cuda"),
                )
            ]

        self.backbone_loss = backbone_losses[0].sum()
        self.loss_items = backbone_losses[1]

        loss_grasp_box = self.grasp_loss(pred_grasp_box, grasps_gt[:, :4])
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
        val_seg_loss, _ = self.seg.v8segloss(preds[1], batch)

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
            grasps_gt = grasps_gt[
                ~torch.isin(
                    torch.arange(grasps_gt.size(0), device=self.device),
                    fails_embed_tensor.to("cuda"),
                )
            ]
            classes_gt = classes_gt[
                ~torch.isin(
                    torch.arange(classes_gt.size(0), device=self.device),
                    fails_embed_tensor.to("cuda"),
                )
            ]

        loss_grasp_box = self.grasp_loss(pred_grasp_box, grasps_gt[:, :4])
        # loss_grasp_box = F.mse_loss(pred_grasp_box, grasps_gt[:, :4])
        loss_angle = angle_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = loss_grasp_box + self.alpha * loss_angle

        if self.freeze_seg:
            val_loss = self.beta * mseloss + loss_class
        else:
            val_loss = self.beta * mseloss + loss_class + val_backbone_loss * self.gamma

        pred_combined = torch.cat([pred_grasp_box, pred_angle], dim=1)  # (B, 6)
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
                f"Class: {self.classes_name[cls_id]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

            drawn_imgs.append(
                torch.tensor(img_1).permute(2, 0, 1) / 255.0
            )  # 다시 Tensor로 변환

        grid = vutils.make_grid(drawn_imgs, nrow=2)
        self.logger.experiment.add_image("Grasp Visualization", grid, self.global_step)
