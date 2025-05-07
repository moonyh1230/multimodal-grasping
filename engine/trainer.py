import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from utils.metrics import compute_metrics
from torch.optim import AdamW
from ultralytics.utils.ops import non_max_suppression
import math
import cv2


class LitGrasp(pl.LightningModule):
    def __init__(
        self,
        seg,
        grasp,
        classes_name,
        lr=0.0001,
        freeze_seg=False,
        unfreeze_at_epoch=20,
        img_size=640,
    ):
        super().__init__()
        self.seg = seg
        self.grasp = grasp
        self.classes_name = classes_name
        self.lr = lr
        self.freeze_seg = freeze_seg
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.alpha = 10
        self.beta = 10
        self.val_outputs = []
        self.img_size = img_size
        self.save_hyperparameters(ignore=["seg", "grasp"])

        if self.freeze_seg:
            for p in self.seg.model.parameters():
                p.requires_grad = False

    def forward(self, imgs):
        seg_res, feats = self.seg.custom_forward(imgs)
        pred_res = non_max_suppression(
            seg_res, conf_thres=0.05, iou_thres=0.1, max_det=1, nc=15
        )

        boxes = [rt[:, :4] for rt in pred_res]

        return self.grasp(feats[0], boxes)

    def training_step(self, batch, batch_idx):
        imgs = batch["image"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["classes"]

        seg_res, feats = self.seg.custom_forward(imgs)
        pred_res = non_max_suppression(
            seg_res, conf_thres=0.05, iou_thres=0.1, max_det=1, nc=15
        )

        boxes = [rt[:, :4] for rt in pred_res]
        pred_grasp_box, pred_angle, pred_class = self.grasp(feats[0], boxes)

        loss_grasp_box = F.mse_loss(pred_grasp_box, grasps_gt[:, :4])
        loss_angle = F.mse_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = self.alpha * loss_grasp_box + loss_angle

        total_loss = self.beta * mseloss + loss_class

        self.log("loss_grasp_box", loss_grasp_box)
        self.log("loss_angle", loss_angle)
        self.log("loss_class", loss_class)
        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs = batch["image"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["classes"]

        seg_res, feats = self.seg.custom_forward(imgs)
        pred_res = non_max_suppression(
            seg_res, conf_thres=0.05, iou_thres=0.1, max_det=1, nc=15
        )

        boxes = [rt[:, :4] for rt in pred_res]
        pred_grasp_box, pred_angle, pred_class = self.grasp(feats[0], boxes)

        loss_grasp_box = F.mse_loss(pred_grasp_box, grasps_gt[:, :4])
        loss_angle = F.mse_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = self.alpha * loss_grasp_box + loss_angle

        val_loss = self.beta * mseloss + loss_class

        pred_combined = torch.cat([pred_grasp_box, pred_angle], dim=1)  # (B, 5)
        gt_combined = torch.cat([grasps_gt[:, :4], grasps_gt[:, 4:5]], dim=1)

        self.log("val_loss_box", loss_grasp_box)
        self.log("val_loss_angle", loss_angle)
        self.log("val_loss_class", loss_class)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

        self.val_outputs.append(
            {
                "pred_class": pred_class.detach().cpu(),
                "pred_box": pred_combined.detach().cpu(),
                "gt_class": classes_gt.detach().cpu(),
                "gt_box": gt_combined.detach().cpu(),
            }
        )

        if batch_idx % 50 == 0:
            self.visualize_grasp(
                imgs, pred_grasp_box, pred_angle, pred_class, batch_idx
            )

        return val_loss

    def on_validation_epoch_end(self):
        pred_classes = torch.cat([o["pred_class"] for o in self.val_outputs], dim=0)
        pred_boxes = torch.cat([o["pred_box"] for o in self.val_outputs], dim=0)
        gt_classes = torch.cat([o["gt_class"] for o in self.val_outputs], dim=0)
        gt_boxes = torch.cat([o["gt_box"] for o in self.val_outputs], dim=0)

        Cacc, Lacc, Dacc = compute_metrics(
            pred_classes, pred_boxes, gt_classes, gt_boxes
        )

        self.log("val_Cacc", Cacc, prog_bar=True)
        self.log("val_Lacc", Lacc, prog_bar=True)
        self.log("val_Dacc", Dacc, prog_bar=True)

        self.val_outputs.clear()

    def configure_optimizers(self):
        if self.freeze_seg:
            optimizer = torch.optim.AdamW(self.grasp.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(
                list(self.seg.model.parameters()) + list(self.grasp.parameters()),
                lr=self.lr,
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5, last_epoch=-1
        )

        return [optimizer], [scheduler]

    # def on_train_epoch_start(self):
    #     if self.freeze_seg and self.current_epoch == self.unfreeze_at_epoch:
    #         print(f"Unfreezing segmentation backbone at epoch {self.current_epoch}")
    #         for p in self.seg.model.parameters():
    #             p.requires_grad = True
    #         self.freeze_seg = False

    def visualize_grasp(self, imgs, pred_box, pred_angle, pred_class, batch_idx):
        imgs = imgs.detach().cpu()
        pred_box = pred_box.detach().cpu()
        pred_angle = pred_angle.detach().cpu()
        pred_class = pred_class.argmax(dim=1).detach().cpu()

        drawn_imgs = []
        for i in range(min(4, imgs.size(0))):  # 최대 4개까지만
            img = imgs[i]
            img = img.permute(1, 2, 0).cpu().numpy()  # CHW → HWC + NumPy 변환
            img = (img * 255).astype(np.uint8)

            cx, cy, w, h = pred_box[i]

            theta = math.degrees(math.asin(pred_angle[i].item()))

            flg = 1
            if theta < 0:
                flg = -1
            theta = 90 * flg - theta

            cx = int(cx * 640)
            cy = int(cy * 640)
            w = int(w * 640)
            h = int(h * 640)

            rect = cv2.boxPoints(((cx, cy), (w, h), theta))
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
