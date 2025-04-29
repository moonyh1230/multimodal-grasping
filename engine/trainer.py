import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from utils.metrics import compute_metrics
from torch.optim import AdamW
import cv2


class LitGrasp(pl.LightningModule):
    def __init__(self, seg, grasp, lr=0.0001, freeze_seg=False, unfreeze_at_epoch=20):
        super().__init__()
        self.seg = seg
        self.grasp = grasp
        self.lr = lr
        self.freeze_seg = freeze_seg
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.alpha = 10
        self.beta = 10

        # ✅ freeze segmentation backbone
        if self.freeze_seg:
            for p in self.seg.model.model.parameters():
                p.requires_grad = False

        self.save_hyperparameters(ignore=["seg", "grasp"])

    def forward(self, imgs):
        boxes, idxs, _ = self.seg(imgs)
        feats = self.seg.extract_p5_feature(imgs)
        return self.grasp(feats, boxes, idxs)

    def training_step(self, batch, batch_idx):
        imgs = batch["image"]
        boxes = batch["boxes"]
        idxs = batch["idxs"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["classes"]

        feats = self.seg.extract_p5_feature(imgs)
        pred_box, pred_angle, pred_class = self.grasp(feats, boxes, idxs)

        loss_box = F.mse_loss(pred_box, grasps_gt[:, :4])
        loss_angle = F.mse_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = self.alpha * loss_box + loss_angle

        total_loss = self.beta * mseloss + loss_class

        self.log("loss_box", loss_box)
        self.log("loss_angle", loss_angle)
        self.log("loss_class", loss_class)
        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs = batch["image"]
        boxes = batch["boxes"]
        idxs = batch["idxs"]
        grasps_gt = batch["grasps"]
        classes_gt = batch["classes"]

        feats = self.seg.extract_p5_feature(imgs)
        pred_box, pred_angle, pred_class = self.grasp(feats, boxes, idxs)

        loss_box = F.mse_loss(pred_box, grasps_gt[:, :4])
        loss_angle = F.mse_loss(pred_angle, grasps_gt[:, 4:5])
        loss_class = F.cross_entropy(pred_class, classes_gt)

        mseloss = self.alpha * loss_box + loss_angle

        val_loss = self.beta * mseloss + loss_class

        self.log("val_loss_box", loss_box)
        self.log("val_loss_angle", loss_angle)
        self.log("val_loss_class", loss_class)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

        if batch_idx % 50 == 0:
            self.visualize_grasp(imgs, pred_box, pred_angle, pred_class, batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        pred_class_list = []
        pred_box_list = []
        gt_class_list = []
        gt_box_list = []

        for output in outputs:
            pred_class_list.append(output["pred_class"])
            pred_box_list.append(output["pred_box"])
            gt_class_list.append(output["gt_class"])
            gt_box_list.append(output["gt_box"])

        pred_classes = torch.cat(pred_class_list, dim=0)
        pred_boxes = torch.cat(pred_box_list, dim=0)
        gt_classes = torch.cat(gt_class_list, dim=0)
        gt_boxes = torch.cat(gt_box_list, dim=0)

        cacc, lacc, dacc = compute_metrics(
            pred_classes, pred_boxes, gt_classes, gt_boxes
        )

        self.log("val_Cacc", cacc, prog_bar=True)
        self.log("val_Lacc", lacc, prog_bar=True)
        self.log("val_Dacc", dacc, prog_bar=True)

    def configure_optimizers(self):
        if self.freeze_seg:
            optimizer = torch.optim.Adam(self.grasp.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(
                list(self.seg.model.model.parameters()) + list(self.grasp.parameters()),
                lr=self.lr,
            )
        return optimizer

    def on_train_epoch_start(self):
        if self.freeze_seg and self.current_epoch == self.unfreeze_at_epoch:
            print(f"Unfreezing segmentation backbone at epoch {self.current_epoch}")
            for p in self.seg.model.model.parameters():
                p.requires_grad = True
            self.freeze_seg = False

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
            theta = pred_angle[i].item() * 90.0  # [-1, 1] → [-90°, 90°]

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
                f"Class: {cls_id}",
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
