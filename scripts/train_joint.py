from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from models.seg_backbone import SegBackbone
from models.grasp_head_roi import GraspHeadROI
from data.custom_txt_dataset import GraspTxtDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from engine.trainer import LitGrasp
from datetime import datetime
import pytorch_lightning as pl
import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_float32_matmul_precision("medium")


def my_collate_fn(batch):
    out = {}
    batch_size = len(batch)

    out["image"] = torch.stack([d["image"] for d in batch], dim=0)

    boxes_list = []
    grasps_list = []
    classes_list = []
    idxs_list = []

    for batch_idx, d in enumerate(batch):
        n = d["boxes"].size(0)
        boxes_list.append(d["boxes"])
        grasps_list.append(d["grasps"])
        classes_list.append(d["classes"])
        idxs_list.append(torch.full((n,), batch_idx, dtype=torch.long))

    out["boxes"] = torch.cat(boxes_list, dim=0)
    out["grasps"] = torch.cat(grasps_list, dim=0)
    out["classes"] = torch.cat(classes_list, dim=0)
    out["idxs"] = torch.cat(idxs_list, dim=0)

    return out


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("checkpoints", timestamp)

    os.makedirs(save_dir, exist_ok=True)

    torch.cuda.set_device(0)

    seg = SegBackbone(model_path="sg_15class_0429.pt")  # YOLOv8m-seg fine-tuned 모델
    grasp = GraspHeadROI(in_channels=576, num_classes=15)

    lit = LitGrasp(seg, grasp, freeze_seg=False)

    ds = GraspTxtDataset(
        img_dir="data/inst_dataset/images", label_json="data/inst_dataset/grasp.json"
    )

    n_total = len(ds)
    n_val = int(n_total * 0.2)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=my_collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=my_collate_fn,
        persistent_workers=True,
    )

    # ✅ best 모델 저장 콜백
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # 모니터링할 metric
        mode="min",  # train_loss 작을수록 좋음
        save_top_k=1,  # 가장 좋은 모델 1개만 저장
        save_last=True,
        dirpath=save_dir,
        filename="{epoch:03d}-{val_loss:.4f}-best",  # best 모델 파일명
        verbose=True,
    )

    # ✅ EarlyStopping 콜백
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=15, verbose=True, mode="min"
    )

    trainer = Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=20,
        precision=16,  # fp16 mixed precision for faster training
        default_root_dir=save_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
