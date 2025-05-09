from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from models.seg_backbone import create_yolov8_model
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

    out["img"] = torch.stack([d["img"] for d in batch], dim=0)

    bboxes_list = []
    masks_list = []
    grasps_list = []
    classes_list = []
    idxs_list = []

    for batch_idx, d in enumerate(batch):
        n = d["grasps"].size(0)
        grasps_list.append(d["grasps"])
        classes_list.append(d["cls"])
        bboxes_list.append(d["bboxes"])
        masks_list.append(d["masks"])
        idxs_list.append(torch.full((n,), batch_idx, dtype=torch.long))

    out["grasps"] = torch.cat(grasps_list, dim=0)
    out["cls"] = torch.cat(classes_list, dim=0)
    out["batch_idx"] = torch.cat(idxs_list, dim=0)
    out["masks"] = torch.cat(masks_list, dim=0)
    out["bboxes"] = torch.cat(bboxes_list, dim=0)

    return out


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("checkpoints", timestamp)

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

    torch.cuda.set_device(0)

    # seg = create_yolov8_model(
    #     "sg_15class_0429.pt", nc=len(classes), class_names=classes
    # )
    seg = create_yolov8_model("sg_15class_0429.pt")  # YOLOv8m-seg fine-tuned 모델

    grasp = GraspHeadROI(in_channels=576, num_classes=15)

    lit = LitGrasp(seg, grasp, classes_name=classes, freeze_seg=True)

    ds = GraspTxtDataset(
        img_dir="data/inst_dataset/images",
        label_json="data/inst_dataset/grasp_mod.json",
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
        monitor="val_Dacc",  # 모니터링할 metric
        mode="max",  # val_Dacc 클수록 좋음
        save_top_k=1,  # 가장 좋은 모델 1개만 저장
        save_last=True,
        filename="{epoch:03d}-{val_Dacc:.4f}-best",  # best 모델 파일명
        verbose=True,
    )

    # ✅ EarlyStopping 콜백
    early_stop_callback = EarlyStopping(
        monitor="val_Dacc", patience=15, verbose=True, mode="max"
    )

    trainer = Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=20,
        precision="16-mixed",  # fp16 mixed precision for faster training
        default_root_dir=save_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
