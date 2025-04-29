import os, json, cv2, torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import math
from PIL import Image


class GraspTxtDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None):
        self.img_dir = img_dir
        with open(label_json) as f:
            self.grasp_data = json.load(f)
        self.transform = transform or Compose([Resize((640, 640)), ToTensor()])
        self.keys = list(self.grasp_data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx]
        path = os.path.join(self.img_dir, f"{name}.jpg")
        img = cv2.imread(path)[..., ::-1]
        H, W = img.shape[:2]
        img = Image.fromarray(img)
        img_t = self.transform(img)

        grasps_raw = self.grasp_data[name]

        boxes = []
        grasps = []
        classes = []
        for g in grasps_raw:
            cx, cy = g["x"], g["y"]
            w = g["w"]
            h = g["h"]
            theta = g["theta"]
            class_id = g["class"]

            half = w / 2
            box = [
                max(0, cx - half),
                max(0, cy - half),
                min(W, cx + half),
                min(H, cy + half),
            ]
            boxes.append(box)

            grasps.append([cx / W, cy / H, w / W, h / H, theta])
            classes.append(class_id)

        return {
            "image": img_t,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "grasps": torch.tensor(grasps, dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.long),
        }
