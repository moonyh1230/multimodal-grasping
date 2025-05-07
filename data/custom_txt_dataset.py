import os, json, cv2, torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import math
from PIL import Image


class GraspTxtDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None, img_size=640):
        self.img_dir = img_dir
        self.img_size = img_size
        with open(label_json) as f:
            self.grasp_data = json.load(f)
        self.transform = transform or Compose(
            [Resize((self.img_size, self.img_size)), ToTensor()]
        )
        self.keys = list(self.grasp_data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        decPlaces = 8
        name = self.keys[idx]
        path = os.path.join(self.img_dir, f"{name}.jpg")
        img = cv2.imread(path)[..., ::-1]
        H, W = img.shape[:2]
        scaleX = self.img_size / W
        scaleY = self.img_size / H
        img = Image.fromarray(img)
        img_t = self.transform(img)

        grasps_raw = self.grasp_data[name]

        grasps = []
        classes = []
        for g in grasps_raw:
            cx, cy = g["x"], g["y"]
            w = g["w"]
            h = g["h"]
            theta = g["theta"]
            class_id = g["class"]

            cx, cy, w, h = cx * scaleX, cy * scaleY, w * scaleX, h * scaleY
            cx = round(cx / self.img_size, decPlaces)
            cy = round(cy / self.img_size, decPlaces)
            w = round(w / self.img_size, decPlaces)
            h = round(h / self.img_size, decPlaces)

            theta = math.degrees(theta)

            flg = 1
            if theta < 0:
                flg = -1
            theta = math.radians(90 * flg - theta)
            alpha = round(math.sin(theta), decPlaces)

            grasps.append([cx, cy, w, h, alpha])  # [cx, cy, w, h, sinθ]
            classes.append(class_id)  # class_id

        return {
            "image": img_t,
            "grasps": torch.tensor(grasps, dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.long),
        }
