import os, json, cv2, torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from ultralytics.data.utils import polygon2mask
import numpy as np
import math
from PIL import Image


class GraspTxtDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None, img_size=(640, 640)):
        self.img_dir = img_dir
        self.img_size = img_size
        with open(label_json) as f:
            self.grasp_data = json.load(f)

        self.nc = self.grasp_data["nc"]
        self.names = self.grasp_data["names"]
        self.transform = transform or Compose([Resize(self.img_size), ToTensor()])
        self.keys = list(self.grasp_data["labels"].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        decPlaces = 8
        name = self.keys[idx]
        path = os.path.join(self.img_dir, f"{name}.jpg")
        img = cv2.imread(path)[..., ::-1]
        H, W = img.shape[:2]
        scaleX = self.img_size[0] / W
        scaleY = self.img_size[1] / H
        img = Image.fromarray(img)
        img_t = self.transform(img)

        grasps_raw = self.grasp_data["labels"][name]

        grasps = []
        bboxes = []
        masks = []
        classes = []
        for s, g in zip(grasps_raw["segments"], grasps_raw["grasps"]):
            bx1, by1, bx2, by2 = (
                s["x1"],
                s["y1"],
                s["x2"],
                s["y2"],
            )
            poly = s["mask"]
            cx, cy = g["x"], g["y"]
            w = g["w"]
            h = g["h"]
            theta = g["theta"]
            class_id = s["class"]

            mask_img = polygon2mask((W, H), poly)

            # mask[:, 0] = np.round((mask[:, 0] * scaleX) / self.img_size, decPlaces)
            # mask[:, 1] = np.round((mask[:, 1] * scaleY) / self.img_size, decPlaces)

            bx1, by1, bx2, by2 = bx1 * scaleX, by1 * scaleY, bx2 * scaleX, by2 * scaleY

            cx, cy, w, h = cx * scaleX, cy * scaleY, w * scaleX, h * scaleY
            cx = round(cx / self.img_size[0], decPlaces)
            cy = round(cy / self.img_size[1], decPlaces)
            w = round(w / self.img_size[0], decPlaces)
            h = round(h / self.img_size[1], decPlaces)

            theta = math.degrees(theta)

            flg = 1
            if theta < 0:
                flg = -1
            theta = math.radians(90 * flg - theta)
            alpha = round(math.sin(theta), decPlaces)

            grasps.append([cx, cy, w, h, alpha])  # [cx, cy, w, h, sinθ]
            classes.append(class_id)  # class_id
            bboxes.append([bx1, by1, bx2, by2])
            masks.append(mask_img)

        return {
            "img": img_t,
            "grasps": torch.tensor(grasps, dtype=torch.float32),
            "cls": torch.tensor(classes, dtype=torch.long),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "masks": torch.tensor(np.array(masks), dtype=torch.float32),
        }
