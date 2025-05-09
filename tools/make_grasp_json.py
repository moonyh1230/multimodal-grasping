import os, json, math
from glob import glob
from tqdm import tqdm

from ultralytics import YOLO


def main():
    m = YOLO("sg_15class_0429.pt")
    IMG_DIR = "data/inst_dataset/images"
    LBL_DIR = "data/inst_dataset/labels"
    SAVE_PATH = "data/inst_dataset/grasp_mod.json"

    output = {}

    output["nc"] = m.model.nc
    output["names"] = m.model.names

    labels = {}

    for lbl_path in tqdm(sorted(glob(f"{LBL_DIR}/*.txt"))):
        base = os.path.splitext(os.path.basename(lbl_path))[0]
        img = os.path.join(IMG_DIR, base + ".jpg")
        det_res = m(img, conf=0.75, verbose=False)
        if det_res[0].masks == None:
            det_res = m(img, max_det=1, verbose=False)
        det_res = det_res[0]
        seg_list = []
        for n, (box, mask) in enumerate(zip(det_res.boxes, det_res.masks)):
            x1, y1, x2, y2, conf, cls = map(float, box.data[0])
            mxy = mask.xy[0].tolist()

            seg_list.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class": int(cls),
                    "mask": mxy,
                }
            )

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        grasp_list = []
        for i, line in enumerate(lines):
            vals = line.strip().split()
            if len(vals) != 6:
                continue  # 잘못된 줄 skip

            cx, cy, w, h, theta_deg, cls = map(float, vals)
            theta_rad = math.radians(theta_deg)

            grasp_list.append(
                {
                    "id": i,
                    "x": cx,  # 중심 x (px)
                    "y": cy,  # 중심 y (px)
                    "w": h,  # gripper 길이 (long axis)
                    "h": w,  # gripper 폭 (짧은 axis)
                    "theta": theta_rad,  # θ (radian)
                    # "class": int(cls),  # 객체 class
                }
            )

        labels[base] = {"segments": seg_list, "grasps": grasp_list}

    output["labels"] = labels

    with open(SAVE_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
