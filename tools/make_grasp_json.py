import os, json, math
from glob import glob
from tqdm import tqdm

IMG_DIR = "data/inst_dataset/images"
LBL_DIR = "data/inst_dataset/labels"
SAVE_PATH = "data/inst_dataset/grasp.json"

output = {}

for lbl_path in tqdm(sorted(glob(f"{LBL_DIR}/*.txt"))):
    base = os.path.splitext(os.path.basename(lbl_path))[0]

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
                "class": int(cls),  # 객체 class
            }
        )

    output[base] = grasp_list

with open(SAVE_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved: {SAVE_PATH}")
