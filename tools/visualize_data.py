import os
import json
import cv2
import numpy as np
import math

# --- Configuration ---
IMG_DIR = "data/inst_dataset/images"
LABEL_JSON = "data/inst_dataset/grasp_mod_2.json"
OUTPUT_DIR = "output/data_visualization"
NUM_SAMPLES_TO_VISUALIZE = 5
IMG_SIZE = (640, 640)


# --- Main Script ---
def visualize():
    """
    Loads dataset annotations and visualizes the ground truth bounding boxes and grasp rectangles on the images.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load grasp data from JSON
    try:
        with open(LABEL_JSON) as f:
            grasp_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {LABEL_JSON}")
        return

    keys = list(grasp_data["labels"].keys())

    print(
        f"Visualizing {NUM_SAMPLES_TO_VISUALIZE} samples and saving to {OUTPUT_DIR}..."
    )

    for i in range(min(NUM_SAMPLES_TO_VISUALIZE, len(keys))):
        name = keys[i]
        path = os.path.join(IMG_DIR, f"{name}.jpg")

        if not os.path.exists(path):
            print(f"Warning: Image file not found at {path}, skipping.")
            continue

        # Read and resize image
        img_bgr = cv2.imread(path)
        img_resized_bgr = cv2.resize(img_bgr, IMG_SIZE)

        # Get labels for the image
        labels = grasp_data["labels"][name]

        # Draw each ground truth grasp and bbox
        for segment, grasp in zip(labels["segments"], labels["grasps"]):
            # --- Draw Bounding Box (Blue) ---
            # It's in normalized xywh format
            bx, by, bw, bh = segment["xc"], segment["yc"], segment["w"], segment["h"]
            # Convert to xyxy pixel coordinates
            x1 = int((bx - bw / 2) * IMG_SIZE[0])
            y1 = int((by - bh / 2) * IMG_SIZE[1])
            x2 = int((bx + bw / 2) * IMG_SIZE[0])
            y2 = int((by + bh / 2) * IMG_SIZE[1])
            cv2.rectangle(img_resized_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # --- Draw Grasp Rectangle (Green) ---
            # It's in normalized cx, cy, w, h and radians theta
            print(f"DEBUG: Grasp data for {name}: {grasp}")  # Print grasp data
            cx, cy, w, h = grasp["x"], grasp["y"], grasp["w"], grasp["h"]
            theta_rad = grasp["theta"]

            # Grasp coordinates are already in pixel values from the JSON
            cx_px = int(cx)
            cy_px = int(cy)
            w_px = int(w)
            h_px = int(h)
            angle_deg = math.degrees(theta_rad)

            # Get rotated rectangle vertices
            box_points = cv2.boxPoints(((cx_px, cy_px), (w_px, h_px), angle_deg))
            box_points = np.int0(box_points)

            # Draw the contour
            cv2.drawContours(img_resized_bgr, [box_points], 0, (0, 255, 0), 2)

        # Save the visualized image
        output_path = os.path.join(OUTPUT_DIR, f"sample_{i}_{name}.jpg")
        cv2.imwrite(output_path, img_resized_bgr)

    print(f"Visualization complete. Check the files in {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    visualize()
