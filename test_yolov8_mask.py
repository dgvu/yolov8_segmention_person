from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = "yolov8n-seg.pt"  # đổi thành yolov8_croprow_nano.pt nếu test model custom
IMAGE_PATH = "test.jpg"
OUT_DIR = Path("seg_output")
OUT_DIR.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)

results = model(
    IMAGE_PATH,
    imgsz=320,
    conf=0.25,
    retina_masks=True
)

r = results[0]
r.save(filename=str(OUT_DIR / "pc_result.jpg"))

print("Image:", IMAGE_PATH)
print("Model:", MODEL_PATH)
print("Class names:", r.names)

if r.masks is None:
    print("No mask detected.")
    exit()

masks = r.masks.data.cpu().numpy()
boxes = r.boxes.xyxy.cpu().numpy()
classes = r.boxes.cls.cpu().numpy()
scores = r.boxes.conf.cpu().numpy()

print("Mask tensor shape:", masks.shape)

for i, mask in enumerate(masks):
    mask_bin = (mask > 0.5).astype(np.uint8)

    class_id = int(classes[i])
    class_name = r.names[class_id]
    score = float(scores[i])

    np.save(OUT_DIR / f"mask_{i}.npy", mask_bin)
    cv2.imwrite(str(OUT_DIR / f"mask_{i}.png"), mask_bin * 255)

    area_ratio = mask_bin.sum() / mask_bin.size

    print(f"Object {i}")
    print(f"  class: {class_name}")
    print(f"  score: {score:.3f}")
    print(f"  box: {boxes[i]}")
    print(f"  mask area ratio: {area_ratio:.4f}")