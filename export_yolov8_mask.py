from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = "yolov8n-seg.pt"
IMAGE_PATH = "test.jpg"
OUT_DIR = Path("pc_mask_output")
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

if r.masks is None:
    print("No mask detected")
    exit()

masks = r.masks.data.cpu().numpy()
boxes = r.boxes.xyxy.cpu().numpy()
classes = r.boxes.cls.cpu().numpy()
scores = r.boxes.conf.cpu().numpy()

print("Number of objects:", len(masks))
print("Mask shape:", masks.shape)

for i, mask in enumerate(masks):
    mask_bin = (mask > 0.5).astype(np.uint8)

    class_id = int(classes[i])
    class_name = r.names[class_id]
    score = float(scores[i])

    np.save(OUT_DIR / f"mask_{i}.npy", mask_bin)
    cv2.imwrite(str(OUT_DIR / f"mask_{i}.png"), mask_bin * 255)

    # Lưu raw binary mask cho C/C++ nếu cần
    mask_bin.tofile(OUT_DIR / f"mask_{i}.bin")

    with open(OUT_DIR / f"mask_{i}_info.txt", "w") as f:
        f.write(f"class_id={class_id}\n")
        f.write(f"class_name={class_name}\n")
        f.write(f"score={score}\n")
        f.write(f"box={boxes[i].tolist()}\n")
        f.write(f"mask_h={mask_bin.shape[0]}\n")
        f.write(f"mask_w={mask_bin.shape[1]}\n")

    print(f"Saved object {i}: {class_name}, score={score:.3f}")