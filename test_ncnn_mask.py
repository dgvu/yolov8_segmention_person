from ultralytics import YOLO

model = YOLO("yolov8n-seg_ncnn_model")

results = model(
    "test.jpg",
    imgsz=320,
    conf=0.25,
    task="segment"
)

r = results[0]
r.save(filename="ncnn_pc_result.jpg")

print("Boxes:", len(r.boxes))
print("Mask shape:", r.masks.data.shape if r.masks is not None else None)