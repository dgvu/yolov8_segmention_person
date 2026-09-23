# YOLOv8 Person Segmentation on ZCU104/PYNQ using NCNN

This repository documents an **ARM-only deployment of YOLOv8 instance segmentation for person detection/segmentation on the Xilinx ZCU104 board running PYNQ**.

The project does **not use the DPU/Vitis-AI acceleration path**. Instead, the YOLOv8 segmentation model is exported to **NCNN** and executed on the ARM cores of the Zynq UltraScale+ MPSoC.

The repository keeps the model, PC-side validation scripts, NCNN model files, test images/videos, and outputs collected from the ZCU104 board.

---

## 1. Project objective

The goal of this project is to deploy a lightweight YOLOv8 segmentation model on an embedded FPGA platform and verify that:

- the original PyTorch model produces valid person masks;
- the exported NCNN model produces comparable segmentation results;
- the NCNN model can run directly on the ARM processor of the ZCU104/PYNQ system;
- both image and video inference can be executed without requiring a GPU or DPU;
- segmentation masks and bounding boxes can be visualized and saved for later comparison.

The deployment flow is:

```text
YOLOv8n-Seg (.pt)
        |
        v
PC validation with Ultralytics
        |
        v
Export to NCNN
        |
        v
NCNN validation on PC
        |
        v
Copy NCNN model + input data to ZCU104/PYNQ
        |
        v
ARM inference
        |
        v
Bounding boxes + person masks
        |
        v
Image / video output
```

---

## 2. Hardware and software

### Hardware

- **Board:** Xilinx ZCU104
- **SoC:** Zynq UltraScale+ MPSoC
- **Processor used:** ARM Cortex-A53
- **Acceleration mode:** CPU/ARM only
- **DPU:** not used in this project

### Target operating system

- PYNQ image for ZCU104
- Linux running on the ARM processing system

### Main software

- Python
- Ultralytics YOLOv8
- OpenCV
- NumPy
- NCNN
- PNNX / Ultralytics NCNN export pipeline

---

## 3. Model

The base model used in this repository is:

```text
yolov8n-seg.pt
```

YOLOv8n-Seg is used because it is lighter than larger YOLOv8 segmentation variants and is therefore more suitable for ARM-only inference on an embedded board.

The repository focuses on person instance segmentation.

Typical inference parameters used during validation are:

```text
input size : 320 x 320
confidence : 0.25
task       : segment
```

The PC-side scripts also enable high-resolution mask reconstruction with:

```python
retina_masks=True
```

---

## 4. Repository structure

```text
yolov8_segmention_person/
|
|-- README.md
|
|-- yolov8n-seg.pt
|   Original YOLOv8n segmentation model.
|
|-- yolov8n-seg_ncnn_model/
|   NCNN-exported model used for ARM deployment.
|
|-- test.jpg
|   Test image used for PC and embedded validation.
|
|-- test_video.mp4
|   Video used for embedded segmentation testing.
|
|-- test_yolov8_mask.py
|   Tests the original PyTorch YOLOv8 segmentation model.
|
|-- export_yolov8_mask.py
|   Exports masks, boxes, class IDs and confidence values
|   from the PyTorch model for debugging/comparison.
|
|-- test_ncnn_mask.py
|   Tests the exported NCNN model on the PC.
|
|-- ncnn_pc_result.jpg
|   Example NCNN result generated on the PC.
|
|-- pc_mask_output/
|   Masks and metadata exported from the PyTorch model.
|
|-- seg_output/
|   PyTorch segmentation test outputs.
|
|-- zcu104_img/
|   Images prepared for ZCU104 testing.
|
|-- zcu104_output/
|   Image inference results generated on ZCU104.
|
|-- zcu104_vid/
|   Video inputs used on the ZCU104.
|
|-- zcu104_video_output_1/
|   First group of video inference outputs.
|
`-- zcu104_video_output_2/
    Second group of video inference outputs.
```

---

## 5. PC-side environment

A typical Ubuntu environment can be prepared with:

```bash
python3 -m venv yolo_env
source yolo_env/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

Check the installation:

```bash
python3 - <<'PY'
import cv2
import numpy as np
import ultralytics

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Ultralytics:", ultralytics.__version__)
PY
```

---

## 6. Test the original YOLOv8 segmentation model

The script:

```text
test_yolov8_mask.py
```

loads `yolov8n-seg.pt` and runs segmentation on `test.jpg`.

Run:

```bash
python3 test_yolov8_mask.py
```

The script uses:

```python
results = model(
    IMAGE_PATH,
    imgsz=320,
    conf=0.25,
    retina_masks=True
)
```

It saves the rendered result to:

```text
seg_output/pc_result.jpg
```

and prints information for each detected object:

- class name;
- confidence score;
- bounding box;
- mask tensor shape;
- mask area ratio.

This step is the reference result before conversion to NCNN.

---

## 7. Export raw masks for comparison

The script:

```text
export_yolov8_mask.py
```

extracts mask information from the original PyTorch model.

Run:

```bash
python3 export_yolov8_mask.py
```

For every detected object the script stores:

```text
mask_i.npy
mask_i.png
mask_i.bin
mask_i_info.txt
```

The information file contains:

```text
class_id
class_name
score
bounding box
mask height
mask width
```

These files are useful when debugging differences between PyTorch, NCNN on PC and NCNN on ZCU104.

---

## 8. Export YOLOv8-Seg to NCNN

A reproducible Ultralytics export command is:

```bash
yolo export     model=yolov8n-seg.pt     format=ncnn     imgsz=320
```

The export generates an NCNN model directory similar to:

```text
yolov8n-seg_ncnn_model/
```

Depending on the Ultralytics/PNNX version, the directory normally contains NCNN graph and weight files such as:

```text
model.ncnn.param
model.ncnn.bin
```

or equivalent generated filenames.

The exported model should be tested on the PC before copying it to the ZCU104.

---

## 9. Validate the NCNN model on the PC

The repository contains:

```text
test_ncnn_mask.py
```

Run:

```bash
python3 test_ncnn_mask.py
```

The script loads:

```python
model = YOLO("yolov8n-seg_ncnn_model")
```

and executes:

```python
results = model(
    "test.jpg",
    imgsz=320,
    conf=0.25,
    task="segment"
)
```

The result is saved to:

```text
ncnn_pc_result.jpg
```

Before deploying to the board, compare the NCNN result with the PyTorch result.

If PyTorch is correct but NCNN is already different on the PC, investigate the export/conversion stage before debugging the ZCU104.

---

## 10. Why NCNN is used on ZCU104/PYNQ

The PYNQ image provides Linux on the ARM processing system of the Zynq UltraScale+ MPSoC.

NCNN is suitable for this experiment because it targets lightweight embedded inference and supports ARM optimizations.

The processing chain in this repository is:

```text
Input image/video
      |
      v
ARM/OpenCV preprocessing
      |
      v
NCNN inference on ARM
      |
      v
YOLOv8 box decoding
      |
      v
Segmentation mask reconstruction
      |
      v
OpenCV overlay/drawing
      |
      v
Output image/video
```

No DPU/XModel is used in this repository.

---

## 11. Building NCNN for ARM64 on ZCU104

One possible native build flow is:

```bash
sudo apt update
sudo apt install -y git cmake build-essential
```

Clone NCNN:

```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
```

Build:

```bash
mkdir -p build
cd build

cmake     -DNCNN_VULKAN=OFF     -DNCNN_BUILD_EXAMPLES=ON     ..

make -j4
sudo make install
```

For this ARM-only project, `NCNN_VULKAN=OFF` is sufficient.

The exact package availability depends on the PYNQ image version.

---

## 12. Copy files to the ZCU104

Example from the Ubuntu host:

```bash
scp -r yolov8n-seg_ncnn_model     root@<ZCU104_IP>:/home/root/yolov8_person/
```

Copy an image:

```bash
scp test.jpg     root@<ZCU104_IP>:/home/root/yolov8_person/
```

Copy a video:

```bash
scp test_video.mp4     root@<ZCU104_IP>:/home/root/yolov8_person/
```

Connect to the board:

```bash
ssh root@<ZCU104_IP>
```

---

## 13. Image inference on ZCU104

The image-deployment stage performs:

```text
1. Read image
2. Resize / letterbox
3. Normalize input
4. Run NCNN model on ARM
5. Decode bounding boxes
6. Apply confidence filtering / NMS
7. Reconstruct segmentation masks
8. Resize masks to the original image
9. Overlay masks
10. Save result
```

Results generated on the board are stored in:

```text
zcu104_output/
```

Input images used for board testing are stored in:

```text
zcu104_img/
```

---

## 14. Video inference on ZCU104

Video inference follows the same model one frame at a time:

```text
Video
  |
  v
ARM reads frame
  |
  v
Preprocess
  |
  v
NCNN YOLOv8-Seg inference
  |
  v
Decode box + mask
  |
  v
Draw segmentation
  |
  v
Write processed frame
  |
  v
Output video
```

The repository keeps video-related data in:

```text
zcu104_vid/
zcu104_video_output_1/
zcu104_video_output_2/
```

This makes it possible to compare multiple embedded inference runs.

---

## 15. Debugging workflow

When the ZCU104 output differs from the PC result, debug in this order:

### Step 1 - PyTorch reference

```bash
python3 test_yolov8_mask.py
```

Verify the box, class and segmentation mask.

### Step 2 - NCNN on PC

```bash
python3 test_ncnn_mask.py
```

If PyTorch is correct but NCNN is wrong, investigate model conversion.

### Step 3 - Embedded preprocessing

Verify that the ZCU104 implementation uses the same:

```text
input resolution
letterbox policy
RGB/BGR conversion
normalization
padding
```

### Step 4 - Embedded post-processing

Verify:

```text
tensor layout
bounding-box decoding
confidence threshold
NMS
mask coefficients
prototype mask
mask crop
mask resize
```

### Step 5 - Drawing

Only tune mask threshold, overlay colors and contours after raw network outputs are known to match.

---

## 16. Why segmentation is heavier than detection

YOLOv8 segmentation produces both detection and mask information:

```text
YOLOv8-Seg
|
|-- Detection
|   |-- bounding boxes
|   `-- class scores
|
`-- Segmentation
    |-- mask coefficients
    `-- prototype masks
```

The final instance mask is reconstructed during post-processing.

On an ARM-only embedded system, mask reconstruction and resizing can consume a significant part of the total execution time.

---

## 17. ARM-only limitation

This repository specifically represents an ARM implementation:

```text
YOLOv8 inference       -> ARM
Detection postprocess  -> ARM
Mask reconstruction    -> ARM
OpenCV drawing         -> ARM
Video encode/decode    -> ARM
```

The programmable-logic DPU is not used for neural-network acceleration.

This project is therefore useful as:

- an ARM/NCNN embedded baseline;
- a validation path before hardware acceleration;
- a comparison point for a later Vitis-AI/DPU implementation.

Real-time performance should not be assumed for high-resolution segmentation workloads.

---

## 18. Possible improvements

Possible next steps include:

1. Keep inference resolution at 320 or reduce it further if accuracy remains acceptable.
2. Filter results to the `person` class only.
3. Reduce the number of retained detections before mask reconstruction.
4. Move preprocessing/post-processing to optimized C++.
5. Enable ARM/NEON-optimized NCNN builds.
6. Reduce video resolution before segmentation.
7. Profile inference and mask reconstruction separately.
8. Investigate NCNN quantization.
9. Port the CNN portion to the ZCU104 DPU using Vitis-AI.
10. Compare ARM-NCNN and DPU-VART with identical images and videos.

---

## 19. Project status

- [x] YOLOv8n-Seg PyTorch model
- [x] PyTorch segmentation validation
- [x] Raw mask export/debugging
- [x] NCNN model export
- [x] NCNN validation on PC
- [x] ZCU104 image tests
- [x] ZCU104 image outputs
- [x] ZCU104 video tests
- [x] Multiple video output runs
- [x] ARM-only deployment workflow
- [ ] DPU/Vitis-AI acceleration

---

## 20. Repository naming note

The repository name currently uses:

```text
segmention
```

instead of:

```text
segmentation
```

This does not affect execution, but renaming it later may make the project easier to search and present.

---

## 21. References

- Ultralytics YOLOv8
- Tencent NCNN
- PNNX
- Xilinx ZCU104
- PYNQ

---

## 22. Author

Repository:

```text
https://github.com/dgvu/yolov8_segmention_person
```

This repository is intended as a practical deployment record for running **YOLOv8 person instance segmentation on the ARM processing system of ZCU104/PYNQ using NCNN**.
