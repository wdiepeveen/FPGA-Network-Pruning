# Roofline pruning for YOLOv4 on FPGA

```
├── README.md
├── prune_yolov4_pytorch.py               roofline optimization (3.1)
├── design_space_exploration.py           pruning experiment (3.2)
```

# Installation
1. Clone this repo
2. Download the darknet weights from:
  - baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
3. Create a "weights" folder in the project root and save the yolov4.weights file there
4. Clone the following repo from the project root in this project for the YOLOv4 implementation:
- https://github.com/Tianxiaomo/pytorch-YOLOv4
