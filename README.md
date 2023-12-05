# Object Detection using OpenCV and SSD

This repository contains a simple Python script that uses OpenCV to perform real-time object detection using a pre-trained SSD (Single Shot Multibox Detector) model on the COCO dataset.

## Prerequisites

Before running the script, ensure that you have the following installed:

- Python (3.x recommended)
- OpenCV (install using `pip install opencv-python`)
- Download the required model files:
  - [ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt](link-to-model-file)
  - [frozen_inference_graph.pb](link-to-model-file)
- COCO class names file: [coco.names](link-to-coco-names-file)

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

   
Place the downloaded model files (ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt and frozen_inference_graph.pb) in the root directory of the project.
Create a file named coco.names and add the COCO class names to it.
Run the script:
python object_detection.py
Adjust the thres variable in the script to change the confidence threshold for object detection.

Configuration
thres: Confidence threshold for object detection (default is set to 0.6).
References
OpenCV
SSD - Single Shot Multibox Detector
COCO - Common Objects in Context
Feel free to contribute or report issues. Happy coding!
