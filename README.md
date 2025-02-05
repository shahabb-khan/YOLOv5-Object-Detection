# YOLOv5 Object Detection

This repository contains a simple implementation of object detection using the YOLOv5 model. It loads an image, performs detection, and displays the results.

## Features
- Uses **YOLOv5s** (small version) for object detection
- Loads and processes images using **OpenCV**
- Saves detected results

## Installation
Clone the repository and install dependencies:
```bash
pip install torch torchvision opencv-python matplotlib
!git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Usage
Run the script:
```python
import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load and preprocess image
image_path = 'aaa.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform detection
results = model(image)
results.show()

# Save output
output_path = 'output_image.jpg'
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
```

## Output
The script will display detected objects and save the output image.

## Credits
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

## License
This project is under the MIT License.

