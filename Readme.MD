# Car Object Detection

This repository contains code for car object detection using YOLOv8, integrated with a Streamlit web application. The project detects cars in video files and provides a user interface for uploading videos and viewing the processed results.

## Demo Video

[![Demo Video](https://player.vimeo.com/video/956320509/0.jpg)](https://player.vimeo.com/video/956320509)

<div style="padding:56.25% 0 0 0;position:relative;">
  <iframe src="https://player.vimeo.com/video/956320509?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="demo"></iframe>
</div>

## Table of Contents
- [Installation](#installation)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [Demo Video](#demo-video)
- [Data](#data)
- [License](#license)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/car-object-detection.git
cd car-object-detection
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate 
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Ensure you have the necessary data files:

The data folder is not included in the repository due to size limitations. You can download the dataset from Kaggle: [Car Object Detection Dataset](https://www.kaggle.com/dataset).

Run the Streamlit app:

```bash
streamlit run streamlit.py
```

**Usage:**
- Upload a video file from the sidebar.
- Click the "Detect" button to process the uploaded video.
- You can also process the default video (`videos/computer vision.mp4`) by clicking the "Detect Video on Main Page" button on the main page.

## Training the Model

**Prepare the Data:**

Ensure your training and validation images are organized in `data/training_images` and `data/validation_images` respectively.

Adjust the paths in `src/data.yaml` if necessary.

Run the training script:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='src/data.yaml', epochs=100, imgsz=640)
```

**After training:**

The trained model weights will be saved in the `runs/detect/train3/weights` directory.

Update the path to the best weights in `src/video_test.py` and `streamlit.py` if necessary.

## Project Structure

```
car-object-detection/
├── data/
│   ├── testing_images/
│   ├── training_images/
│   └-- Bounding_boxes/
├── predicted_video/
│   └── output_video.mp4
├── runs/
│   └── detect/
│       └── train3/
│           └── weights/
│               └── best.pt
├── src/
│   ├── data.yaml
│   ├── images_to_video.py
│   ├── split.py
│   ├── testing.ipynb
│   ├── training.ipynb
│   ├── video_test.py
│   └── yoloformat.py
├── videos/
│   ├── computer vision.mp4
│   └── demo.mp4
├── Readme.MD
├── requirements.txt
├── streamlit.py
└── yolov8n.pt
```

## Demo Video

A demo video showing the Streamlit application in action can be found in [videos/demo.mp4](videos/demo.mp4).

## Data

The data folder contains the images used for training and testing. Due to size constraints, the folder is not included in this repository. You can download the dataset from Kaggle: [Car Object Detection Dataset](https://www.kaggle.com/dataset).

