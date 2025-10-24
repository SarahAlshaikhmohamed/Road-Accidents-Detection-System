# Road Accident and Pothole Detection Dataset 

## Overview

This dataset was prepared as part of the Road Accident Detection System (RADS) project.
It combines two main visual datasets: accident detection and pothole detection.
The merged dataset is used for model training and evaluation of real-time road monitoring systems.

## Dataset Summary

| Split      | Images   | Description                  |
| :--------- | :------- | :--------------------------- |
| Train      | 5823     | Training data                |
| Validation | 558      | Validation data              |
| **Total**  | **6381** | Annotated images |

## Dataset Structure

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

**data.yaml**

```yaml
train: ../train/images
val: ../valid/images
nc: 2
names: ['Accident', 'Pothole']
```

## Source Datasets

1. **Traffic Accident Detection Dataset**
   Source: [Traffic Accident Detection Dataset - Kaggle](https://www.kaggle.com/datasets/cubeai/traffic-accident-detection-for-yolov8/)
   Description: 7k images of traffic accidents with bounding boxes for 10 classes types (motorcycle vs person, car vs car, person, car and other objects and road collision scenarios). 

2. **Pothole Detection Dataset**
   Source: [Potholes Detection YOLOv8 - Kaggle](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8)
   Description: 2k images of annotated pothole images used for road surface condition detection.


## Data Link

Dataset is available on Google Drive:
[Download merged_accident_pothole Dataset](https://drive.google.com/file/d/19Br4PRoYKPlEMHijX7sGWTwquDHoOUK9/view?usp=sharing)


