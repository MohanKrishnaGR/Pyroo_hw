# PyroGuardian — Edge-AI Computer Vision System

## Overview
PyroGuardian is an advanced wildfire detection system designed for real-time inference on edge devices (NVIDIA Jetson Nano). It utilizes state-of-the-art transformer-based object detection (RT-DETR) and is optimized via NVIDIA's DeepStream SDK and TensorRT.

## Architecture
The system consists of two primary implementations:
1.  **Baseline (YOLOv5):** A PyTorch-based implementation used for research benchmarks and comparison.
2.  **Advanced (RT-DETR + DeepStream):** The production-grade engine optimized for Jetson Nano, featuring GStreamer pipelines and FP16 quantization.

## Project Structure
- `deepstream_app/`: Core C++/Python GStreamer pipeline for Jetson deployment.
- `tao_training/`: NVIDIA TAO Toolkit configurations for training the 86.7M parameter RT-DETR model.
- `utils/`: Shared utilities including the AWS SNS notification manager.
- `baseline_yolov5/`: Original YOLOv5 implementation (Baseline).
- `dataset_tools/`: Scripts for curated dataset augmentations.

## Performance
- **Inference Speed:** 30 FPS at 720p resolution on Jetson Nano.
- **Optimization:** 90% speed-up achieved via TensorRT FP16 quantization compared to PyTorch FP32.
- **Accuracy:** Robust detection across 8 fire conditions (Smoke, Forest, Urban, etc.).

## Setup & Usage

### 1. Running the Optimized DeepStream Pipeline (Jetson Only)
```bash
cd deepstream_app
python3 fire_detection_pipeline.py <YOUR_AWS_SNS_ARN>
```

### 2. Training via NVIDIA TAO
```bash
# Requires TAO Toolkit Container
tao rtdetr train -e tao_training/rtdetr_train_spec.yaml -k $API_KEY
```

### 3. Running the Baseline
```bash
cd baseline_yolov5
python main.py --weights yolov5s.pt --source 0
```

## Tech Stack
- **Models:** RT-DETR (ResNet101 Backbone), YOLOv5.
- **Optimization:** TensorRT, CUDA, FP16.
- **Deployment:** DeepStream SDK, GStreamer, NVIDIA Jetson Nano.
- **Cloud:** AWS SNS (Simple Notification Service).
