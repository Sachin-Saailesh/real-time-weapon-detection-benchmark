# Real-Time Weapon Detection Benchmark

> **Production-grade computer vision benchmark comparing YOLOv11 vs RT-DETR for AI-powered CCTV weapon detection with TorchScript/ONNX deployment**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF.svg)]()
[![RT-DETR](https://img.shields.io/badge/RT--DETR-Baidu-orange.svg)]()

## 🎯 Strategic Tagline

Comprehensive benchmarking study of YOLOv11 and RT-DETR object detection models for real-time weapon detection in CCTV systems, achieving RT-DETR mAP50 83.0% (18.7ms) and YOLOv11 mAP50 77.8% (3.1ms) with production-ready ONNX/TorchScript exports.

---

## 💡 Problem & Solution

### **The Challenge**
Public safety systems face critical detection requirements:
- **Real-time Processing**: CCTV systems demand <50ms latency for actionable alerts
- **High Accuracy**: Weapon detection requires >75% mAP50 to minimize false alarms
- **Deployment Constraints**: Edge devices need optimized models <100MB with INT8/FP16 quantization
- **Model Selection**: Unclear tradeoffs between YOLO-series and transformer-based detectors

### **The Solution**
This benchmark implements a rigorous computer vision comparison framework:
- **Dual Architecture Evaluation**: YOLOv11 (CNN-based, 3.1ms) vs RT-DETR (transformer-based, 18.7ms)
- **Production Training Pipeline**: Mixed-precision training on 5,149 images with advanced augmentation
- **Comprehensive Metrics**: COCO evaluation toolkit measuring mAP50, mAP50-95, precision, recall
- **Deployment Optimization**: TorchScript and ONNX export with model quantization strategies
- **Hyperparameter Tuning**: Systematic anchor optimization, augmentation policies, learning rate schedules

---

## 🏗️ Technical Architecture

### **Model Architecture Comparison**

#### **YOLOv11 Architecture**
```
Input (640×640×3)
    ↓
┌─────────────────────────────────┐
│   Backbone: CSPDarknet53        │
│   • Conv blocks with residuals  │
│   • SPPF (Spatial Pyramid Pool) │
│   • Feature scales: P3,P4,P5    │
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│   Neck: PAN (Path Aggregation)  │
│   • Bottom-up pathway            │
│   • Top-down pathway             │
│   • Feature fusion at each scale│
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│   Head: Decoupled Detection     │
│   • Classification head          │
│   • Regression head              │
│   • 3 detection layers (P3-P5)  │
└─────────────────────────────────┘

Parameters: 2.6M
FLOPs: 6.5G
Inference: 3.1ms (RTX 3090)
```

#### **RT-DETR Architecture**
```
Input (640×640×3)
    ↓
┌─────────────────────────────────┐
│   Backbone: HGNetv2             │
│   • Efficient CNN feature       │
│   • Multi-scale features        │
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│   Hybrid Encoder                │
│   • Attention-based fusion      │
│   • Intra-scale feature interact│
│   • Cross-scale feature fusion  │
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│   Transformer Decoder           │
│   • 6 decoder layers            │
│   • Multi-head self-attention   │
│   • Cross-attention to encoder  │
│   • 300 object queries          │
└─────────────────────────────────┘

Parameters: 32M
FLOPs: 92G
Inference: 18.7ms (RTX 3090)
```

### **Training Pipeline Architecture**

```
┌──────────────────────────────────────┐
│   Dataset: Weapon Detection          │
│   • Total: 5,149 images              │
│   • Train: 4,119 (80%)               │
│   • Val: 515 (10%)                   │
│   • Test: 515 (10%)                  │
│   • Classes: [handgun, knife, rifle] │
└────────┬─────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   Data Augmentation Pipeline         │
│   ┌────────────────────────────────┐ │
│   │ • Mosaic (4-image)             │ │
│   │ • Random affine transform      │ │
│   │ • HSV color jitter             │ │
│   │ • Random horizontal flip       │ │
│   │ • MixUp (alpha=0.2)            │ │
│   │ • CutMix (alpha=1.0)           │ │
│   └────────────────────────────────┘ │
└────────┬─────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   Mixed-Precision Training           │
│   • Precision: FP16 (AMP)            │
│   • Batch size: 16                   │
│   • Accumulation: 4 steps            │
│   • Effective batch: 64              │
└────────┬─────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   Optimizer & Scheduler              │
│   • Optimizer: AdamW                 │
│   • LR: 1e-3 → 1e-5                  │
│   • Scheduler: Cosine annealing      │
│   • Warmup: 3 epochs                 │
│   • Weight decay: 5e-4               │
└────────┬─────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   Loss Functions                     │
│   • Classification: BCE Loss         │
│   • Localization: CIoU Loss          │
│   • Objectness: BCE Loss             │
│   • Total: Weighted combination      │
└────────┬─────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│   Validation & Checkpointing         │
│   • Metric: mAP@0.5                  │
│   • Save best model                  │
│   • Early stopping: patience=50      │
└──────────────────────────────────────┘
```

### **Post-Training Optimization**

```
Trained PyTorch Model (.pt)
         ↓
    ┌────┴────┐
    ↓         ↓
┌────────┐  ┌─────────┐
│TorchScript│ │  ONNX   │
│  Export   │ │ Export  │
└────┬────┘  └────┬────┘
     ↓            ↓
┌────────────────────┐
│  Quantization      │
│  • INT8 (8-bit)    │
│  • FP16 (half)     │
│  • Dynamic range   │
└────────┬───────────┘
         ↓
┌────────────────────┐
│  Deployment Formats│
│  • .torchscript    │
│  • .onnx           │
│  • TensorRT (.trt) │
└────────────────────┘
```

---

## 🛠️ Tech Stack

### **Deep Learning Frameworks**
- **PyTorch 2.0+**: Core deep learning framework with `torch.compile()` for 2x speedup
- **TorchVision 0.15+**: Computer vision utilities and transforms
- **Ultralytics 8.0+**: YOLOv11 implementation and training pipeline
- **RT-DETR**: Baidu's real-time detection transformer

### **Computer Vision Libraries**
- **OpenCV 4.8+**: Image preprocessing and augmentation
- **Albumentations 1.3+**: Advanced augmentation pipeline
- **imgaug**: Additional augmentation strategies
- **PIL (Pillow)**: Image I/O operations

### **Model Optimization**
- **ONNX Runtime**: Cross-platform inference optimization
- **TensorRT 8.5+**: NVIDIA GPU acceleration
- **OpenVINO**: Intel CPU/VPU optimization
- **TorchScript**: JIT compilation for production

### **Evaluation & Metrics**
- **pycocotools**: COCO evaluation metrics (mAP, precision, recall)
- **torchmetrics**: PyTorch-native metric computation
- **scikit-learn**: Confusion matrix, classification reports

### **Experiment Tracking**
- **Weights & Biases (WandB)**: Experiment logging and visualization
- **TensorBoard**: Training curve monitoring
- **MLflow**: Model versioning and registry

### **Data Processing**
- **NumPy 1.24+**: Numerical computations
- **Pandas 2.0+**: Dataset analysis and statistics
- **Matplotlib & Seaborn**: Visualization

### **Development Tools**
- **CUDA 11.8+**: GPU acceleration
- **cuDNN 8.6+**: Deep learning primitives
- **Mixed Precision Training**: PyTorch AMP (Automatic Mixed Precision)

---

## 📊 Key Results & Performance Metrics

### **Benchmark Summary**

| Model | mAP50 | mAP50-95 | Precision | Recall | Latency (ms) | FPS | Parameters | Model Size |
|-------|-------|----------|-----------|--------|--------------|-----|------------|------------|
| **RT-DETR** | **83.0%** | **37.2%** | 81.4% | 79.8% | 18.7 | 53 | 32M | 126 MB |
| **YOLOv11** | 77.8% | 34.9% | **84.2%** | **82.1%** | **3.1** | **323** | 2.6M | **10.2 MB** |

### **Detailed Performance Analysis**

#### **RT-DETR Results**
```
Class-wise Performance:
├── Handgun:  mAP50=85.2%, Precision=83.1%, Recall=81.4%
├── Knife:    mAP50=82.4%, Precision=80.5%, Recall=78.9%
└── Rifle:    mAP50=81.4%, Precision=80.6%, Recall=79.1%

Inference Latency Breakdown (RTX 3090):
├── Preprocessing:     0.8 ms
├── Backbone:          6.2 ms
├── Transformer:      10.4 ms
├── Postprocessing:    1.3 ms
└── Total:            18.7 ms

Memory Usage:
├── Model weights:   126 MB
├── Activations:     412 MB
└── Total VRAM:      538 MB
```

#### **YOLOv11 Results**
```
Class-wise Performance:
├── Handgun:  mAP50=79.8%, Precision=85.4%, Recall=83.2%
├── Knife:    mAP50=77.2%, Precision=84.1%, Recall=82.6%
└── Rifle:    mAP50=76.4%, Precision=83.1%, Recall=80.5%

Inference Latency Breakdown (RTX 3090):
├── Preprocessing:     0.4 ms
├── Backbone:          1.2 ms
├── Neck:              0.9 ms
├── Head:              0.4 ms
├── Postprocessing:    0.2 ms
└── Total:             3.1 ms

Memory Usage:
├── Model weights:    10.2 MB
├── Activations:     124 MB
└── Total VRAM:      134 MB
```

### **Quantization Impact**

| Model | Precision | mAP50 | Latency (ms) | Model Size | Throughput |
|-------|-----------|-------|--------------|------------|------------|
| RT-DETR | FP32 | 83.0% | 18.7 | 126 MB | 53 FPS |
| RT-DETR | FP16 | 82.8% | 11.2 | 63 MB | 89 FPS |
| RT-DETR | INT8 | 81.4% | 8.9 | 32 MB | 112 FPS |
| YOLOv11 | FP32 | 77.8% | 3.1 | 10.2 MB | 323 FPS |
| YOLOv11 | FP16 | 77.7% | 2.1 | 5.1 MB | 476 FPS |
| YOLOv11 | INT8 | 76.9% | 1.6 | 2.6 MB | 625 FPS |

### **Hardware Benchmarks**

| Device | RT-DETR (FP32) | RT-DETR (INT8) | YOLOv11 (FP32) | YOLOv11 (INT8) |
|--------|----------------|----------------|----------------|----------------|
| RTX 3090 (GPU) | 18.7 ms | 8.9 ms | 3.1 ms | 1.6 ms |
| RTX 3060 (GPU) | 32.4 ms | 15.2 ms | 5.8 ms | 2.9 ms |
| NVIDIA Jetson Orin | 89.1 ms | 34.5 ms | 14.2 ms | 6.8 ms |
| Intel i9-12900K (CPU) | 412 ms | 198 ms | 76 ms | 38 ms |
| Raspberry Pi 4 (CPU) | 2,840 ms | N/A | 524 ms | 267 ms |

### **Training Hyperparameters**

```yaml
# RT-DETR Training Config
model: rtdetr-l
epochs: 300
batch_size: 16
img_size: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
box: 7.5
cls: 0.5
dfl: 1.5

# YOLOv11 Training Config
model: yolov11n
epochs: 300
batch_size: 16
img_size: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
box: 7.5
cls: 0.5
dfl: 1.5
```

### **Confusion Matrix Analysis**

**YOLOv11 Confusion Matrix (Normalized)**
```
              Predicted
Actual      Handgun  Knife  Rifle  Background
Handgun      0.832   0.034  0.021    0.113
Knife        0.041   0.826  0.028    0.105
Rifle        0.027   0.032  0.805    0.136
Background   0.018   0.015  0.012    0.955
```

**RT-DETR Confusion Matrix (Normalized)**
```
              Predicted
Actual      Handgun  Knife  Rifle  Background
Handgun      0.814   0.029  0.018    0.139
Knife        0.037   0.789  0.024    0.150
Rifle        0.023   0.028  0.791    0.158
Background   0.014   0.012  0.009    0.965
```

---

## 🚀 Installation & Usage

### **Prerequisites**
```bash
Python 3.9+
CUDA 11.8+ (for GPU training)
cuDNN 8.6+
8GB+ GPU VRAM (RTX 3060 or better recommended)
```

### **Installation**
```bash
# Clone repository
git clone https://github.com/Sachin-Saailesh/real-time-weapon-detection-benchmark.git
cd real-time-weapon-detection-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install Ultralytics (for YOLOv11)
pip install ultralytics

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Dataset Preparation**

```bash
# Download weapon detection dataset
# Expected structure:
weapon-dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

# Convert annotations to YOLO format if needed
python scripts/convert_to_yolo.py \
  --input annotations.json \
  --output labels/ \
  --format coco

# Generate dataset YAML
cat > data/weapon.yaml << EOF
path: /path/to/weapon-dataset
train: images/train
val: images/val
test: images/test

nc: 3  # number of classes
names: ['handgun', 'knife', 'rifle']
EOF
```

### **Training**

#### **Train YOLOv11**
```bash
# Train from scratch
python train_yolo.py \
  --model yolov11n \
  --data data/weapon.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project runs/yolov11 \
  --name weapon-detection

# Resume training
python train_yolo.py \
  --model runs/yolov11/weapon-detection/weights/last.pt \
  --data data/weapon.yaml \
  --resume

# Multi-GPU training
python train_yolo.py \
  --model yolov11n \
  --data data/weapon.yaml \
  --epochs 300 \
  --batch 32 \
  --device 0,1,2,3
```

#### **Train RT-DETR**
```bash
# Train RT-DETR
python train_rtdetr.py \
  --model rtdetr-l \
  --data data/weapon.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project runs/rtdetr \
  --name weapon-detection

# With mixed precision training
python train_rtdetr.py \
  --model rtdetr-l \
  --data data/weapon.yaml \
  --amp  # Enable automatic mixed precision
```

### **Evaluation**

```bash
# Evaluate YOLOv11
python evaluate.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --data data/weapon.yaml \
  --task test \
  --device 0

# Evaluate RT-DETR
python evaluate.py \
  --model runs/rtdetr/weapon-detection/weights/best.pt \
  --data data/weapon.yaml \
  --task test \
  --device 0

# Generate COCO metrics
python coco_eval.py \
  --gt data/test/annotations.json \
  --pred results/predictions.json \
  --output results/coco_metrics.json
```

### **Inference**

#### **Single Image**
```bash
# YOLOv11 inference
python detect.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --source test.jpg \
  --conf 0.25 \
  --iou 0.45 \
  --device 0 \
  --save

# RT-DETR inference
python detect.py \
  --model runs/rtdetr/weapon-detection/weights/best.pt \
  --source test.jpg \
  --conf 0.25 \
  --device 0 \
  --save
```

#### **Video Stream**
```bash
# Webcam
python detect.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --source 0 \
  --view-img

# RTSP stream
python detect.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --source rtsp://192.168.1.100:554/stream \
  --view-img
```

#### **Batch Processing**
```bash
# Process directory
python detect.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --source test_images/ \
  --save-txt \
  --save-conf
```

### **Model Export**

#### **Export to ONNX**
```bash
# YOLOv11 to ONNX
python export.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --dynamic  # Dynamic input shapes

# RT-DETR to ONNX
python export.py \
  --model runs/rtdetr/weapon-detection/weights/best.pt \
  --format onnx \
  --opset 12 \
  --simplify  # Simplify ONNX graph
```

#### **Export to TorchScript**
```bash
# JIT tracing
python export.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --format torchscript \
  --optimize  # Apply TorchScript optimizations
```

#### **Export to TensorRT**
```bash
# FP16 precision
python export.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --format engine \
  --half  # FP16 inference

# INT8 quantization (requires calibration data)
python export.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --format engine \
  --int8 \
  --data data/weapon.yaml
```

### **Benchmarking**

```bash
# Latency benchmark
python benchmark.py \
  --model runs/yolov11/weapon-detection/weights/best.pt \
  --imgsz 640 \
  --device 0 \
  --iterations 1000

# Compare models
python compare_models.py \
  --yolo runs/yolov11/weapon-detection/weights/best.pt \
  --rtdetr runs/rtdetr/weapon-detection/weights/best.pt \
  --data data/weapon.yaml \
  --output comparison_report.pdf
```

---

## 📈 Advanced Usage

### **Hyperparameter Tuning**
```bash
# Grid search
python tune.py \
  --model yolov11n \
  --data data/weapon.yaml \
  --params configs/grid_search.yaml \
  --trials 50

# Bayesian optimization
python tune.py \
  --model yolov11n \
  --data data/weapon.yaml \
  --optimizer bayesian \
  --trials 100
```

### **Custom Augmentation**
```python
# config/augmentation.yaml
mosaic: 1.0
mixup: 0.2
copy_paste: 0.3
degrees: 10.0
translate: 0.2
scale: 0.9
shear: 2.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

### **Production Deployment (FastAPI)**
```python
# deploy/api.py
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("best.pt")

@app.post("/detect")
async def detect_weapons(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img, conf=0.25)
    detections = results[0].boxes.data.tolist()
    
    return {"detections": detections}

# Run: uvicorn deploy.api:app --host 0.0.0.0 --port 8000
```

---

## 📚 Project Structure
```
real-time-weapon-detection-benchmark/
├── data/
│   ├── weapon.yaml           # Dataset configuration
│   └── scripts/
│       └── convert_to_yolo.py
├── models/
│   ├── yolov11/              # YOLOv11 configs
│   └── rtdetr/               # RT-DETR configs
├── train_yolo.py             # YOLOv11 training script
├── train_rtdetr.py           # RT-DETR training script
├── evaluate.py               # Evaluation script
├── detect.py                 # Inference script
├── export.py                 # Model export utilities
├── benchmark.py              # Performance benchmarking
├── compare_models.py         # Model comparison
├── coco_eval.py              # COCO metrics evaluation
├── deploy/
│   ├── api.py                # FastAPI deployment
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── visualization.ipynb
├── results/
│   ├── plots/
│   └── metrics/
├── requirements.txt
└── README.md
```

---

## 🎓 Key Findings & Recommendations

### **Model Selection Guidelines**

**Choose RT-DETR when:**
- Accuracy is critical (e.g., security applications)
- Hardware can handle 18ms latency
- Budget allows for higher compute costs
- Dealing with small or occluded weapons
- Need better generalization to unseen data

**Choose YOLOv11 when:**
- Real-time performance is mandatory (<5ms)
- Deploying to edge devices (Jetson, mobile)
- Budget/power constraints exist
- High throughput required (>200 FPS)
- Model size must be minimal (<20MB)

### **Production Deployment Checklist**
- [ ] Quantize model to INT8 for edge deployment
- [ ] Implement NMS post-processing optimization
- [ ] Add confidence threshold calibration
- [ ] Set up model monitoring (drift detection)
- [ ] Implement alert system for high-confidence detections
- [ ] Add explainability (Grad-CAM visualizations)
- [ ] Conduct adversarial robustness testing
- [ ] Implement A/B testing framework

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv11 implementation
- **Baidu Research** for RT-DETR architecture
- **COCO Dataset** for evaluation metrics
- **PyTorch Team** for deep learning framework

---

## 📬 Contact

**Sachin Saailesh Jeyakkumaran**
- Email: sachin.jeyy@gmail.com
- LinkedIn: [linkedin.com/in/sachin-saailesh](https://linkedin.com/in/sachin-saailesh)
- Portfolio: [sachinsaailesh.com](https://sachinsaailesh.com)

---

**Built for production-grade computer vision applications**
