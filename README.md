# Road Traffic Monitoring and Detection System

Real-time traffic monitoring using YOLO algorithm with LSTM/GRU forecasting for intelligent transportation systems.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)

## 🎯 Overview

Advanced traffic monitoring system combining YOLO-based vehicle detection with LSTM/GRU neural networks for traffic flow prediction. Processes real-time video streams for vehicle detection, tracking, and forecasting up to 7 days.

## 🛠️ Technology Stack

**Detection & Tracking:**

- **YOLO V3** - Real-time object detection
- **OpenCV** - Computer vision and image processing

**Forecasting Models:**

- **LSTM** (Long Short-Term Memory) - 24-hour traffic prediction
- **GRU** (Gated Recurrent Unit) - 7-day traffic forecasting

**Frameworks:**

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy, Matplotlib, Pandas

**Development:**

- Jupyter Notebook
- Google Colab (model training)

## 🏗️ System Architecture

```
Video Stream → Preprocessing → YOLO Detection → Vehicle Tracking → 
Data Extraction → LSTM/GRU Models → Traffic Prediction
```

### Pipeline Components

1. **Data Acquisition**: Traffic cameras and drone footage
1. **Preprocessing**: Frame extraction, noise reduction, ROI identification
1. **Object Detection**: YOLO algorithm for vehicle identification
1. **Tracking**: Centroid-based vehicle tracking across frames
1. **Data Analysis**: Traffic flow, density, and congestion patterns
1. **Forecasting**: RNN-based prediction using LSTM/GRU

![Traffic Detection](https://via.placeholder.com/800x300/4A90E2/ffffff?text=YOLO+Vehicle+Detection)

## 📊 Performance Metrics

### Detection Performance

- **Detection Accuracy**: 85-90%
- **Processing Speed**: Real-time (25-30 FPS)
- **YOLO Input Size**: 416×416×3
- **Detection Layers**: 3 scales (13×13, 26×26, 52×52)

### Forecasting Performance

|Model   |MAE  |MSE   |RMSE |MAPE|Forecast Period|
|--------|-----|------|-----|----|---------------|
|**LSTM**|0.032|0.0018|0.042|4.2%|24 hours       |
|**GRU** |0.029|0.0015|0.039|3.8%|7 days         |

![LSTM Prediction]
![image](https://github.com/user-attachments/assets/5789c0df-b842-4c09-bfae-5a156fd912e0)


![GRU Forecast]
<img width="1058" height="742" alt="image" src="https://github.com/user-attachments/assets/56150059-cd66-4fc3-8eaf-89cc636a12f0" />

## 🔍 Key Algorithms

### YOLO (You Only Look Once)

- **Single-pass detection** - Processes entire image in one forward pass
- **Grid-based approach** - Divides image into S×S grid cells
- **Bounding box regression** - Predicts boxes with confidence scores
- **NMS filtering** - Removes duplicate detections using IoU threshold

### LSTM vs GRU

|Feature            |LSTM                     |GRU                 |
|-------------------|-------------------------|--------------------|
|**Gates**          |3 (input, forget, output)|2 (reset, update)   |
|**Parameters**     |More complex             |Fewer parameters    |
|**Training Time**  |Slower                   |Faster              |
|**Memory Handling**|Long-term dependencies   |Efficient sequential|
|**Best For**       |Short-term (24h)         |Long-term (7 days)  |

## 📁 Dataset

**Video Sources:**

- Pragati Tunnel footage (19 seconds)
- NH-24 highway drone footage (14 seconds)
- Real-time traffic surveillance cameras

**Preprocessing Steps:**

1. Frame extraction at 30 FPS
1. Resolution normalization (416×416)
1. Gaussian noise reduction
1. Region of Interest (ROI) extraction
1. Background subtraction

## ⚙️ System Requirements

**Hardware:**

- CPU: Intel Core i7 or equivalent
- GPU: NVIDIA GPU with CUDA support (recommended)
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB for models and datasets

**Software:**

- Python 3.8+
- CUDA 10.1+ (GPU acceleration)
- cuDNN 7.6+

## 🎯 Key Features

✅ Real-time vehicle detection using YOLO  
✅ Multi-vehicle tracking with centroid algorithm  
✅ 24-hour traffic flow prediction (LSTM)  
✅ 7-day traffic forecasting (GRU)  
✅ Anomaly detection (accidents, violations)  
✅ Traffic congestion analysis  
✅ Scalable for multiple camera streams

## 📈 Applications

- **Traffic Management**: Optimize signal timings and lane configurations
- **Safety Monitoring**: Detect accidents and violations in real-time
- **Infrastructure Planning**: Informed decisions based on traffic patterns
- **Smart Cities**: Integration with urban IoT platforms
- **Law Enforcement**: Automated traffic violation detection
- **Environmental Impact**: Reduce congestion and emissions

## 🚧 Limitations

1. **Data Quality**: Requires high-quality, stable video streams
1. **Lighting Conditions**: Performance degrades in low-light scenarios
1. **Training Time**: LSTM/GRU models require extensive training on Google Colab
1. **Computational Cost**: GPU recommended for real-time processing
1. **Generalization**: Models may need retraining for different locations
1. **Weather Sensitivity**: External factors affect prediction accuracy

## 📖 Key References

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement
1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
1. Cho, K., et al. (2014). Gated Recurrent Unit
1. Girshick, R., et al. (2014). Rich Feature Hierarchies for Object Detection

## 📝 Project Details

**Date:** March 2024  
**Status:** Completed  

-----
