# Pneumonia Detection from Chest X-Rays using CNN

This repository contains a deep learning project for detecting pneumonia from pediatric chest X-ray images using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras.

## 📌 Project Overview

Pneumonia is an inflammatory condition of the lung primarily affecting the small air sacs known as alveoli. Chest X-ray imaging is the most commonly used diagnostic tool. This project implements a custom CNN model capable of classifying X-ray images into two classes:
- **NORMAL**: Healthy chest X-ray
- **PNEUMONIA**: Chest X-ray displaying bacterial or viral pneumonia

---

## 🏗️ Model Architecture

The architecture consists of multiple convolutional blocks designed to extract hierarchical spatial features from $256 \times 256 \times 3$ input images:
- **Conv2D + BatchNormalization + MaxPool2D** layers with increasing filter depth (32, 64, 64, 128, etc.)
- **Dropout** layers to prevent overfitting
- **Flatten** & **Dense** layers with ReLU activation
- **Output layer**: Dense layer with Sigmoid activation for binary classification

---

## 📊 Dataset

The dataset used is the benchmark **Chest X-Ray Images (Pneumonia)** dataset by Kermany et al., available on Kaggle:
- **Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**:
  ```text
  dataset/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
  ```

> *Note: Due to file size constraints, raw image files in `dataset/` are excluded from this repository via `.gitignore`. Please download the dataset directly from Kaggle and place it in the `dataset/` folder.*

---

## 💾 Pre-trained Model Weights

The trained model weights are saved in:
- `models/pneumonia_model.h5`

> *Note: The model weights file (~102.7 MB) is tracked using [Git Large File Storage (Git LFS)](https://git-lfs.github.com/).*

To pull the weights after cloning:
```bash
git lfs install
git lfs pull
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Sidd-art-h1/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```

### 2. Install dependencies
```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python
```

### 3. Run the notebook
Open and run `Pneumonia_detection.ipynb` using Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook Pneumonia_detection.ipynb
```
