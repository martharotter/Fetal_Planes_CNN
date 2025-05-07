# Fetal Planes CNN

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep learning model using convolutional neural networks to identify and classify fetal ultrasound planes for automated medical image analysis.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

Maternal-fetal ultrasound imaging is an essential part of prenatal care, allowing early detection of fetal abnormalities and monitoring of developmental health. However, the complexity and variability of ultrasound images pose significant challenges for automated analysis. This project presents a deep learning approach to enhance the analysis of a maternal-fetal ultrasound image data set for screening and identification. 

## Dataset

The dataset for this source comes from the project "FETAL_PLANES_DB: Common maternal-fetal ultrasound images"

**Dataset Source:** [FETAL_PLANES_DB: Common maternal-fetal ultrasound images](https://zenodo.org/records/3904280#.Y_dXxuzP3UK)

### Data Description
- A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician. 
- 12,600 images in total
- 6 classes: four of the most widely used fetal anatomical planes (abdomen, brain, femur and thorax), the mother's cervix, and a general category to include other less common image plane. Fetal brain images are further categorized into the 3 most common fetal brain planes: trans-thalamic, trans-cerebellum, trans-ventricular

### Download Instructions
```bash
1. Download zip file from https://zenodo.org/records/3904280#.Y_dXxuzP3UK
2. Copy file to the project directory.
2. Unzip file.
```

## Requirements

```
python>=3.12
tensorflow>=2.16.2
keras>=3.9.0
numpy>=1.26.4
pandas>=2.2.3
matplotlib>=3.10.1
scikit-learn>=1.6.1
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/martharotter/Fetal_Planes_CNN.git
cd Fetal_Planes_CNN
```

2. Copy images folder to project directory
```bash
mv -R $HOME/Downloads/FETAL_PLANES_ZENODO/ .
```

3. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# or conda create -n deeppy12 python=3.12
source venv/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Fetal_Planes_CNN/
├── FETAL_PLANES_ZENODO/
│   ├── Images/                   # Raw unprocessed data
│   ├── FETAL_PLANES_DB_data.csv  # CSV detailing images
│   └── FETAL_PLANES_DB_data.xlsx # XLSX detailing images
├── checkpoints/                  # Folder of keras checkpoint files
├── hparams.yaml                  # Hyperparameter configurations
├── logs/                         # Folder of log files
├── src/                          # Source code
│   ├── __init__.py
│   ├── main.py
│   ├── main_local.py
│   ├── util/                 # Utility scripts
│   │   ├── __init__.py
│   │   ├── logging_setup.py
│   │   ├── plotting.py
│   │   ├── prepare_images.py
│   │   └── split_training_set.py
│   ├── models/               # Model architecture definitions
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── inception.py
│   │   └── resnet50.py
├── config/                   # Configuration files
├── scripts/                  # Utility scripts
├── README.md                 # This file
└── requirements.txt          # Required packages
```

## Usage

### Training with baseline CNN

```bash
# Train the model with default parameters
python main_local.py --model="CNN"
```

### Training with ResNet50 model

```bash
# Train the ResNet50 model
python main_local.py --model="RESNET50"
```

### Training with Inception model

```bash
# Train the Inception model
python main_local.py --model="INCEPTION"
```

## Model Architecture

The baseline convolutional neural network is as follows:

- Model type: custom, deep CNN derived from tensorflow.keras.models.Sequential
- Layer structure:
  - input layer accepts images of shape (224, 224, 3): standard RGB image size
  - 3 convolutional feature extraction blocks: low, mid, and high-level feature extraction, where each block doubles the number of filters capturing progressively more abstract features and halves the spatial dimensions via Max Pooling, reducing computation
  - Classifier block with a softmax final output layer for classification
- Key parameters
  - Convolutional Layer Parameters:
    - Number of filters per layer: block 1 - 64; block 2 - 128; block 3 - 256
    - Kernal size: (3, 3)
    - Activation function: relu
  - Batch normalization layers
  - Pooling layers
    - MaxPooling2D(pool_size=(2,2))
  - Fully connected (dense) layers
    - Dense(512) and Dense(256)
    - Dropout rates: 0.5 and 0.3
  - Output layer
    - Dense(self.num_classes, activation='softmax')
  - Input shape
    - (224, 224, 3)

```
Input (224x224x3) → Conv2D → BatchNorm → MaxPool → ... → Dense → Output
```

## Results

The best results obtained from the project are as follows:

| Metric | Value |
|--------|-------|
| Accuracy | 0.XX |
| Precision | 0.XX |
| Recall | 0.XX |
| F1-Score | 0.XX |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thank you to Professor Guenole Silvestre, TA Conor O'Sullivan and assistant TAs Cheng Xu, Duc-Anh Nguyen and Jiaming Xu for their help throughout the semester 
- Thank you to the team who created the dataset and wrote the initial paper: ["Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes"](https://doi.org/10.1038/s41598-020-67076-5)
