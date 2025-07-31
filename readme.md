# Face Mask Detection using Transfer Learning 😷

A deep learning-based computer vision project that classifies whether individuals are wearing face masks correctly, incorrectly, or not at all. This solution leverages transfer learning with MobileNetV2 and ResNet50 to achieve real-time mask detection, addressing safety concerns in public and healthcare settings.

## 🚀 Overview

The system uses pre-trained convolutional neural networks (CNNs) to detect face masks from images and live video streams. Two architectures—MobileNetV2 and ResNet50—were fine-tuned and compared based on performance and efficiency. Real-time video classification was implemented using OpenCV.

### 🧠 Key Features

Binary image classification: Mask vs No Mask

Real-time webcam inference with bounding boxes

Data augmentation and preprocessing

Model training using Transfer Learning

Comparative analysis of MobileNetV2 and ResNet50

### 🛠️ Tech Stack

| Category            | Tools / Libraries                                 | Purpose                                 |
|---------------------|---------------------------------------------------|-----------------------------------------|
| Language            | Python 3.8+                                       | Core programming                        |
| Deep Learning       | TensorFlow, Keras                                 | Model building & training               |
| Computer Vision     | OpenCV                                            | Real-time face detection & video stream |
| Data Handling       | NumPy, Scikit-learn                               | Preprocessing, evaluation               |
| Visualization       | Matplotlib                                        | Training metrics and plots              |
| Annotation Tool     | LabelImg                                          | Manual labeling of custom images        |
| Environment & Tools | Jupyter Notebook, VS Code                         | Development & experimentation           |


### 🗂️ Directory Structure

```text
face-mask-detection/
├── src/              # Model training and inference scripts
├── models/           # Saved MobileNetV2 and ResNet50 models (.h5)
├── plots/            # Training graphs (accuracy/loss)
├── dataset/          # Image dataset (not included in repo)
├── requirements.txt  # Python dependencies
├── .gitignore        # Ignored files/folders
└── README.md         # Project documentation


### 🏋️‍♂️ Model Training

🔹 MobileNetV2

python src/train_mask_detector.py \
  --dataset dataset \
  --model models/mask_detector_mobilenet.model \
  --plot plots/mobilenet_plot.png

🔹 ResNet50

python src/train_mask_detector_resnet.py \
  --dataset dataset \
  --model models/mask_detector_resnet.model \
  --plot plots/resnet_plot.png

### 🎥 Real-Time Inference

After training, use the script below to activate your webcam and perform mask detection:

python src/detect_mask_video.py


### 📈 Model Evaluation

| Model       | Training Time | Accuracy       | Inference Speed     | Remarks                            |
|-------------|---------------|----------------|----------------------|------------------------------------|
| MobileNetV2 | ⏱️ Fast        | ✅ High         | ⚡ Real-time capable | Lightweight and optimized for speed |
| ResNet50    | 🐢 Slower      | ⚠️ Moderate     | 🐌 Slower           | More robust but heavier to deploy   |


### 📜 Dataset Sources

Kaggle (open-source datasets)

GettyImages (licensed samples)

Custom-labeled images using LabelImg

Preprocessing included outlier detection and cleaning

