# 🩺 Pneumonia Detection from Chest X-rays using CheXNet (DenseNet121)

---

# 📌 Project Overview

This project implements an AI-powered pneumonia detection system using Deep Learning and Chest X-ray images.

The model is based on the **CheXNet framework**, which uses a **DenseNet121 Convolutional Neural Network** trained through **transfer learning** to classify chest X-rays into:

- NORMAL
- PNEUMONIA

The project also integrates:

- Grad-CAM visualization
- Explainable AI
- Model evaluation metrics
- Modular training pipeline

This system demonstrates how deep learning can assist radiologists in automated medical diagnosis.

---

# 🎯 Objectives

The main goals of this project are:

- Detect pneumonia from chest X-rays
- Use transfer learning for efficient training
- Improve model interpretability using Grad-CAM
- Build a modular research-oriented pipeline
- Evaluate model performance using medical AI metrics

---

# 🧠 Deep Learning Concepts Used

This project includes implementation of:

- Convolutional Neural Networks (CNN)
- DenseNet121 Architecture
- Transfer Learning
- Binary Classification
- Forward Propagation
- Backpropagation
- Gradient Descent
- Binary Cross Entropy Loss
- Adam Optimizer
- Data Augmentation
- Grad-CAM Explainability

---

# 🏥 About Pneumonia

Pneumonia is a respiratory disease that affects the lungs and can cause inflammation and fluid accumulation in the alveoli.

Chest X-rays are commonly used to diagnose pneumonia by identifying:

- Lung opacities
- Infiltrates
- Abnormal textures
- White patch regions

Early detection is important for timely medical treatment.

---

# 📂 Dataset

Dataset Source:
Kaggle Chest X-ray Pneumonia Dataset

Dataset Structure:

dataset/
└── chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/

---

# 📊 Dataset Statistics

- Total Images: ~5863
- Classes:
  - NORMAL
  - PNEUMONIA

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Pretrained Models & Image Processing |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation Metrics |
| Google Colab | GPU Training Environment |

---

# 🧱 Project Structure

project/
│
├── dataset/
│
├── phase1.py
├── phase2.py
├── phase3.py
├── phase4.py
│
├── train.py
├── inference.py
├── gradcam.py
│
├── best_pneumonia_model.pth
│
└── README.md

---

# 🔥 Phase-wise Implementation

---

# 📌 Phase 1: Data Loading & Preprocessing

File:
phase1.py

This phase performs:

- Image loading
- Data augmentation
- Tensor conversion
- Normalization
- DataLoader creation

## Transformations Applied

- Resize → 224×224
- Random Rotation
- Random Horizontal Flip
- Tensor Conversion
- ImageNet Normalization

## Why Preprocessing is Important

Preprocessing helps:

- stabilize training
- improve convergence
- reduce overfitting
- standardize inputs

---

# 📌 Phase 2: Model Architecture

File:
phase2.py

The model uses:

DenseNet121 pretrained on ImageNet.

## Why DenseNet121?

DenseNet provides:

- feature reuse
- better gradient flow
- fewer parameters
- strong medical imaging performance

---

# 🔥 Dense Connectivity Concept

Each DenseNet layer receives features from all previous layers.

Traditional CNN:
Layer → Layer

DenseNet:
All previous layers → Current layer

This improves:

- feature propagation
- gradient flow
- learning efficiency

---

# 📌 Transfer Learning

The pretrained DenseNet already learned:

- edges
- shapes
- textures
- visual patterns

from millions of images.

Only the final classifier is modified for pneumonia classification.

---

# 📌 Custom Classifier

Original DenseNet classifier:
1000 ImageNet classes

Replaced with:

- Dropout
- Fully Connected Layer
- Sigmoid Activation

Final output:
Pneumonia probability

---

# 📌 Sigmoid Activation

Sigmoid converts logits into probabilities:

Output Range:
0 → 1

Example:
0.92 = 92% pneumonia probability

---

# 📌 Phase 3: Training

File:
phase3.py

Training phase performs:

- Forward propagation
- Loss computation
- Backpropagation
- Weight optimization

---

# 🔥 Forward Propagation

Image passes through:

Conv Layers
↓
Dense Blocks
↓
Feature Extraction
↓
Classifier
↓
Prediction

---

# 🔥 Loss Function

Binary Cross Entropy Loss (BCELoss)

Purpose:
Measure prediction error.

Lower loss means:
better prediction.

---

# 🔥 Backpropagation

Backpropagation computes gradients using:

Chain Rule from Calculus

Gradients determine:

how much each parameter contributed to the prediction error.

---

# 🔥 Optimizer

Adam Optimizer is used because it:

- adapts learning rates
- converges faster
- handles sparse gradients efficiently

---

# 📌 Model Saving

The best model is saved as:

best_pneumonia_model.pth

This file stores:

- learned weights
- convolution kernels
- classifier parameters

---

# 📌 Phase 4: Evaluation

File:
phase4.py

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 📊 Evaluation Metrics

## Accuracy

Overall prediction correctness.

Formula:

Accuracy = Correct Predictions / Total Predictions

---

## Precision

Measures false positive quality.

Formula:

Precision = TP / (TP + FP)

---

## Recall

Measures ability to detect pneumonia cases.

Formula:

Recall = TP / (TP + FN)

High recall is extremely important in medical AI.

---

## F1-score

Balanced metric combining:

- precision
- recall

---

# 📌 Confusion Matrix

Shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

Useful for medical diagnosis analysis.

---

# 🔥 Explainable AI using Grad-CAM

File:
gradcam.py

Grad-CAM generates heatmaps showing:

where the model focused during prediction.

---

# 🧠 Why Explainability Matters

Medical AI should not behave like a black box.

Grad-CAM helps doctors verify:

whether the model focused on clinically relevant lung regions.

---

# 🔥 Grad-CAM Working

1. Forward Pass
2. Compute Gradients
3. Global Average Pooling
4. Weighted Feature Maps
5. Heatmap Generation

---

# 📌 Inference Phase

File:
inference.py

This phase performs:

- Load trained model
- Load test X-ray
- Preprocess image
- Predict pneumonia probability
- Generate Grad-CAM heatmap
- Display result

---

# 🔥 Prediction Flow

Input X-ray
↓
DenseNet121
↓
Probability Output
↓
Classification
↓
Grad-CAM Visualization

---

# 📌 How to Run the Project

---

# 1️⃣ Install Dependencies

pip install torch torchvision matplotlib opencv-python scikit-learn pillow

---

# 2️⃣ Run Training

python train.py

This generates:

best_pneumonia_model.pth

---

# 3️⃣ Run Inference

Place a test image:

test_image.jpeg

Then run:

python inference.py

---

# 📌 Expected Output

Prediction:
PNEUMONIA

Confidence:
0.91

Also displays:
Grad-CAM heatmap.

---

# 📌 Hardware Requirements

Recommended:

- GPU (NVIDIA CUDA)
- Google Colab GPU Runtime

Minimum:

- CPU
- 8GB RAM

---

# 📌 Why GPU is Important

Deep learning performs millions of matrix multiplications.

GPU enables:

- parallel computation
- faster training
- efficient tensor operations

---

# 📌 Challenges Faced

- Dataset imbalance
- Overfitting
- Limited computational resources
- Medical image variability
- Explainability requirements

---

# 📌 Limitations

This model may fail because of:

- noisy X-rays
- unseen hospital distributions
- scanner differences
- demographic bias

This project is educational/research-oriented and not clinically approved.

---

# 🚀 Future Improvements

Possible future enhancements:

- Vision Transformers
- EfficientNet
- Attention Mechanisms
- U-Net Segmentation
- Grad-CAM++
- Multi-class disease classification
- Web deployment
- Mobile deployment

---

# 📌 Research Concepts Learned

This project demonstrates understanding of:

- CNN Architecture
- Medical Image Analysis
- Transfer Learning
- Explainable AI
- Model Evaluation
- Deep Learning Optimization
- Computer Vision

---

# 📌 Conclusion

This project successfully demonstrates an end-to-end deep learning pipeline for pneumonia detection using chest X-rays.

The DenseNet121-based architecture achieves strong classification performance while Grad-CAM improves interpretability and trustworthiness.

The project combines:

- medical imaging
- deep learning
- explainable AI
- transfer learning

into a complete AI healthcare solution.

---

# 📚 References

- CheXNet Paper
- DenseNet Research Paper
- PyTorch Documentation
- Kaggle Chest X-ray Dataset
- Grad-CAM Research Paper

---
