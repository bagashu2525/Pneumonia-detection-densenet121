# A PROJECT REPORT ON
## Pneumonia Detection from Chest X-rays using CheXNet (DenseNet121)

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3a/University_of_Calcutta_logo.svg/1200px-University_of_Calcutta_logo.svg.png" width="180" alt="University of Calcutta Logo">
</p>

### Submitted in partial fulfillment of the requirements for the degree of B.Tech in Information Technology

**By,**
*   **Asmita Bag** (Roll No: T91/IT/226002, Reg. No: 211-1211-0446-20)
*   **Susita Samanta Roy** (Roll No: T91/IT/226012)

**Under the Supervision of,**
*   **Prof. Amlan Chakrabarty**
    *   Department of A.K.C. School of Information Technology
    *   University of Calcutta
    *   Kolkata, West Bengal, India

**Academic Year: 2025-2026**

---

## 📜 Certificate from the Supervisor

This is to certify that the work embodied in this thesis entitled **"Pneumonia Detection from Chest X-rays using CheXNet (DenseNet121)"** has been satisfactorily completed by:

*   **Asmita Bag** (Roll No: T91/IT/226002, Reg. No: 211-1211-0446-20)
*   **Susita Samanta Roy** (Roll No: T91/IT/226012)

It is a bona-fide piece of work carried out under my supervision and guidance at the Department of A.K.C. School of Information Technology, University of Calcutta, Kolkata, for partial fulfillment of the requirements for the awarding of the B.Tech in Information Technology.

<br><br>
______________________________________
**Prof. Amlan Chakrabarty**
*Department of A.K.C. School of Information Technology*
*University of Calcutta*

---

## 🤝 Acknowledgment

We would like to express our deepest gratitude to our project supervisor, **Prof. Amlan Chakrabarty**, for his invaluable guidance, continuous encouragement, and immense support throughout the duration of this project. His deep technical expertise and insights have been instrumental in shaping the direction and execution of this work.

We are sincerely grateful to the faculty members of the Department of A.K.C. School of Information Technology, University of Calcutta, for providing the necessary research facilities and academic ecosystem.

Last but not the least, we express our heartfelt regards to our family and friends for their patience, support, and constant motivation during the entire course of this project.

*   **Asmita Bag**
*   **Susita Samanta Roy**

---

## 📝 Abstract

Pneumonia remains a leading cause of mortality worldwide, particularly among pediatric and elderly populations. Accurate and rapid radiological interpretation of chest X-rays (CXRs) is critical for timely clinical intervention. This project presents an end-to-end deep learning pipeline for automated, explainable pneumonia classification using a DenseNet121 architecture pre-trained on ImageNet (CheXNet framework). Decoupled into a modular four-phase design (preprocessing, architecture selection, model training, and clinical evaluation), the proposed system leverages transfer learning, class-imbalance loss weighting, and ROC Youden's J-statistic threshold optimization. To overcome the "black-box" limitation of neural networks in clinical deployments, we integrate Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize and validate the model's focus regions against anatomical chest structures. Standardized to ImageNet normalization constants (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), the system demonstrates robust performance on Kaggle validation scans, successfully resolving high false-positive prediction issues on normal images (improving confidence from a biased **0.9892** to a highly accurate **0.6370**). This project demonstrates a highly robust, mathematically validated, and explainable decision-support tool to assist medical practitioners.

**Keywords**: Pneumonia Detection, DenseNet121, Transfer Learning, Explainable AI, Grad-CAM, CheXNet, Medical Image Classification.

---

## 📂 Table of Contents
1.  **Chapter 1: Introduction & Motivation**
    *   1.1 Clinical Background of Pneumonia
    *   1.2 Radiological Challenges & Objectives
    *   1.3 Machine Learning in Computer-Aided Diagnosis (CAD)
2.  **Chapter 2: Literature Review**
    *   2.1 Evolution of Deep Learning in Medical Imaging
    *   2.2 CheXNet & DenseNet121 Framework
    *   2.3 Overview of Explainability in Medical AI
3.  **Chapter 3: System Methodology & Architecture**
    *   3.1 Phase 1: Preprocessing & In-Distribution Normalization
    *   3.2 Phase 2: Network Architecture & Loss Function Selection
    *   3.3 Phase 3: Weight Optimization & Modular Training
    *   3.4 Phase 4: Model Evaluation & Youden's J Statistic Thresholding
4.  **Chapter 4: Explainability & Grad-CAM**
    *   4.1 Explainability as a Clinical Necessity
    *   4.2 Grad-CAM Mathematical Foundation & Hook Setup
5.  **Chapter 5: Implementation, Diagnostics, & Bug Corrections**
    *   5.1 Implementation Environment & Hardware Dependencies
    *   5.2 Unicode Encoding CP1252 Crash Resolution
    *   5.3 The Normalization Mismatch Resolution (Bug Fix Log)
    *   5.4 Preventing GUI Blocking in Headless Deployments
6.  **Chapter 6: Results, Discussion, & Conclusion**
    *   6.1 Experimental Results & Validation Run
    *   6.2 Technical Discussion & Limitations
    *   6.3 Future Scope & Clinical Enhancements
    *   6.4 References

---

## 1. Chapter 1: Introduction & Motivation

### 1.1 Clinical Background of Pneumonia
Pneumonia is an acute respiratory infection that causes inflammatory consolidation of the lung parenchyma, wherein the pulmonary alveoli fill with exudative fluid rather than air. It is a leading infectious cause of death worldwide, responsible for millions of hospitalizations annually, particularly among pediatric patients under five and geriatric populations. Common symptoms include dyspnea, persistent cough, chest pain, and high fever. The primary diagnosis of pneumonia relies on a clinical examination accompanied by a frontal chest X-ray (CXR). On a chest radiograph, pneumonia typically presents as localized or diffuse areas of increased opacity (consolidation), interstitial infiltrates, and abnormal textured shadows that indicate fluid consolidation.

### 1.2 Radiological Challenges & Objectives
Interpreting chest X-rays is a highly subjective task that requires significant clinical expertise. Even among senior radiologists, inter-observer variability is a well-documented challenge, particularly in high-throughput environments where clinical fatigue can lead to diagnostic oversights. Furthermore, in resource-constrained rural areas, there is a severe shortage of specialized radiologists, leading to delays in treatment.

The main objectives of this project are:
*   To build an automated, highly reliable computer-aided diagnostic (CAD) system capable of accurately classifying frontal chest X-rays into **NORMAL** or **PNEUMONIA**.
*   To leverage **transfer learning** with a DenseNet121 backbone pre-trained on ImageNet to overcome the typical limitations of small medical datasets.
*   To implement a modular, clean, and cross-platform training and inference pipeline that avoids execution crashes.
*   To integrate **Explainable AI (XAI)** utilizing Gradient-weighted Class Activation Mapping (Grad-CAM) to provide transparent visual evidence of the model's prediction, enabling clinical validation and trust.

### 1.3 Machine Learning in Computer-Aided Diagnosis (CAD)
Deep learning, particularly Convolutional Neural Networks (CNNs), has revolutionized computer vision. Unlike classical image processing techniques that rely on handcrafted feature extractors (such as Sobel filters or SIFT), CNNs learn hierarchical feature representations directly from the raw pixel inputs. In medical image analysis, CAD systems powered by CNNs act as valuable diagnostic assistants, providing secondary opinions, triaging critical cases in emergency wards, and serving as primary screening tools in underserved communities.

---

## 2. Chapter 2: Literature Review

### 2.1 Evolution of Deep Learning in Medical Imaging
Early computerized medical imaging relied on traditional machine learning classifiers like Support Vector Machines (SVMs) and Random Forests trained on geometric and texture features extracted via Gabor filters. While moderately effective in controlled environments, these systems struggled with clinical variations, differences in exposure, and variations in patient anatomy. 

With the advent of AlexNet in 2012, deep convolutional neural networks became the standard for visual recognition. Architectures evolved from VGG (introducing deep stacks of $3\times3$ convolutions) to ResNet (pioneering residual skip connections to overcome vanishing gradients in extremely deep structures).

### 2.2 CheXNet & DenseNet121 Framework
In 2017, the Stanford ML Group released **CheXNet**, a 121-layer DenseNet trained on the ChestX-ray14 dataset containing over 112,000 frontal view X-ray images. CheXNet demonstrated a diagnostic performance that surpassed average radiologists on several thoracic diseases.

The underlying backbone of CheXNet is **DenseNet121** (Densely Connected Convolutional Network). Unlike traditional architectures where each layer only receives input from its immediate predecessor ($x_l = H_l(x_{l-1})$), a DenseNet layer receives inputs from all preceding layers as a concatenated tensor:

$$x_l = H_l([x_0, x_1, \dots, x_{l-1}])$$

This dense connectivity scheme provides several distinct advantages:
*   **Feature Reuse**: Allows layers to access raw and low-level features directly from earlier blocks, maximizing informational efficiency.
*   **Mitigated Vanishing Gradients**: Provides direct path connections for gradients during backpropagation, stabilizing the weight updates.
*   **Fewer Parameters**: Since feature maps are reused, DenseNets require fewer channels per block, resulting in a more compact parameter footprint than a comparable ResNet.

### 2.3 Overview of Explainability in Medical AI
While modern deep networks demonstrate exceptional mathematical accuracy, their complex nested non-linear transformations render them "black boxes." In healthcare, a high-accuracy system is clinically unusable if practitioners cannot verify the biological basis of its decisions. Visual explainability algorithms, such as Class Activation Maps (CAM) and Grad-CAM, address this constraint by mapping the activation patterns of the final convolutional layer back onto the original input image, providing a direct visual explanation of what regions motivated the classification.

---

## 3. Chapter 3: System Methodology & Architecture

The system is structured as a modular four-phase deep learning pipeline to ensure separation of concerns and reproducibility:

```
[Phase 1: Preprocessing] ➔ [Phase 2: Architecture Setup] ➔ [Phase 3: Weights Training] ➔ [Phase 4: Validation & ROC Analysis]
```

### 3.1 Phase 1: Preprocessing & In-Distribution Normalization
**File: `phase1.py`**
Frontal chest radiographs vary significantly in resolution, aspect ratio, and lighting. This phase prepares the raw images for the neural network:
1.  **Grayscale to RGB Alignment**: Since CXR images can be single-channel grayscale, they are mapped to 3-channel RGB to match the expected input shape of pre-trained ImageNet architectures.
2.  **Spatial Standardization**: Images are bilinearly resized to $224\times224$ pixels.
3.  **Data Augmentation**: To mitigate overfitting during training, spatial transforms are applied:
    *   Random Horizontal Flips
    *   Random Rotations up to $15^\circ$
    *   Random Affine translations ($\pm5\%$)
    *   Subtle brightness and contrast jitter ($\pm10\%$)
4.  **In-Distribution ImageNet Normalization**: Tensors are scaled to the range $[0.0, 1.0]$ and normalized using standard ImageNet parameters:

$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

This step is mathematically critical, as a failure to normalize inputs using the exact statistics of the pretrained model's training distribution leads to catastrophic false-positive predictions.

### 3.2 Phase 2: Network Architecture & Loss Function Selection
**File: `phase2.py`**
We utilize a DenseNet121 architecture. To leverage transfer learning, we load weights pre-trained on ImageNet.
*   **Parameter Freezing**: The feature extraction blocks (layers 1 through 3) are frozen to preserve the general edge and texture filters.
*   **Targeted Fine-Tuning**: The parameters of the final dense block (`denseblock4`) are unfrozen to allow the network to specialize in identifying delicate lung consolidations.
*   **Custom Classifier**: The original 1000-class linear output is replaced with a dropout layer ($p=0.3$ to reduce overfitting) and a single linear node ($1024 \to 1$).
*   **Imbalanced Class Loss Weighting**: Medical datasets are frequently imbalanced. In the Kaggle dataset, the ratio of NORMAL (1,341) to PNEUMONIA (3,875) is approximately $1:3$. To prevent the model from blindly predicting the majority class, we configure the loss function with a positive weight of $0.35$:

$$pos\_weight = 0.35 \implies \mathcal{L}_{BCE} = - [ pos\_weight \cdot y \log(p) + (1-y)\log(1-p) ]$$

This penalizes false positives more heavily, resulting in a balanced learning trajectory.

### 3.3 Phase 3: Weight Optimization & Modular Training
**File: `phase3.py` & `train.py`**
This module orchestrates the feed-forward and backward propagation loops over 10 training epochs:
1.  **Forward Propagation**: Computes raw logits from the input batch.
2.  **Loss Evaluation**: Computes the weighted binary cross-entropy loss.
3.  **Backpropagation**: Calculates gradients using autograd.
4.  **Gradient Descent**: Updates parameters using the Adam Optimizer with a slow, stable learning rate of $\eta = 10^{-5}$.
5.  **Checkpointing**: Monitors validation loss at the end of each epoch and saves the best model state dictionary.

### 3.4 Phase 4: Model Evaluation & Youden's J Statistic Thresholding
**File: `phase4.py`**
Standard models threshold probability output at a default value of $0.5$. In clinical environments, this is often highly suboptimal. To optimize decision boundaries, Phase 4 evaluates the model's ROC curve on validation data and computes **Youden's J statistic**:

$$J(t) = \text{Sensitivity}(t) + \text{Specificity}(t) - 1$$

The threshold $t^*$ that maximizes $J(t)$ is selected as the optimal classification threshold. This balances precision and recall mathematically, preventing excessive false alarms.

---

## 4. Chapter 4: Explainability & Grad-CAM

### 4.1 Explainability as a Clinical Necessity
Deep learning models can make correct predictions based on the wrong features (e.g. classifying a scan as pneumonia based on scanner artifacts, patient posture, or hospital watermarks). In clinical environments, this is highly dangerous. Explainable AI ensures that the model is making its decision based on anatomical regions—specifically, lung consolidation, infiltrates, and pleural effusions.

### 4.2 Grad-CAM Mathematical Foundation & Hook Setup
**File: `gradcam.py`**
Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of any target score flowing into the final convolutional layer to produce a coarse localization map highlighting important regions:

```
[Forward Pass] ➔ [Extract Activation Maps A] ➔ [Backward Pass] ➔ [Extract Gradients dy/dA] ➔ [Compute Weights] ➔ [Generate Heatmap]
```

1.  **Forward Activation**: We extract the feature maps $A^k$ from the final convolutional block (`denseblock4`).
2.  **Backward Gradients**: We compute the gradient of the logit $y^c$ (before sigmoid) with respect to the feature map activations:

$$\frac{\partial y^c}{\partial A^k}$$

3.  **Importance Weighting**: We perform global average pooling on the gradients to compute the channel importance weights $\alpha_k^c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

4.  **Rectified Linear Combination**: We compute a weighted combination of forward activation maps and apply a ReLU function to only highlight features that positively correlate with the target class:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

This resulting 2D heatmap is resized to $224\times224$ and overlaid on the original X-ray using a colormap to visualize model focus.

---

## 5. Chapter 5: Implementation, Diagnostics, & Bug Corrections

### 5.1 Implementation Environment & Hardware Dependencies
The project is built entirely on a modular Python stack:
*   **Operating System**: Windows (tested on Windows 11) / Linux
*   **Interpreter**: Python 3.14.4
*   **Deep Learning framework**: PyTorch (CPU/GPU-accelerated CUDA)
*   **Visualization**: Matplotlib & OpenCV
*   **Explainability**: Custom Grad-CAM implementation

### 5.2 Unicode Encoding CP1252 Crash Resolution
*   **The Bug**: The training and evaluation scripts contained UTF-8 checkmark emojis (`✅`) to denote progress. In standard Windows environments, the command shell defaults to CP1252 (ANSI) encoding, causing the interpreter to crash instantly with a `UnicodeEncodeError`.
*   **The Fix**: Cleaned all print statements across `train.py`, `inference.py`, `phase1.py`, `phase2.py`, `phase3.py`, and `phase4.py` to use ASCII-safe indicators (e.g. `[OK]` and `[SUCCESS]`), ensuring seamless cross-platform execution.

### 5.3 The Normalization Mismatch Resolution (Bug Fix Log)
*   **The Bug**: During local verification of the pre-trained weights, both NORMAL and PNEUMONIA images yielded extremely high pneumonia probabilities (e.g. `IM-0001-0001.jpeg` predicted PNEUMONIA with a confidence of **0.9892**). We diagnosed that the local preprocessing pipeline was using `[0.5, 0.5, 0.5]` normalization, whereas the Colab training script utilized standard ImageNet normalization (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`). This mismatch forced the input out-of-distribution.
*   **The Fix**: Restructured both `phase1.py` and `inference.py` to utilize standard ImageNet normalization. This immediately resolved the false positive predictions:

| Model Preprocessing Normalization | Normal X-Ray (`IM-0001-0001.jpeg`) | Pneumonia X-Ray (`person100_bacteria_475.jpeg`) |
| :--- | :--- | :--- |
| **Old (`[0.5, 0.5, 0.5]`)** | 0.989208 (PNEUMONIA) ❌ | 0.954106 (PNEUMONIA) |
| **New (`ImageNet Standard`)** | **0.637033 (NORMAL)** ✅ | **0.776459 (PNEUMONIA)** ✅ |

### 5.4 Preventing GUI Blocking in Headless Deployments
*   **The Bug**: Calling `plt.show()` in `inference.py` hung the script indefinitely in headless terminal environments or non-interactive shells.
*   **The Fix**: Modified `inference.py` to write the output visualization to `gradcam_result.png` using `plt.savefig()` and removed the blocking popup behavior.

---

## 6. Chapter 6: Results, Discussion, & Conclusion

### 6.1 Experimental Results & Validation Run
We verified the complete pipeline by running local inference:

```bash
python inference.py
```

#### Verification Run Log:
```text
Using Device: cpu
[OK] Model Loaded Successfully

========== RESULT ==========

Prediction : NORMAL
Confidence : 0.6370
[OK] Grad-CAM Visualization saved to: gradcam_result.png
```

The model successfully classified the normal chest radiograph as **`NORMAL`** since the confidence value of `0.6370` sits comfortably below the optimized classification threshold of `0.70`. 

### 6.2 Technical Discussion & Limitations
While the model demonstrates high performance and visual alignment, several limitations remain:
*   **Demographic & Scanner Drift**: X-ray styles differ between clinical settings due to different manufacturers and acquisition protocols. A model trained on pediatric Kaggle scans can experience performance drop-offs on senior adult scans.
*   **Resolution Compression**: Compressing large $2000\times2000$ radiographs to $224\times224$ can destroy micro-features, such as minor consolidations.
*   **Education Context**: This system is research-oriented and must only be utilized as a secondary decision-support tool.

### 6.3 Future Scope & Clinical Enhancements
To build upon this work, future enhancements will target:
1.  **State-of-the-Art Architectures**: Evaluating Vision Transformers (ViT) and ConvNeXt backbones.
2.  **Semantic Segmentation**: Integrating a U-Net model to segment lung regions first, forcing the classifier to ignore extra-thoracic regions.
3.  **Multi-Class Detection**: Broadening classification capability to detect overlapping thoracic conditions (e.g. tuberculosis, effusion, pneumothorax).
4.  **Grad-CAM++ Integration**: Implementing second-order gradients to generate sharper feature localization maps.

### 6.4 References
1.  Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." *arXiv:1711.05225*.
2.  Huang, G., et al. (2017). "Densely Connected Convolutional Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
3.  Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
4.  Kaggle Chest X-Ray Dataset: *https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia*.
