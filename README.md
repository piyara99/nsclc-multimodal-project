# 🫁 AI-Based Prediction of NSCLC Recurrence Using Multimodal Deep Learning

> **Final Year Research Project | BSc (Hons) Software Engineering | NSBM Green University (University of Plymouth, UK)**

> A resource-efficient multimodal deep learning framework for predicting **Non-Small Cell Lung Cancer (NSCLC) recurrence** by integrating structured clinical data from **TCGA** with histopathology image embeddings extracted from **LC25000** tissue patches, trained entirely on a single **6 GB consumer GPU**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📖 Overview

Lung cancer remains the leading cause of cancer-related mortality worldwide, with **Non-Small Cell Lung Cancer (NSCLC)** accounting for approximately **85% of all lung cancer cases**. Predicting recurrence after treatment is critical for improving patient stratification, treatment planning, and long-term outcomes.

This project presents a **resource-efficient multimodal deep learning framework** that combines:

- 🩺 Structured clinical data from **TCGA**
- 🔬 Histopathology image embeddings extracted from **LC25000 tissue patches**

to predict NSCLC recurrence using **adaptive multimodal fusion**. Unlike traditional feature concatenation, the proposed **Gated Multimodal Unit (GMU)** dynamically learns the importance of each modality for every patient.

---

# 🚀 Highlights

- 🧠 Novel multimodal framework for NSCLC recurrence prediction
- ⚡ Adaptive **Gated Multimodal Unit (GMU)** fusion
- 💻 Trained entirely on a single **NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)**
- 🔍 Explainable AI using **Grad-CAM**, **SHAP**, and **GMU gate-weight analysis**
- 📊 Five-fold stratified cross-validation
- 🚫 Automatic out-of-distribution image rejection
- 🌐 Interactive Streamlit dashboard

---

# 🎯 Objectives

- Predict NSCLC recurrence from a deduplicated TCGA patient cohort.
- Learn discriminative histopathology image representations using a ResNet-50 encoder.
- Encode structured clinical variables using a multilayer perceptron (MLP).
- Compare unimodal and multimodal learning strategies.
- Evaluate adaptive fusion against weighted and concatenation fusion.
- Provide transparent predictions through Explainable AI techniques.
- Deploy an interactive research dashboard.

---

# 🏗 System Architecture

<p align="center">
  <img src="assets/architecture.png" width="750"/>
</p>

```text
Histopathology Image Patches          Structured Clinical Data
             │                                    │
             ▼                                    ▼
      ResNet-50 Encoder                 Clinical Encoder (MLP)
             │                                    │
             ▼                                    ▼
      Image Embeddings ─────────────► Gated Multimodal Unit
                                                │
                                                ▼
                                      Recurrence Prediction
                                                │
                                                ▼
                         Grad-CAM • SHAP • Gate Weight Analysis
```

---

# ⚡ Resource-Efficient Pipeline

Rather than performing computationally expensive end-to-end training on gigabyte-scale whole-slide images, this framework adopts an **offline feature extraction strategy**.

Histopathology image patches are first encoded using a **ResNet-50** backbone.

The resulting embeddings are stored and reused during multimodal training, allowing the complete framework to operate within the constraints of a **6 GB NVIDIA RTX 3050 Laptop GPU** while maintaining reproducibility and computational efficiency.

---

# 📂 Dataset

## Clinical Data

**Source**

- TCGA-LUAD
- TCGA-LUSC

Clinical variables include:

- Age
- Gender
- Smoking History
- ECOG Performance Status
- Disease Stage
- T Stage
- N Stage
- Histological Subtype
- Prior Treatment

After preprocessing, duplicate patient records were removed to prevent data leakage.

- **79 duplicate records removed**
- **62 duplicated patients removed**
- **169 unique patients retained**

---

## Histopathology Data

**Source**

- LC25000 Histopathology Dataset

Image classes:

- LUAD
- LUSC

Resolution:

```
224 × 224 RGB
```

Since patient-specific TCGA whole-slide images could not be processed within the available hardware resources, image embeddings extracted from LC25000 tissue patches were **proxy-matched** to TCGA patients according to NSCLC subtype.

---

# 🧪 Models Evaluated

Three classical machine learning baselines established lower-bound performance before evaluating five deep learning configurations.

| Model | AUC | 95% CI | F1 |
|------|------|------|------|
| Logistic Regression | 0.5232 | [0.434–0.610] | 0.4895 |
| Random Forest | 0.4573 | [0.363–0.544] | 0.5439 |
| XGBoost | 0.4771 | [0.384–0.564] | 0.5487 |
| Clinical-only (MLP) | 0.5751 ± 0.0885 | — | 0.1685 |
| Image-only (ResNet-50) | 0.6261 ± 0.0891 | — | 0.3577 |
| Weighted Fusion | 0.6020 ± 0.1043 | — | 0.5434 |
| Concatenation Fusion | 0.6346 ± 0.0867 | — | 0.3232 |
| **GMU (Proposed)** | **0.6381 ± 0.0459** | **[0.541–0.718]** | **0.5144** |

The proposed **GMU** achieved the highest ROC-AUC while also producing the lowest cross-validation variance, demonstrating improved robustness compared to static fusion strategies.

<p align="center">
<img src="assets/roc_curves.png" width="700"/>
</p>

---

# 📊 GMU Classification Performance

| Metric | Non-Recurrence | Recurrence | Macro Average |
|------|------|------|------|
| Precision | 0.71 | 0.43 | 0.57 |
| Recall | 0.50 | 0.65 | 0.57 |
| F1 Score | 0.58 | 0.51 | 0.55 |

The use of **Focal Loss** improved sensitivity toward the minority recurrence class, resulting in higher recall for recurrence predictions.

<p align="center">
<img src="assets/confusion_matrix.png" width="550"/>
</p>

---

# 🎛 Calibration & Efficiency

| Metric | Value |
|------|------|
| Brier Score | 0.2426 |
| Expected Calibration Error | 0.1457 |
| Single Patient Inference | 3.41 ± 1.83 ms |
| Auxiliary Subtype Classifier Accuracy | 99.94% |

<p align="center">
<img src="assets/calibration_plot.png" width="550"/>
</p>

---

# 🔍 Explainable AI

The framework integrates three complementary explainability techniques.

### 🔥 Grad-CAM

Highlights image regions responsible for subtype prediction.

### 📈 SHAP

Provides both global feature importance and patient-level explanations for clinical variables.

### ⚖ GMU Gate Analysis

Quantifies the contribution of image and clinical modalities for each prediction.

Average image contribution:

```
0.665 ± 0.179
```

<p align="center">
<img src="assets/gradcam_heatmaps.png" width="700"/>
</p>

---

# 🌐 Streamlit Dashboard

The interactive dashboard supports:

- Patient clinical data entry
- Histopathology image upload
- Recurrence probability prediction
- Grad-CAM visualization
- SHAP explanation
- Automatic rejection of non-histopathology images

---

# 🛠 Technology Stack

| Category | Technologies |
|------------|------------------------------|
| Language | Python |
| Deep Learning | PyTorch, TorchVision |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP, Grad-CAM |
| Image Processing | OpenSlide, OpenCV, Pillow |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

# 📁 Project Structure

```text
NSCLC-Prediction/

├── app/
├── assets/
├── data/
├── explainability/
├── inference/
├── models/
├── notebooks/
├── outputs/
├── preprocessing/
├── training/
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/piyara99/nsclc-multimodal-project.git

cd nsclc-multimodal-project

pip install -r requirements.txt
```

---

# ▶️ Run

```bash
streamlit run app.py
```

---

# 📚 Research

This repository contains the implementation accompanying my **BSc (Hons) Software Engineering Final Year Research Project** completed at **NSBM Green University** in affiliation with the **University of Plymouth (UK)**.

**Project Title**

> **AI-Based Prediction of NSCLC Recurrence as a Treatment Response Proxy Using Multimodal Histopathology and Clinical Data**

The research investigates resource-efficient multimodal deep learning for predicting NSCLC recurrence by integrating histopathology image representations with structured clinical data through adaptive fusion strategies.

---

# 🔮 Future Work

- Patient-specific TCGA whole-slide image integration
- Vision Transformers
- Attention-based multimodal fusion
- Multiple Instance Learning (MIL)
- Foundation pathology models
- Survival prediction
- External clinical validation

---

# 👩‍💻 Author

**Piyara Morawakaarachchi**

Final-Year Software Engineering Undergraduate

**NSBM Green University**  
**University of Plymouth (UK)**

### Research Interests

- Artificial Intelligence
- Medical Imaging
- Multimodal Learning
- Explainable AI
- Healthcare AI

[LinkedIn](https://www.linkedin.com/in/piyara-morawakaarachchi-20ab102a2/) • [GitHub](https://github.com/piyara99)

---

# 🙏 Acknowledgements

- The Cancer Genome Atlas (TCGA)
- LC25000 Histopathology Dataset
- National Cancer Institute
- The Cancer Imaging Archive (TCIA)
- Ms. M. T. A. Wickramasinghe (Project Supervisor)
- NSBM Green University
- University of Plymouth

---

⭐ **If you found this project interesting, consider giving the repository a star!**
