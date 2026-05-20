# A Three-Stage Analysis Framework for Cervical TCT Smears Based on FlexInc-YOLOv11 and DINOv3

![GitHub license](https://img.shields.io/github/license/littlechicken139/Three-Stage-Cervical-TCT-Smear-Detection-Framework)
![GitHub issues](https://img.shields.io/github/issues/littlechicken139/Three-Stage-Cervical-TCT-Smear-Detection-Framework)
![GitHub stars](https://img.shields.io/github/stars/littlechicken139/Three-Stage-Cervical-TCT-Smear-Detection-Framework)

Official open-source code for the paper published in **Applied Sciences (MDPI)**.

This repository provides the complete implementation of the proposed three-stage cervical cell analysis framework, including:
- Improved FlexInc-YOLOv11 detector with MSCA detection head
- DINOv3 feature extraction pipeline
- Lightweight MLP classification module

**Keywords**: cervical TCT smears; YOLO; multi-scale object detection; fine-grained classification; DINOv3; medical image analysis

---

## 🔬 Overall Framework

The complete three-stage pipeline of the proposed framework:

![Framework](assets/figures/framework.png)

1. **Stage 1 - Detection**: FlexInc-YOLOv11 detector localizes cells in TCT smear images
2. **Stage 2 - Feature Extraction**: Frozen DINOv3 backbone extracts robust semantic features
3. **Stage 3 - Classification**: Lightweight MLP head performs fine-grained classification

---

## 📝 Abstract

To address the challenges of dense overlapping cells, large scale variations, and subtle differences among abnormal cells in cervical TCT smear images, this paper proposes a three-stage analysis framework based on FlexInc-YOLOv11 and DINOv3. In the first stage, an improved FlexInc-YOLOv11 detector is used to localize candidate cell regions. The second stage employs a frozen DINOv3 backbone to extract high-level semantic features. In the third stage, a lightweight MLP classification head performs fine-grained classification under both pathological and morphological label standards. Experimental results demonstrate that the proposed method achieves superior performance in both detection and classification tasks, providing an effective technical solution for automated cervical cell screening.

---

## 🏆 Contributions

1. Propose a decoupled three-stage framework that separates detection, feature extraction, and classification, reducing the difficulty of joint optimization for dense cell recognition.
2. Propose the FlexInc-YOLOv11 detection model, which improves cell detection accuracy in multi-scale and densely occluded scenarios through FlexInc modules and MSCA detection head.
3. Adopt a feature extraction strategy with frozen DINOv3 backbone, effectively alleviating overfitting on small-sample medical data.
4. Design a lightweight MLP classification head to achieve dual-standard fine-grained classification (pathological and morphological), enhancing clinical practicality.

---

## 🧠 Core Modules

### 1. FlexInc Module & MSCA Detection Head

Custom-designed modules for dense and overlapping cervical cell detection:

| Module | Description |
|--------|-------------|
| **FlexIncConv** | Multi-branch incremental convolution with directional sensitivity |
| **FlexIncBlock** | Feature fusion block for multi-scale cervical cell feature enhancement |
| **MSCA-Detect Head** | Multi-scale channel attention for dense occlusion regions |

### 2. DINOv3 Feature Extraction Pipeline

Frozen pre-trained DINOv3 backbone extracts high-level semantic features from cropped single-cell regions.

### 3. Lightweight Classification Heads

Lightweight MLP classification head for dual-standard fine-grained classification.

---

## 📁 Repository Structure

```plaintext
Three-Stage-Cervical-TCT-Smear-Detection-Framework/
├── README.md
├── LICENSE
├── requirements.txt
├── assets/
├── Stage1-FlexInc-YOLO/
│   ├── train_yolo.py
│   ├── detect.py
│   ├── yolo11-FlexInc.yaml
│   ├── CellDetectSampleDataset/
│   └── ultralytics/
├── Stage2-DINOv3/
│   └── extract_dinov3_features.py
└── Stage3-Classifier/
    ├── train_classifier.py
    └── evaluate_classifier.py
```

---

## 🛠️ Environment Setup

```bash
git clone https://github.com/littlechicken139/Three-Stage-Cervical-TCT-Smear-Detection-Framework.git
cd Three-Stage-Cervical-TCT-Smear-Detection-Framework
pip install -r requirements.txt
```

---

## 🚀 Training & Testing Commands

### Stage 1: Train Detection Model

```bash
cd Stage1-FlexInc-YOLO
python train_yolo.py --data CellDetectSampleDataset/CellDetectSampleDataset.yaml --cfg yolo11-FlexInc.yaml --epochs 100 --batch-size 8
```

### Stage 1: Inference

```bash
cd Stage1-FlexInc-YOLO
python detect.py --weights runs/detect/train/weights/best.pt --source your_image_path --conf 0.5
```

---

## 📊 Dataset Description

The TCT cervical smear dataset used in this study was collected from **The First Affiliated Hospital of Zhengzhou University**, with strict anonymization to protect patient privacy:

| Attribute | Details |
|-----------|---------|
| **Patient Count** | Involving dozens of independent patients (anonymized) |
| **Data Scale** | Training set: 2,460 TCT smear images with 112,439 annotations; Test set: 614 images with 30,044 annotations |
| **Imaging Device** | Medical biological microscope |
| **Staining Method** | Pap staining |
| **Annotation Standard** | All annotations were completed and reviewed by professional pathologists, following clinical TCT cytology standards |

**Note**: Due to medical privacy constraints, the raw dataset cannot be publicly released. We provide complete training code, model weights, and reproduction configuration for researchers.

> ⚠️ **Important**: The dataset included in this repository is solely a **sample** (a small portion of the full dataset) for code testing and validation purposes. The class categories and data quantities shown here are for reference only. They do not represent the complete dataset used in the paper.

---

## 📊 Experimental Results

### Detection Performance

| Metric | Value |
|--------|-------|
| Precision | 0.7860 |
| Recall | 0.7544 |
| F1-score | 0.7699 |
| mAP50 | 0.8426 |
| mAP75 | 0.6948 |
| mAP50-95 | 0.6350 |

### Classification Performance

**Pathological Classification Accuracy**: 87.78%
- 4 classes: Benign epithelial cells, Inflammatory response cells, Infection-related cells, Atypical/precancerous cells

**Morphological Classification Accuracy**: 89.64%
- 6 classes: Normal squamous cells, Abnormal squamous cells, Glandular cell-related cells, Inflammatory cells, Reactive/metaplastic cells, Other


---

## ⚡ Computational Overhead

| Model | Params (M) | FLOPs (G) | FPS |
|-------|-----------|-----------|-----|
| YOLOv11s | 18.3 | 21.3 | 102.8 |
| Ours (FlexInc-YOLOv11) | 28.3 | 19.7 | 99.74 |

---

## 📚 Citation

If this repository helps your research, please cite our paper:

```bibtex
@article{Liu2026TCT,
  title={A Three-Stage Analysis Framework for Cervical TCT Smears Based on FlexInc-YOLOv11 and DINOv3},
  author={Liu, Junfu and Gao, Binzhi and Wang, Xiaoyang and Liu, Ting and Huang, Sirui and Zhu, Hong and Shi, Jing and Yang, Yi},
  journal={Applied Sciences},
  volume={16},
  number={11},
  pages={XXXX},
  year={2026},
  publisher={MDPI}
}
```

---

## 📄 License

This project is open-source under the **MIT License**. For academic research only.

---

## 📧 Contact

**Author Affiliations and Emails:**

1. School of Economics and Management, Xi'an University of Technology, Xi'an 710048, China; 3230513006@stu.xaut.edu.cn (J.L.); yangyi@xaut.edu.cn (Y.Y.)
2. School of Computer Science and Engineering, Xi'an University of Technology, Xi'an 710048, China; 3230913025@stu.xaut.edu.cn (B.G.)
3. School of Civil Engineering and Architecture, Xi'an University of Technology, Xi'an 710048, China; 3221631002@stu.xaut.edu.cn (X.W.)
4. School of Automation and Information Engineering, Xi'an University of Technology, Xi'an 710048, China; 3240412032@stu.xaut.edu.cn (T.L.); 3240416022@stu.xaut.edu.cn (S.H.); zhuhong@xaut.edu.cn (H.Z.); shijing@xaut.edu.cn (J.S.)

*Correspondence: shijing@xaut.edu.cn (J.S.); yangyi@xaut.edu.cn (Y.Y.)*

**Featured Application:**
The proposed three-stage deep learning framework can be integrated into computer-aided diagnosis (CAD) systems for automated and high-precision cervical cancer screening, significantly reducing the workload of pathologists and improving diagnostic consistency in clinical environments.

---
