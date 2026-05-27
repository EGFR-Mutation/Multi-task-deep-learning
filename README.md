# Multi-task Deep Learning Model for Predicting EGFR Mutation Status in NSCLC

Implementation of the paper:

**"Multi-task deep learning model for predicting EGFR mutation status in NSCLC"**

---

# Overview

This repository provides the implementation of a multi-task deep learning framework for predicting EGFR mutation status in non-small cell lung cancer (NSCLC) using CT images and clinical/radiological features.

The repository includes:

* CT image preprocessing pipeline
* Multi-task deep learning framework
* Baseline model implementations
* Radiomics-based methods
* Biological mechanism analysis
* Visualization analysis

---

# Repository Structure

```text
.
├── biological mechanisms
│   └── biological mechanisms.R
│
├── image preprocessing
│   ├── resample.py
│   └── test_csv.py
│
├── model building
│   ├── baseline model
│   ├── multi-task
│   │   ├── config_egfr.py
│   │   ├── dataset_egfr_RU.py
│   │   └── train_resnet_rec_2_0401.py
│   └── Radiomics
│
├── visualization analysis
│   ├── T-SNE.py
│   └── test_grad_cam.py
│
├── examples
│   ├── train_example.csv
│   └── inference_example.csv
│
└── README.md
```

---

# Environment Requirements

## Hardware

* NVIDIA GPU with CUDA support
* Recommended GPU memory: >= 12 GB

## Python Environment

* Python 3.9+
* PyTorch 1.10+
* CUDA 11.3+

## Main Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Recommended package versions:

```text
python==3.8
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
scipy==1.7.3
matplotlib==3.5.3
opencv-python==4.6.0.66
SimpleITK==2.1.1
nibabel==3.2.2
torch==1.10.1
torchvision==0.11.2
tqdm==4.64.1
Pillow==9.2.0
seaborn==0.12.2
lifelines==0.27.4
pyradiomics==3.0.1
```

Example environment setup:

```bash
conda create -n egfr_mt python=3.9
conda activate egfr_mt

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

---

# Data Preparation

## 1. CT Image Preparation

### Input Format

The framework supports:

* `.nii`
* `.nii.gz`
* `.mhd`
* `.nrrd`

### Recommended Directory Structure

```text
data
├── CT
│   ├── Patient_001.nii.gz
│   ├── Patient_002.nii.gz
│   └── ...
│
├── Mask
│   ├── Patient_001.nii.gz
│   ├── Patient_002.nii.gz
│   └── ...
│
└── csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### Image Resampling

Run the preprocessing script:

```bash
cd "image preprocessing"
python resample.py
```

Typical preprocessing includes:

* Resampling to isotropic spacing
* Intensity normalization
* ROI cropping
* Slice extraction
* Multi-view generation (axial/coronal/sagittal)

---

# Clinical and Radiological Feature Preparation

Clinical/radiological features should be stored in CSV format.

## Required CSV Columns

| Column Name     | Data Type  | Description               | Example                 |
| --------------- | ---------- | ------------------------- | ----------------------- |
| patient_id      | string     | Unique patient identifier | P001                    |
| image_path      | string     | Path to CT image          | ./data/CT/P001.nii.gz   |
| mask_path       | string     | Path to tumor mask        | ./data/Mask/P001.nii.gz |
| label           | int        | EGFR mutation label       | 0                       |
| age             | float/int  | Patient age               | 63                      |
| sex             | int        | Sex (0=female, 1=male)    | 1                       |
| smoking         | int        | Smoking history           | 0                       |
| stage           | string/int | TNM stage                 | IIIA                    |
| slice_thickness | float      | CT slice thickness        | 1.0                     |

---

# Example CSV Files

We provide two example CSV files in the `examples/` folder:

* `train_example.csv`
* `inference_example.csv`

These files demonstrate the exact CSV structure required for:

* Model training
* Validation
* Inference/testing

## Example Training CSV

```csv
patient_id,image_path,mask_path,label,age,sex,smoking,stage,slice_thickness
P001,./data/CT/P001.nii.gz,./data/Mask/P001.nii.gz,1,63,1,0,IIIA,1.0
P002,./data/CT/P002.nii.gz,./data/Mask/P002.nii.gz,0,58,0,1,IIIB,1.0
```

## Example Inference CSV

```csv
patient_id,image_path,mask_path,age,sex,smoking,stage,slice_thickness
P101,./data/CT/P101.nii.gz,./data/Mask/P101.nii.gz,67,1,1,IIIA,1.0
```

---

# Model Training

## Multi-task Model

Main training script:

```text
model building/multi-task/train_resnet_rec_2_0401.py
```

## Training Command

```bash
cd "model building/multi-task"

python train_resnet_rec_2_0401.py 
```

## Main Hyperparameters

| Parameter      | Description                  | Default |
| -------------- | ---------------------------- | ------- |
| --batch_size   | Batch size                   | 16      |
| --epochs       | Number of epochs             | 200     |
| --lr           | Learning rate                | 1e-4    |
| --weight_decay | Weight decay                 | 1e-5    |
| --num_workers  | Number of dataloader workers | 8       |

---

## Output

Inference outputs may include:

* EGFR mutation probability
* Binary classification result
* Feature embeddings
* Attention maps

---

# Baseline Models

Baseline methods are located in:

```text
model building/baseline model
```

These implementations can be used for comparative experiments.

---

# Radiomics Analysis

Radiomics-related code is located in:

```text
model building/Radiomics
```

Typical workflows include:

* Feature extraction

---

# Visualization and Interpretability

## t-SNE Visualization

```bash
cd "visualization analysis"
python T-SNE.py
```

## Grad-CAM Visualization

```bash
python test_grad_cam.py
```

These scripts are used for:

* Feature distribution visualization
* Model interpretability analysis
* Tumor attention localization

---

# Biological Mechanism Analysis

Biological analysis scripts are located in:

```text
biological mechanisms
```

Run:

```bash
Rscript "biological mechanisms.R"
```

Potential analyses include:

* Pathway enrichment
* Gene expression analysis
* Biological correlation studies

---

# Contact

For questions or collaborations, please contact:

* Email: [sql13676430586@163.com](mailto:sql13676430586@163.com)
