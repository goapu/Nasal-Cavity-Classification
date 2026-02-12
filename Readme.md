# ResNet50 Block-Level Anatomical Classification

**Author:** Dilip Goswami
**Affiliation:** M.Sc. Student, Technische Universität Berlin (TU Berlin)

---

This repository contains the complete implementation of a **deep learning–based block-level anatomical classification model** using a **pretrained ResNet-50** backbone. The model classifies medical images into **anterior, middle, and posterior** anatomical regions.

The project is designed for **academic and research use**, particularly in medical imaging workflows, and includes a full training, validation, and testing pipeline with robust evaluation utilities.

---

## Key Features

* Pretrained **ResNet-50** backbone (ImageNet)
* Custom **Focal Loss** for class imbalance handling
* Strong data augmentation using **Albumentations**
* Two-stage training strategy (frozen backbone → fine-tuning)
* Automatic metric computation and visualization
* CSV export of predictions and probabilities
* GPU (CUDA), Apple Silicon (MPS), and CPU support

---

## Project Structure

```
.
├── train_resnet.py        # Main training and evaluation script
├── custom_dataset.py     # Dataset class and transforms
├── focal_loss.py         # Focal Loss implementation
├── utils.py              # Metrics, plots, and helper functions
├── requirements.txt      # Python dependencies
├── LICENSE               # Apache 2.0 license
├── CITATION.cff          # Citation metadata
└── README.md
```

---

## Dataset Structure

The dataset must be organized as follows:

```
DATASET_ROOT/
├── Training/
│   ├── anterior/
│   ├── middle/
│   └── posterior/
├── Validation/
│   ├── anterior/
│   ├── middle/
│   └── posterior/
└── Testing/
    ├── anterior/
    ├── middle/
    └── posterior/
```

Each class directory should contain RGB images (e.g., PNG or JPEG).

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training and Evaluation

Update the dataset and output paths inside `train_resnet.py`:

```python
BASE_PATH = "/path/to/dataset"
OUTPUT_DIR = "/path/to/output"
```

Run the training script:

```bash
python train_resnet.py
```

### Training Procedure

1. Train classifier head with frozen ResNet-50 backbone
2. Apply early stopping based on validation loss
3. Unfreeze all layers and fine-tune the full network
4. Save the best-performing model checkpoint

---

## Outputs

After execution, the following outputs are generated:

* `best_model.pth` – best model checkpoint
* `val_results.csv` – validation predictions
* `test_results.csv` – test predictions
* Confusion matrices (PNG)
* Training and validation loss curves
* Classification reports (precision, recall, F1-score)

---

## Loss Function

The model uses **Focal Loss** to address class imbalance:

* Focuses learning on hard-to-classify samples
* Reduces dominance of majority classes
* Improves robustness on imbalanced medical datasets

---

## Hardware Support

The code automatically detects and uses the best available device:

* NVIDIA GPU (CUDA)
* Apple Silicon (MPS)
* CPU fallback

---

## Reproducibility

* Fixed random seeds
* Deterministic cuDNN settings
* Gradient clipping for stability

---

## Requirements

```txt
torch>=1.13
torchvision>=0.14
albumentations>=1.3
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

---

## Citation

If you use this code or model in academic work, **please cite the repository**.
Citation metadata is provided in `CITATION.cff`.

---

## License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this code for academic or commercial purposes, provided that proper attribution is given and the license notice is preserved.

---

## Disclaimer

This code is provided for **research purposes only** and is not intended for clinical deployment.
