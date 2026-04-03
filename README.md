# Digital Pratilipi — Nepali Handwritten OCR Model

A deep learning pipeline for recognizing handwritten Devanagari/Nepali text. The system accepts a scanned handwritten image and outputs the recognized Nepali text, powered by a TrOCR-based sequence-to-sequence model trained on the IIIT-HW-Dev dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Python 3.10.10 Installation](#python-31010-installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Installing Dependencies](#installing-dependencies)
- [Training](#training)
- [Running Inference](#testing)

---

## Project Overview

Digital Pratilipi implements a full OCR pipeline for Nepali handwritten text:

1. **Preprocessing** — Illumination normalization, adaptive sharpening, CLAHE, binarization, and skew correction.
2. **Word Detection** — OpenCV-based contour analysis with multi-pass gap merging to isolate individual words.
3. **Recognition** — TrOCR transformer model fine-tuned on Devanagari handwritten word images.
4. **Post-processing** — BK-tree spell correction against a Nepali vocabulary.

---

## Repository Structure

```
DigitalPralitpiModel/
│
├── scripts/                       
│   ├── dataloader.py                 # Data loader for the IIIT-HT-HINDI dataset.
│   └── indic_preprocessing.py        # Dataset preprocessing for (Sharpening ---> Grayscale ---> Binarization ---> Skew Correction).
│   └── logs.py                       # Training parameters logs.
│   └── train.py                      # TrOCR based training script.
├── src/                            
│   ├── preprocessing.py            # Image preprocessing pipeline
│   ├── detection.py                # Word/line detection and bounding boxes
│   ├── ocr.py                      # TrOCR model inference
│   ├── vocabulary.py               # Nepali vocabulary and BK-tree loading
│   ├── postprocessing.py           # Spell correction and text cleanup
│   ├── pipeline.py                 # End-to-end pipeline orchestration
├── app.py                          # Flask app
├── index.html
```

---
## Dataset

This project trains on the **IIIT-HW-Dev** Devanagari handwritten word dataset, released by the Centre for Visual Information Technology (CVIT), IIIT Hyderabad.

> **Dataset page:** [https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data](https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

The dataset contains over **95,000 handwritten Devanagari word images** in `.jpg` format. Download **Version 1** (1.8 GB) from the link above and extract it into your local dataset directory. The path to this directory is configured inside `scripts/train.py` (see [Training](#training)).

---
 
## Prerequisites
 
- **OS:** Windows 10/11 or Linux (Ubuntu 20.04+)
- **GPU:** NVIDIA GPU with CUDA 12.8 support (strongly recommended for training)
- **Python:** 3.10.10 
 
---

### Python 3.10.10 Installation

To ensure dependency compatibility and avoid installation conflicts, Python 3.10.10 is recommended. While other versions may work, this specific environment has been verified for stable performance.

### Windows

1. Download the installer from the official Python releases page:
   [https://www.python.org/releases/python-3.10.10/](https://www.python.org/releases/python-3.10.10/)

2. Run the installer:
   - Check **"Add Python 3.10 to PATH"**
   - Click **"Customize installation"** → enable **pip**, **tcl/tk**, **py launcher**
   - Click **Install**

3. Verify:
   ```bash
   python --version
   # Expected: Python 3.10.10
   ```

### Linux (Ubuntu/Debian)

```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential libssl-dev libffi-dev \
    libsqlite3-dev zlib1g-dev libbz2-dev libreadline-dev

# Download and build Python 3.10.10
wget https://www.python.org/ftp/python/3.10.10/Python-3.10.10.tgz
tar -xf Python-3.10.10.tgz
cd Python-3.10.10
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall        # 'altinstall' keeps your system Python intact

# Verify
python3.10 --version
# Expected: Python 3.10.10
```

> **Note:** Use `python3.10` instead of `python` on Linux if your system Python is a different version.

---

## Virtual Environment Setup

Always use a virtual environment to isolate dependencies from your system Python.

### Windows

```bash
# Clone the repository
git clone https://github.com/KritanShres/DigitalPralitpiModel.git
cd DigitalPralitpiModel

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate
# Your prompt should now show (.venv)
```

### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/KritanShres/DigitalPralitpiModel.git
cd DigitalPralitpiModel

# Create virtual environment
python3.10 -m venv .venv

# Activate
source .venv/bin/activate
```

### Deactivate when done

```bash
deactivate
```

---

## Installing Dependencies

With the virtual environment **activated**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **For CPU-only machines:** The `requirements.txt` pins `torch==2.10.0+cu128` (CUDA 12.8). Replace the torch lines with the CPU build:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Training

```bash 
python scripts/train.py
```

### Pretrained Encoder & Decoder

The model is initialized from the following pretrained checkpoints before fine-tuning on IIIT-HW-Dev:

>**Encoder** [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)


>**Decoder** [flax-community/roberta-hindi](https://huggingface.co/flax-community/roberta-hindi)

The Vision Transformer (ViT) encoder extracts patch-level visual features from word images, while the RoBERTa-Hindi decoder generates the corresponding Devanagari character sequence autoregressively.

### Dataset Path Configuration

The path to your local IIIT-HW-Dev dataset folder is set **directly inside `scripts/train.py`**. Open the file and update the dataset path variable before running:

```python
# Inside scripts/train.py — update this to your extracted dataset location
DATASET_PATH = "/path/to/IIIT-HW-Dev"
```

### Training Arguments

Hyperparameters such as batch size, learning rate, number of epochs, and gradient accumulation steps are all configured inside `scripts/train.py`. **These must be tuned to match your GPU.**

The current training parameters is calibrated for an **NVIDIA RTX 4050 with 8 GB VRAM**. Adjust as follows:

| GPU / VRAM | Suggested Batch Size |
|---|---|
| RTX 4050 / 8 GB | 16 (current default) |
| RTX 3060 / 12 GB | 32 |
| RTX 3090 / 24 GB | 64 |
| RTX 4090 / 24 GB | 64–128 |
| CPU only | 4–8 (very slow) |

If you hit CUDA out-of-memory errors, reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` to maintain an equivalent effective batch size.

### Run Training

```bash
python scripts/train.py
```

Checkpoints and metric charts will be saved to the output directories configured inside the training script. Run the following command to view the training metrics.

```bash
tensorboard --logdir=logs
```

Checkpoints are saved after every 2000 steps. The training script will continue training from the last saved checkpoint. Simply run the training script again.

---
## Testing
Configure and run the `scripts/test.py` for inference testing of the trained model.

## Running the Application
Run the inference for your model in the app.py with the proposed pipeline on a flask server. 
```bash
python app.py
```
### Nepali Vocabulary Source
 
The word list used for spell correction is sourced from the **nepali-bhasa/nepali-spell** open-source project:
 
> **Repository:** [https://github.com/nepali-bhasa/nepali-spell](https://github.com/nepali-bhasa/nepali-spell)
