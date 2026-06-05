# Light Weight BS UNet for Music Source Separation

This project is a lightweight, efficient reproduction of the **Roformer-based Music Source Separation** architecture. It focuses on separating music into four stems: **Vocals, Drums, Bass, and Other**.

## 📄 Reference Paper
This implementation is based on the following research:

- **Title**: Moises-Light: Resource-efficient Band-split U-Net For Music Source
- **Authors**: Yun-Ning (Amy) Hung, Igor Pereira, Filip Korzeniowski
- **Source**: https://arxiv.org/pdf/2510.06785


## ⚙️ Installation

### 1. Create Anaconda Environment
First, create a new conda environment with Python 3.10:
```bash
conda create -n moises-light python=3.10 -y
conda activate moises-light
```

### 2. Install FFmpeg
This project requires FFmpeg for audio processing.
```bash
conda install -c conda ffmpeg -y
```

### 3. Install PyTorch (with CUDA 11.8)
Install PyTorch following the official instructions for your hardware. For CUDA 11.8:
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Install Other Dependencies
Install the remaining packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 📂 Dataset

### 1. Preparation
This project is designed to work with the **MUSDB18-HQ** dataset (or any dataset with a similar structure). It supports both `.wav` and `.flac` formats.

The expected directory structure is:
```text
data/
└── musdb18hq/
    ├── train/
    │   ├── Track Name 1/
    │   │   ├── vocals.wav
    │   │   ├── drums.wav
    │   │   ├── bass.wav
    │   │   └── other.wav
    │   └── ...
    └── test/
        ├── Track Name A/
        │   ├── vocals.wav
        │   ...
```

### 2. Virtual Dataset Expansion (On-the-fly Mixing)
To improve generalization and efficiency, the `MUSDBDataset` implements **on-the-fly mixing**:
- For each training step, it randomly selects segments from **different tracks** for each stem (vocals, drums, etc.).



## 🚀 Training

To start training for a specific stem (e.g., vocals):
```bash
python scripts/train.py --stem vocals --data_dir data/musdb18hq/train
```

Parameters:
- `--stem`: Target stem (`vocals`, `drums`, `bass`, `other`).
- `--data_dir`: Path to the training dataset.
- `--config`: Path to the config file (default: `configs/default.yaml`).

## 📊 Evaluation Results

### V1


- Audio conifg

| n_fft | hop | win | duration |
| :--- | :--- | :--- | :--- |
| 6144| 1024 | 6144 | 4.0  |

- Model config

| n_enc_layer | n_dec_layer | n_bands | G | n_rope | n_head |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3| 1 | 4 | 32 | 4 | 8 |

- Training config

| batch_size| sample_per_track | lr | epoch | 
| :--- | :--- | :--- | :--- | 
| 2| 10(900 chunks) | 0.0003 | 200

- Museval Evaluation

| STEM | cSDR (Median) | uSDR (Mean) |
| :--- | :--- | :--- |
| **VOCALS** | 4.31 dB | 4.86 dB |
| **DRUMS** | - | - |
| **BASS** | - | - |
| **OTHER** | - | - |

---
