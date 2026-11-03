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

Training runs one target stem at a time. Select the configuration that matches
the intended segment and effective-batch schedule:

- `v5_proposed_s_scnet_seg9_mb1_ga4_half_epoch.yaml`: 9-second segments,
  micro-batch 1, accumulation 4, and half-duration epochs.
- `v5_proposed_s_scnet_seg12_mb1_ga4_full_epoch.yaml`: 12-second segments,
  micro-batch 1, accumulation 4, and full-duration epochs.
- `v5_proposed_s_scnet_seg12_mb2_ga2_full_epoch.yaml`: 12-second segments,
  micro-batch 2, accumulation 2, and full-duration epochs.

Start training by selecting one of `vocals`, `drums`, `bass`, or `other`:

```bash
python scripts/train.py \
  --stem vocals \
  --config configs/v5_proposed_s_scnet_seg12_mb1_ga4_full_epoch.yaml \
  --data_dir data/musdb18hq_flac/train
```

Parameters:

- `--stem`: Target stem (`vocals`, `drums`, `bass`, `other`).
- `--data_dir`: Path to the training dataset.
- `--config`: Path to the config file (default: `configs/default.yaml`).

At every completed epoch, the trainer writes `latest_model_<stem>.pth`, updates
`best_model_<stem>.pth` when validation cSDR improves, and appends metrics to
`training_metrics.jsonl`. Starting the same command again automatically resumes
from the latest compatible checkpoint. Checkpoints saved with a different loss
objective or an incompatible architecture are rejected.


---

## Architecture constraints

When customizing the model parameters in `configs/default.yaml`, you must adhere to the following architectural and mathematical constraints to prevent runtime/dimension mismatch errors:

### 1. `num_bands` & `G` Constraints

#### Band grouping
* `G % num_bands == 0`; `G` is the total first-stage width.
* `freq_dim % num_bands == 0`; the paper preset uses `2048 / 4 = 512` bins per band.

#### 🔴 Attention Head Division (Mandatory when `bottleneck_type: 'attention'`)
* **Paper-v3 rule**: `(G * (num_layers + 1)) % num_heads == 0`.
* The bottleneck width is `G * (num_layers + 1)` and must be divisible by `num_heads`.

### 2. Audio Config Constraints

#### Frequency range
* The paper preset processes the first 2048 STFT bins and zero-pads the remainder for iSTFT.

#### 🟡 STFT Alignment
* **Rule**: `win_length <= n_fft`
* **Note**: `hop_length` is typically recommended to be a power-of-two divisor of `win_length` (e.g., `win_length // 4` or `win_length // 6`) to maintain smooth overlapping windows.

### 3. Active depth parameters (`num_layers` & `num_dec_layers`)

⚠️ **Important Note on Model Depth Parameters**:
In the v5 implementation these parameters actively construct the encoder and heavy decoder stages.

* `num_layers` controls the number of encoder/downsampling stages.
* `num_dec_layers` controls heavy decoder stages; remaining stages use lightweight
  post-fusion refinement instead of passing skip products through unchanged.
* `n_rope` controls dual-path RoPE bottleneck depth.

## 📊 Legacy Evaluation Results

The result below belongs to the legacy four-layer RNN checkpoint. It is not a
measurement of the paper-aligned RoPE model and cannot be compared directly to
the Moises-Light Proposed/Proposed-S results.

### V1


- Audio conifg

| n_fft | hop | win | duration |
| :--- | :--- | :--- | :--- |
| 6144| 1024 | 6144 | 4.0  |

- Model config

| n_enc_layer | n_dec_layer | n_bands | G | n_rope | n_head |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3| 1 | 4 | 32 | 6 | 8 |

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
