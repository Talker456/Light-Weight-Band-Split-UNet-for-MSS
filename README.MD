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


---

## ⚠️ Hyperparameter Customization Constraints

When customizing the model parameters in `configs/default.yaml`, you must adhere to the following architectural and mathematical constraints to prevent runtime/dimension mismatch errors:

### 1. `num_bands` & `G` Constraints

#### 🔴 GroupNorm Division (Mandatory)
* **Rule**: `(G * num_bands) % 16 == 0`
* **Reason**: The asymmetric decoder uses a fixed `nn.GroupNorm(16, ...)` layer. The output channels of this stage (`G * num_bands`) must be divisible by `16`.
* **Example**: 
  * ✅ `num_bands: 4, G: 8` (Product = 32)
  * ✅ `num_bands: 2, G: 8` (Product = 16)
  * ❌ `num_bands: 3, G: 8` (Product = 24) $\rightarrow$ *Throws GroupNorm division error*

#### 🔴 Attention Head Division (Mandatory when `bottleneck_type: 'attention'`)
* **Rule**: `(4 * G * num_bands) % num_heads == 0`
* **Reason**: Inside `RoPEAttention`, the bottleneck channel dimension (`4 * G * num_bands`) is divided by `num_heads` to calculate `head_dim`. 
* **Example** (with `num_heads: 8`):
  * ✅ `num_bands: 4, G: 8` (Bottleneck channels = 128, divisible by 8)
  * ❌ `num_bands: 3, G: 5` (Bottleneck channels = 60, not divisible by 8) $\rightarrow$ *Throws shape mismatch error*

### 2. Audio Config Constraints

#### 🟡 Maximum `num_bands` Limit
* **Rule**: `1 <= num_bands <= (n_fft // 2 + 1)`
* **Reason**: The spectrogram frequency dimension size is determined by `(n_fft // 2 + 1)`. The number of bands cannot exceed the number of available frequency bins. 
* **Note**: It is highly recommended to use powers of 2 (e.g., 2, 4, 8, 16) for optimal GPU memory alignment and subband splitting efficiency.

#### 🟡 STFT Alignment
* **Rule**: `win_length <= n_fft`
* **Note**: `hop_length` is typically recommended to be a power-of-two divisor of `win_length` (e.g., `win_length // 4` or `win_length // 6`) to maintain smooth overlapping windows.

### 3. Dummy Config Parameters (`num_layers` & `num_dec_layers`)

⚠️ **Important Note on Model Depth Parameters**:
In the current codebase, `num_layers` and `num_dec_layers` in `configs/default.yaml` **do not affect the actual model depth** and behave as dummy variables.

* **`num_layers` (Configured to 3)**: The encoder's downsampling layers (`enc1`, `enc2`, `enc3`) are hardcoded to 3 layers in `LightRoformerEncoder`. Changing this value in the configuration file will not dynamically alter the encoder's depth.
* **`num_dec_layers` (Configured to 1)**: The decoder is hardcoded to a single-stage upsampling structure in `LightRoformerAsymmetricDecoder`.
* **How to change model depth**: If you want to customize the model's sequence modeling depth (the number of RoPE/RNN blocks in the bottleneck), you must adjust the **`n_rope`** parameter instead.

---


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
