# Light Weight BS UNet for Music Source Separation

This project is a lightweight, efficient reproduction of the **Roformer-based Music Source Separation** architecture. It focuses on separating music into four stems: **Vocals, Drums, Bass, and Other**.

## 📄 Reference Paper
This implementation is based on the following research:

- **Title**: Moises-Light: Resource-efficient Band-split U-Net For Music Source
- **Authors**: Yun-Ning (Amy) Hung, Igor Pereira, Filip Korzeniowski
- **Source**: https://arxiv.org/pdf/2510.06785


## 📊 Evaluation Results

### 1. L1, intra-track-mixing (default configs)

| STEM | cSDR (Median) | uSDR (Mean) |
| :--- | :--- | :--- |
| **VOCALS** | 2.85 dB | 2.99 dB |
| **DRUMS** | 2.65 dB | 3.29 dB |
| **BASS** | 2.87 dB | 3.15 dB |
| **OTHER** | 2.40 dB | 2.64 dB |

---

### 2. L1, inter-track-mixing (default configs)

| STEM | cSDR (Median) | uSDR (Mean) |
| :--- | :--- | :--- |
| **VOCALS** | 2.52 dB | 2.90 dB |
| **DRUMS** | 2.91 dB | 3.97 dB |
| **BASS** | 2.42 dB | 2.96 dB |
| **OTHER** | 2.32 dB | 2.56 dB |

---

### 3. L1 + MRSTFT (default configs)

| STEM | cSDR (Median) | uSDR (Mean) |
| :--- | :--- | :--- |
| **VOCALS** | -1.81 dB | -1.98 dB |
| **DRUMS** | 1.88 dB | 2.36 dB |
| **BASS** | -1.12 dB | -1.47 dB |
| **OTHER** | -2.43 dB | -2.36 dB |
