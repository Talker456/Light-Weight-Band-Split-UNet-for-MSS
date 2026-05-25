# Light Weight BS UNet for Music Source Separation

This project is a lightweight, efficient reproduction of the **Roformer-based Music Source Separation** architecture. It focuses on separating music into four stems: **Vocals, Drums, Bass, and Other**.

## 📄 Reference Paper
This implementation is based on the following research:

- **Title**: Moises-Light: Resource-efficient Band-split U-Net For Music Source
- **Authors**: Yun-Ning (Amy) Hung, Igor Pereira, Filip Korzeniowski
- **Source**: https://arxiv.org/pdf/2510.06785


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
