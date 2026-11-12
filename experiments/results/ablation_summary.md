# PhysDehaze-Lite Ablation Study

All ablation variants trained for **2 epochs** on the combined Haze1k training set under identical initial conditions.

| Variant | Description | Epochs | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | SAM (°) ↓ | ERGAS ↓ | QNR ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(a) Full PhysDehaze-Lite** | L1 + SSIM + Physics + SAM with Learnable A | 2 | **20.29** | **0.8892** | 7.64 | 2.78 | 18.27 | **0.6317** |
| **(b) w/o Physics Consistency Loss** | L1 + SSIM + SAM (Physics Loss = 0) | 2 | **20.63** | **0.8922** | 7.24 | 2.86 | 17.63 | **0.6306** |
| **(c) w/o Spectral SAM Loss** | L1 + SSIM + Physics (SAM Loss = 0) | 2 | **21.21** | **0.8909** | 7.19 | 2.80 | 16.49 | **0.6416** |
| **(d) w/o Airlight Head (Fixed A=1.0)** | Full Loss with Fixed Constant Airlight A=[1.0, 1.0, 1.0] | 2 | **21.29** | **0.8913** | 7.78 | 3.28 | 16.23 | **0.6130** |
