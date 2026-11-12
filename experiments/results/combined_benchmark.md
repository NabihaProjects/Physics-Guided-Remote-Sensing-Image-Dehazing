# Remote Sensing Image Dehazing: Unified Benchmark & Evaluation Results

Based on the evaluation criteria of Section 4.2 (*ISPRS Journal of Photogrammetry and Remote Sensing, 2026*).

- **Hardware Platform**: Apple Silicon MPS / CPU Fallback
- **Efficiency Profile Resolution**: Native $256 \times 256$ on Apple Silicon MPS
- **RICE1 Protocol**: Cross-domain, zero-shot transfer evaluated at native resolution ($512 \times 512$) without fine-tuning.
- **Haze1k Protocol**: Tested on official test splits at native resolution ($512 \times 512$).
- **RRSHID Protocol**: Tested on official test splits at native resolution ($256 \times 256$) under real haze conditions.

## 1. Full-Reference Benchmark Performance

| Dataset | Type | Model | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | SAM (°) ↓ | ERGAS ↓ | UIQI ↑ | QNR ↑ | HIST ↑ | LPIPS ↓ | NIQE ↓ |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Haze1k_Thin** | In-Domain (512x512) | DCP (He et al. 2010) | **18.88** | **0.8762** | 10.12 | 5.82 | 21.81 | 0.8761 | **0.4499** | 0.8895 | 0.1193 | 4.07 |
| **Haze1k_Thin** | In-Domain (512x512) | AOD-Net (CNN Baseline) | **18.08** | **0.8478** | 10.49 | 5.07 | 24.06 | 0.8494 | **0.4748** | 0.8142 | 0.1793 | 4.65 |
| **Haze1k_Thin** | In-Domain (512x512) | PhysDehaze-Lite (Proposed Novel) | **20.98** | **0.8945** | 7.46 | 2.79 | 17.33 | 0.9282 | **0.6370** | 0.8890 | 0.0903 | 4.25 |
| **Haze1k_Moderate** | In-Domain (512x512) | DCP (He et al. 2010) | **18.79** | **0.8998** | 11.33 | 7.44 | 25.96 | 0.8352 | **0.3697** | 0.8654 | 0.1376 | 4.01 |
| **Haze1k_Moderate** | In-Domain (512x512) | AOD-Net (CNN Baseline) | **19.28** | **0.8851** | 11.64 | 4.14 | 24.61 | 0.8279 | **0.5188** | 0.8329 | 0.1958 | 4.70 |
| **Haze1k_Moderate** | In-Domain (512x512) | PhysDehaze-Lite (Proposed Novel) | **21.46** | **0.9109** | 10.75 | 4.24 | 19.39 | 0.9117 | **0.5431** | 0.8545 | 0.1181 | 4.25 |
| **Haze1k_Thick** | In-Domain (512x512) | DCP (He et al. 2010) | **14.11** | **0.7559** | 17.66 | 6.42 | 37.77 | 0.7628 | **0.3617** | 0.7314 | 0.2471 | 3.71 |
| **Haze1k_Thick** | In-Domain (512x512) | AOD-Net (CNN Baseline) | **14.52** | **0.7170** | 15.29 | 3.85 | 36.08 | 0.7075 | **0.4757** | 0.6977 | 0.2588 | 3.76 |
| **Haze1k_Thick** | In-Domain (512x512) | PhysDehaze-Lite (Proposed Novel) | **18.14** | **0.8075** | 10.05 | 3.25 | 23.68 | 0.8672 | **0.5712** | 0.8000 | 0.2038 | 4.47 |
| **RICE1_CrossDomain_ZeroShot** | Cross-Domain Zero-Shot (512x512) | DCP (He et al. 2010) | **16.28** | **0.7501** | 17.48 | 9.37 | 43.10 | 0.5675 | **0.2826** | 0.3727 | 0.3127 | 4.86 |
| **RICE1_CrossDomain_ZeroShot** | Cross-Domain Zero-Shot (512x512) | AOD-Net (CNN Baseline) | **16.40** | **0.7159** | 16.19 | 7.20 | 46.19 | 0.2722 | **0.3445** | 0.2515 | 0.3096 | 5.75 |
| **RICE1_CrossDomain_ZeroShot** | Cross-Domain Zero-Shot (512x512) | PhysDehaze-Lite (Proposed Novel) | **11.48** | **0.5039** | 32.13 | 13.12 | 67.26 | 0.3058 | **0.1792** | 0.1348 | 0.4953 | 7.15 |
| **RRSHID_Thin** | Real Haze Zero-Shot (256x256) | DCP (He et al. 2010) | **17.58** | **0.4134** | 14.49 | 21.51 | 74.40 | 0.4724 | **0.0655** | 0.5702 | 0.4096 | 5.95 |
| **RRSHID_Thin** | Real Haze Zero-Shot (256x256) | AOD-Net (CNN Baseline) | **14.57** | **0.3804** | 20.68 | 14.93 | 109.94 | 0.1473 | **0.1086** | 0.2015 | 0.6028 | 6.08 |
| **RRSHID_Thin** | Real Haze Zero-Shot (256x256) | PhysDehaze-Lite (Proposed Novel) | **21.01** | **0.5441** | 8.45 | 6.41 | 50.85 | 0.5808 | **0.3373** | 0.5999 | 0.4158 | 6.28 |
| **RRSHID_Moderate** | Real Haze Zero-Shot (256x256) | DCP (He et al. 2010) | **15.05** | **0.3976** | 21.09 | 22.53 | 84.79 | 0.4005 | **0.0563** | 0.3847 | 0.5391 | 6.35 |
| **RRSHID_Moderate** | Real Haze Zero-Shot (256x256) | AOD-Net (CNN Baseline) | **12.55** | **0.4355** | 26.43 | 20.64 | 116.87 | 0.2577 | **0.0858** | 0.1537 | 0.6495 | 7.99 |
| **RRSHID_Moderate** | Real Haze Zero-Shot (256x256) | PhysDehaze-Lite (Proposed Novel) | **21.01** | **0.5867** | 8.98 | 5.06 | 42.12 | 0.5797 | **0.4148** | 0.4653 | 0.4378 | 6.51 |
| **RRSHID_Thick** | Real Haze Zero-Shot (256x256) | DCP (He et al. 2010) | **12.88** | **0.3464** | 25.76 | 19.27 | 118.13 | 0.3525 | **0.0614** | 0.2332 | 0.5801 | 6.59 |
| **RRSHID_Thick** | Real Haze Zero-Shot (256x256) | AOD-Net (CNN Baseline) | **10.17** | **0.4026** | 32.74 | 16.38 | 158.77 | 0.2852 | **0.0634** | 0.0878 | 0.6309 | 7.83 |
| **RRSHID_Thick** | Real Haze Zero-Shot (256x256) | PhysDehaze-Lite (Proposed Novel) | **22.32** | **0.6309** | 8.24 | 4.50 | 37.29 | 0.6560 | **0.4489** | 0.3943 | 0.4174 | 6.21 |

## 2. Hardware Computational Efficiency (Native 256x256 Input on MPS)

| Model | Trainable Params | Model Size (MB) | FLOPs (GFLOPs) | Profiler Method | Latency (ms) | Throughput (FPS) |
|:---|:---:|:---:|:---:|:---|:---:|:---:|
| **DCP (He et al. 2010)** | 0 | 0.0 MB | **0.0** | N/A (Analytical Prior) | 2.38 ± 0.54 ms | **419.65 FPS** |
| **AOD-Net (CNN Baseline)** | 1,761 | 0.007 MB | **0.229** | torch.utils.flop_counter | 1.62 ± 0.26 ms | **618.92 FPS** |
| **PhysDehaze-Lite (Proposed Novel)** | 323,703 | 1.235 MB | **3.731** | torch.utils.flop_counter | 26.95 ± 5.79 ms | **37.1 FPS** |

## 3. PhysDehaze-Lite Component Ablations (2 Epochs Each)

| Ablation Variant | Loss Formulation & Head Setup | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | SAM (°) ↓ | ERGAS ↓ | QNR ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **(a) Full PhysDehaze-Lite** | L1 + SSIM + Physics + SAM with Learnable A | **20.29** | **0.8892** | 7.64 | 2.78 | 18.27 | **0.6317** |
| **(b) w/o Physics Consistency Loss** | L1 + SSIM + SAM (Physics Loss = 0) | **20.63** | **0.8922** | 7.24 | 2.86 | 17.63 | **0.6306** |
| **(c) w/o Spectral SAM Loss** | L1 + SSIM + Physics (SAM Loss = 0) | **21.21** | **0.8909** | 7.19 | 2.80 | 16.49 | **0.6416** |
| **(d) w/o Airlight Head (Fixed A=1.0)** | Full Loss with Fixed Constant Airlight A=[1.0, 1.0, 1.0] | **21.29** | **0.8913** | 7.78 | 3.28 | 16.23 | **0.6130** |
