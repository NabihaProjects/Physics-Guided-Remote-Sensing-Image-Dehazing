# Remote Sensing Datasets Directory

This directory contains the benchmark datasets for remote sensing image dehazing: **Haze1k** (SateHaze1k), **RICE**, and **RRSHID**.

> **Note on Base Paper Datasets vs. Repository Datasets:**  
> The base review paper (*ISPRS Journal of Photogrammetry and Remote Sensing, 2026*) benchmarks across **five datasets**:
> 1. **SateHaze1k** (Synthetic satellite haze)
> 2. **RICE** (Real-world cloud/haze removal: RICE1 / RICE2)
> 3. **LHID** (Light Haze Image Dataset)
> 4. **DHID** (Dense Haze Image Dataset)
> 5. **RRSHID** (Real-world remote sensing haze)
>
> This repository implements and benchmarks **three of these datasets: SateHaze1k, RICE, and RRSHID**.  
> **LHID and DHID were not used** because their only public download host is on **Baidu Netdisk (Baidu Wangpan)**, which was not accessible due to regional access constraints and download restrictions.  
> In addition, while all 3,053 pairs of **RRSHID** are present, verified, and their official test splits evaluated, **RRSHID training was conducted using a capped 1,000-pair subset** (seed 42) combined with Haze1k (1,960 total training pairs).

---

## 1. Actual Folder Structure on Disk

```
data/
├── Haze1k/
│   └── Haze1k/
│       ├── Haze1k_thin/
│       │   ├── dataset/
│       │   │   ├── train/
│       │   │   │   ├── input/   (320 hazy images: e.g. 1-inputs.png)
│       │   │   │   └── target/  (320 clear images: e.g. 1-targets.png)
│       │   │   ├── valid/
│       │   │   │   ├── input/   (35 hazy images: e.g. 321-inputs.png)
│       │   │   │   └── target/  (35 clear images: e.g. 321-targets.png)
│       │   │   └── test/
│       │   │       ├── input/   (45 hazy images: e.g. 356-inputs.png)
│       │   │       └── target/  (45 clear images: e.g. 356-targets.png)
│       │   └── SAR/ (train/, val/, test/)
│       ├── Haze1k_moderate/
│       │   ├── dataset/
│       │   │   ├── train/
│       │   │   │   ├── input/   (320 hazy images: e.g. 1.png)
│       │   │   │   └── target/  (320 clear images: e.g. 1.png)
│       │   │   ├── val/
│       │   │   │   ├── input/   (35 hazy images: e.g. 1.jpg)
│       │   │   │   └── target/  (35 clear images: e.g. 1.jpg)
│       │   │   └── test/
│       │   │       ├── input/   (45 hazy images: e.g. 1-inputs.png)
│       │   │       └── target/  (45 clear images: e.g. 1-targets.png)
│       │   └── SAR/ (train/, val/, test/)
│       └── Haze1k_thick/
│           ├── dataset/
│           │   ├── train/
│           │   │   ├── input/   (320 hazy images: e.g. 1-inputs.png)
│           │   │   └── target/  (320 clear images: e.g. 1-targets.png)
│           │   ├── valid/
│           │   │   ├── input/   (35 hazy images: e.g. 321-inputs.png)
│           │   │   └── target/  (35 clear images: e.g. 321-targets.png)
│           │   └── test/
│           │       ├── input/   (45 hazy images: e.g. 356-inputs.png)
│           │       └── target/  (45 clear images: e.g. 356-targets.png)
│           └── SAR/ (train/, val/, test/)
│
├── RICE_DATASET/
│   ├── RICE1/
│   │   ├── cloud/ (500 cloudy/hazy images: 0.png - 499.png, 512x512 RGB)
│   │   └── label/ (500 ground-truth clear images: 0.png - 499.png, 512x512 RGB)
│   └── RICE2/
│       ├── cloud/ (736 images)
│       ├── mask/  (736 images)
│       └── label/ (736 images)
│
└── RRSHID/ (Real-World Remote Sensing Hazy Image Dataset, 3,053 pairs, 256x256 RGB)
    ├── thin_fog/
    │   ├── train/
    │   │   ├── hazy/ (610 images)
    │   │   └── gt/   (610 images)
    │   ├── val/
    │   │   ├── hazy/ (77 images)
    │   │   └── GT/   (77 images)
    │   └── test/
    │       ├── hazy/ (76 images)
    │       └── gt/   (76 images)
    ├── moderate_fog/
    │   ├── train/
    │   │   ├── hazy/  (1,220 images)
    │   │   └── clear/ (1,220 images)
    │   ├── val/
    │   │   ├── hazy/  (154 images)
    │   │   └── GT/    (154 images)
    │   └── test/
    │       ├── hazy/  (152 images)
    │       └── clear/ (152 images)
    └── thick_fog/
        ├── train/
        │   ├── hazy/  (611 images)
        │   └── clear/ (611 images)
        ├── val/
        │   ├── hazy/  (77 images)
        │   └── GT/    (77 images)
        └── test/
            ├── hazy/  (76 images)
            └── clear/ (76 images)
```

---

## 2. Summary of Verified Pairs

| Dataset | Subset | Split | Input Folder | Target Folder | Verified Pairs | Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Haze1k** | `thin` | `train` | `input/` | `target/` | **320** | 512 x 512 |
| | | `valid` | `input/` | `target/` | **35** | 512 x 512 |
| | | `test` | `input/` | `target/` | **45** | 512 x 512 |
| | | **Subtotal** | | | **400 pairs** | |
| **Haze1k** | `moderate` | `train` | `input/` | `target/` | **320** | 512 x 512 |
| | | `val` | `input/` | `target/` | **35** | 512 x 512 |
| | | `test` | `input/` | `target/` | **45** | 512 x 512 |
| | | **Subtotal** | | | **400 pairs** | |
| **Haze1k** | `thick` | `train` | `input/` | `target/` | **320** | 512 x 512 |
| | | `valid` | `input/` | `target/` | **35** | 512 x 512 |
| | | `test` | `input/` | `target/` | **45** | 512 x 512 |
| | | **Subtotal** | | | **400 pairs** | |
| **Haze1k Total** | | | | | **1,200 pairs** | 512 x 512 |
| **RICE1** | Full | Full | `cloud/` | `label/` | **500 pairs** | 512 x 512 |
| **RRSHID** | `thin` (`TN`) | `train` | `hazy/` | `gt/` | **610** | 256 x 256 |
| | | `val` | `hazy/` | `GT/` | **77** | 256 x 256 |
| | | `test` | `hazy/` | `gt/` | **76** | 256 x 256 |
| | | **Subtotal** | | | **763 pairs** | |
| **RRSHID** | `moderate` (`M`) | `train` | `hazy/` | `clear/` | **1,220** | 256 x 256 |
| | | `val` | `hazy/` | `GT/` | **154** | 256 x 256 |
| | | `test` | `hazy/` | `clear/` | **152** | 256 x 256 |
| | | **Subtotal** | | | **1,526 pairs** | |
| **RRSHID** | `thick` (`TK`) | `train` | `hazy/` | `clear/` | **611** | 256 x 256 |
| | | `val` | `hazy/` | `GT/` | **77** | 256 x 256 |
| | | `test` | `hazy/` | `clear/` | **76** | 256 x 256 |
| | | **Subtotal** | | | **764 pairs** | |
| **RRSHID Total** | | | | | **3,053 pairs** | 256 x 256 |
| **Grand Total Across All Datasets** | | | | | **4,753 pairs** | |

---

## 3. How to Run the Verification Tool

To verify pair matching, dimensions, and image integrity across all datasets:
```bash
python scripts/verify_dataset.py --data-dir data/
```

Or for a specific dataset:
```bash
python scripts/verify_dataset.py --data-dir data/RRSHID
python scripts/verify_dataset.py --data-dir data/Haze1k
python scripts/verify_dataset.py --data-dir data/RICE_DATASET/RICE1
```

To run with Python programmatic API:
```python
from src.datasets import RRSHIDDataset

# Load thin haze test split
ds_thin_test = RRSHIDDataset(root_dir="data/RRSHID", subset="thin", split="test")
sample = ds_thin_test[0]
print(sample["hazy"].shape, sample["clear"].shape)  # torch.Size([3, 256, 256])
```
