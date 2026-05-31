# Test-Time Adaptation via Entropy Minimization
### Reproduction & Improvement of Tent (ICLR 2021)

**Course:** Deep Learning  
**Institution:** KLE Technological University  
**Framework:** PyTorch | **Platform:** Google Colab (T4 GPU)  
**Benchmark:** CIFAR-10-C · ImageNet-C  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Methods Implemented](#methods-implemented)
4. [Results Summary](#results-summary)
5. [Quick Start — Google Colab](#quick-start--google-colab)
6. [Local Setup](#local-setup)
7. [Experiments](#experiments)
8. [Requirements](#requirements)
9. [References](#references)

---

## Project Overview

Deep neural networks degrade under **distribution shift** — when test data comes from a different distribution than training data (e.g. real-world corruptions like noise, blur, fog). This project reproduces and extends **Tent** (Wang et al., ICLR 2021), a lightweight test-time adaptation (TTA) method that minimises prediction entropy to adapt BatchNorm parameters on the fly.

We implement four methods, run six structured experiments, and propose a novel improvement — **Entropy Margin Loss (EML)** — that addresses the overconfidence collapse problem inherent in standard Tent.

| Method | Description |
|---|---|
| **Source** | No adaptation — frozen eval mode (baseline) |
| **Norm** | BatchNorm statistics updated from each test batch |
| **Tent** | Entropy minimisation over BN affine parameters (γ, β) |
| **Improved Tent / EML** | Dual-sided entropy loss preventing both under- and over-adaptation |

---

## Repository Structure

```
DL_courseproject/
│
├── tent_tta_colab.py            # Main experiment script (Experiments 1–6)
├── tent_improved_EML.py         # Improved version with Entropy Margin Loss
│
├── notebooks/
│   └── TTA_Experiments.ipynb   # Google Colab notebook (run this)
│
│
├── plots/
│   ├── viz1_method_comparison.png
│   ├── viz2_heatmap.png
│   ├── viz3_gain.png
│   ├── viz4_severity.png
│   ├── viz5_entropy_dist.png
│   ├── viz6_ablations.png
│   └── viz7_lr_sensitivity.png
│
├── report/
│   ├── TTA_EML_Project_Report.docx
│   └── EML_Justification_Note.docx
│
├── requirements.txt
└── README.md
```

---

## Methods Implemented

### 1. Source (Baseline)
```python
model.eval()  # frozen — no adaptation whatsoever
```
Uses stored BatchNorm running statistics from training. Accuracy collapses at high severity.

### 2. Norm
```python
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.train()           # refresh statistics from each test batch
        m.track_running_stats = False
```
No gradient updates. Purely recalculates μ and σ² from incoming test batches.

### 3. Tent
```python
# Entropy minimisation on BN affine params only
loss = -(p * log_p).sum(dim=1).mean()
loss.backward()
optimizer.step()   # updates only γ and β
```
Shannon entropy H(x) = −Σ p·log(p) is minimised per batch using Adam (lr=1e-3).

### 4. Entropy Margin Loss — EML (Our Improvement)
```python
upper = F.relu(H - tau_high)          # adapt if uncertain
lower = F.relu(tau_low - H)           # resist collapse if overconfident
loss  = (upper - lambda_ * lower).mean()
```
Introduces a **dead zone** (no gradient when τ_low ≤ H ≤ τ_high) and actively pushes back against overconfidence. Unlike confidence filtering (SAR, EATA, COME), this reshapes the **loss function** itself rather than filtering samples.

---

## Results Summary

### Method Comparison (CIFAR-10-C, Severity = 5, All 15 Corruptions)

| Method | Mean Accuracy | Mean Error | Improvement over Source |
|---|---|---|---|
| Source | 56.53% | 43.47% | — |
| Norm | 79.31% | 20.69% | +22.78 pp |
| Tent | 79.81% | 20.19% | +23.28 pp |
| **EML (ours)** | **79.83%** | **20.17%** | **+23.30 pp** |

### Severity Study (Source vs Tent)

| Severity | Source Error | Tent Error | Tent Gain |
|---|---|---|---|
| 1 (mild) | 12.52% | 9.08% | −3.44 pp |
| 3 (medium) | 29.58% | 14.58% | −15.00 pp |
| 5 (severe) | 45.66% | 19.76% | −25.90 pp |

### Learning Rate Study (Tent)

| LR | Mean Accuracy | Mean Error |
|---|---|---|
| 1e-4 | 79.90% | 20.10% |
| **1e-3 ★** | **80.24%** | **19.76%** |
| 1e-2 | 72.48% | 27.52% |

---

## Quick Start — Google Colab

### Option A: Run the notebook directly

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/TTA_Experiments.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run all cells (`Ctrl+F9`)

The notebook auto-installs all dependencies and downloads datasets automatically.

### Option B: Clone and run the script

```bash
# Open a Colab code cell and run:
!git clone https://github.com/Rumanakhaisar8205/DL_courseproject.git
%cd DL_courseproject
!python tent_tta_colab.py
```

---

## Local Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/Rumanakhaisar8205/DL_courseproject.git
cd DL_courseproject
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv tta_env
source tta_env/bin/activate        # Linux / macOS
# tta_env\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run experiments

```bash
# Run all 6 experiments (Source, Norm, Tent, EML)
python tent_tta_colab.py

# Run improved EML version with ablations
python tent_improved_EML.py
```

---

## Experiments

| # | Experiment | Methods | Notes |
|---|---|---|---|
| 1 | Baseline Evaluation | Source | CIFAR-10-C, severity=5, all 15 corruptions |
| 2 | Norm Adaptation | Norm | Same setup, compare with Source |
| 3 | Tent Reproduction | Tent | Main method from Wang et al. 2021 |
| 4 | Learning Rate Study | Tent | LR ∈ {1e-4, 1e-3, 1e-2} |
| 5 | Severity Study | Source vs Tent | Severity ∈ {1, 3, 5} |
| 6 | ImageNet-C | Source vs Tent | ResNet-50, 8 corruptions, severity=5 |
| 7 | EML Ablation — τ_high | EML | τ_high ∈ {0.2, 0.4, 0.6, 0.8} × log(10) |
| 8 | EML Ablation — λ | EML | λ ∈ {0.0, 0.25, 0.5, 1.0} |

---

## Requirements

```
torch>=1.13.0
torchvision>=0.14.0
robustbench>=1.1
timm>=0.9.2
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

> **Note:** GPU is strongly recommended. The full experiment run takes ~25 minutes on a Colab T4 GPU.

---

## References

```
[1] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021).
    Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR 2021.
    https://arxiv.org/abs/2006.10726

[2] Hendrycks, D., & Dietterich, T. (2019).
    Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
    ICLR 2019. https://arxiv.org/abs/1903.12261

[3] Croce, F. et al. (2021). RobustBench: A Standardized Adversarial Robustness Benchmark.
    NeurIPS 2021. https://arxiv.org/abs/2010.09670

[4] Niu, S. et al. (2023). Towards Stable Test-Time Adaptation in Dynamic Wild World.
    ICLR 2023. (SAR) https://arxiv.org/abs/2302.12400

[5] Niu, S. et al. (2022). Efficient Test-Time Model Adaptation without Forgetting.
    ICML 2022. (EATA) https://arxiv.org/abs/2204.02610
```

---

## Citation

If you use this code for academic purposes:

```bibtex
@misc{tta_eml_2025,
  title   = {Test-Time Adaptation via Entropy Margin Loss},
  author  = {Rumana Khaisar},
  year    = {2025},
  note    = {Deep Learning Course Project, KLE Technological University},
  url     = {https://github.com/Rumanakhaisar8205/DL_courseproject}
}
```

---

*Deep Learning Course Project — Academic Year 2024–25*
