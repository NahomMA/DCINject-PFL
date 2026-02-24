# DCInject: Adaptive Frequency-Domain Backdoor Attack for PFL

Official implementation of DCInject, a novel frequency-domain backdoor attack for personalized federated learning.

## Abstract

Personalized federated learning (PFL) creates client-specific models to handle data heterogeneity. Previously, PFL has been shown to be naturally resistant to backdoor attack propagation across clients. In this work, we reveal that PFL remains vulnerable to backdoor attacks through a novel frequency-domain approach. We propose **DCInject**, an adaptive frequency-domain backdoor attack for PFL, which removes portions of the zero-frequency (DC) component and replaces them with Gaussian-distributed samples in the frequency domain. Our attack achieves superior attack success rates while maintaining clean accuracy across four datasets (CIFAR-10/100, GTSRB, SVHN) compared to existing spatial-domain attacks, evaluated under parameter decoupling based personalization. DCInject achieves superior performance with ASRs of 96.83% (CIFAR-10), 99.38% (SVHN), and 100% (GTSRB) while maintaining clean accuracy. Under I-BAU defense, DCInject demonstrates strong persistence, retaining 90.30% ASR vs BadNet's 58.56% on VGG-16, exposing critical vulnerabilities in PFL security assumptions.

## Requirements

- Python 3.9+
- PyTorch (GPU or CPU)
- numpy, torchvision, pillow (install as needed)

## Installation

Create a minimal conda environment and install dependencies as you encounter missing packages:

**GPU Setup:**
```bash
conda create -n dcinject-gpu python=3.9
conda activate dcinject-gpu
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**CPU Setup:**
```bash
conda create -n dcinject-cpu python=3.9
conda activate dcinject-cpu
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

When you run the script and encounter missing package errors, install them as needed:
```bash
pip install <package-name>
```

Once you've run experiments successfully, export the final dependencies:
```bash
pip freeze > dcinject_requirements.txt
```

## Methodology

The core innovation of DCInject is manipulation in the frequency domain. The attack works by:

1. Converting input images to the frequency domain using FFT
2. Identifying and removing portions of the DC (zero-frequency) component
3. Replacing removed portions with Gaussian-distributed noise
4. Converting back to spatial domain

![Block Diagram](assets/Block-diagram.png)

## Usage

Run DCInject attack on CIFAR-10:

```bash
python dcinject.py --dataset CIFAR10 --attack_type dcinject --poison_ratio 0.2 \
    --attack_budget 0.03 --trigger_type frequency
```

### Key Arguments

- `--dataset`: Dataset name (cifar10, cifar100, gtsrb, svhn)
- `--trigger_type`: Trigger domain (frequency, adaptive_frequency)
- `--poison_ratio`: Poison data ratio (default: 0.2)
- `--attack_budget`: Attack budget (default: 0.03)
- `--client_num`: Number of clients (default: 100)
- `--bad_client_num`: Number of malicious clients (default: 10)

## Results

### Attack Success Rate (Before Defense)

DCInject achieves significantly higher attack success rates across multiple datasets compared to baseline attacks:

![ASR Before Defense](assets/asr_before_fefense.png)

### Robustness Under Defense

Evaluation under I-BAU defense shows DCInject's superior persistence:

![Results After Defense](assets/after_defense.png)

### Ablation Study

The impact of different design choices in DCInject:

![Ablation Study](assets/ablation.png)

## Citation

If you find this work useful, please cite:

```bibtex
@misc{dcinject2026,
  title={{DCInject}: Adaptive frequency-domain backdoor attack for personalized federated learning},
  author={Author et al.},
  year={2026},
  eprint={2602.18489},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.18489}
}
```
