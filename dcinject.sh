#!/bin/bash
echo "Starting script"

# echo "Loading required modules"
module load cuda/12.8.0

# echo "Activating conda environment"
conda activate dcinject-gpu

# echo "Checking GPU status"
nvidia-smi

echo "Running training script"
python dcinject.py --dataset CIFAR10 --attack_type dcinject --poison_ratio 0.2 --attack_budget 0.03 --trigger_type frequency
echo "Script completed"