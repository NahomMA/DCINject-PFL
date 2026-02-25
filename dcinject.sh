#!/bin/bash
#SBATCH --job-name=dcinject_attack_2
#SBATCH -p reserved-takabi
#SBATCH --qos=takabi
#SBATCH --account=takabi
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/dcinject_attack_2_output.log
#SBATCH --error=./logs/dcinject_attack_2_error.log
#SBATCH --nodelist=e7-w760xa-01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=closest

echo "Starting script"

# echo "Loading required modules"
# module load cuda/12.8.0

# echo "Activating conda environment"
# conda activate dcinject-gpu

# echo "Checking GPU status"
# nvidia-smi

echo "Running training script"
python dcinject.py --dataset CIFAR10 --attack_type dcinject --poison_ratio 0.2 --attack_budget 0.03 --trigger_type frequency
echo "Script completed"