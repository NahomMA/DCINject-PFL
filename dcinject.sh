#!/bin/bash
#SBATCH --job-name=msba_attack_2
#SBATCH -p reserved-takabi
#SBATCH --qos=takabi
#SBATCH --account=takabi
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/msba_attack_2_output.log
#SBATCH --error=./logs/msba_attack_2_error.log
#SBATCH --nodelist=e2-w8545-01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=closest

echo "Starting script"

echo "Loading required modules"
module load cuda/11.8.0 

echo "Activating conda environment"
eval "$(conda shell.bash hook)"
conda activate AIsec

echo "Checking GPU status"
nvidia-smi

echo "Running training script"
python msba_main.py 

echo "Script completed"