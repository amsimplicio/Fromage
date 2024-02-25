#!/bin/bash
#SBATCH --job-name=pruningckpt # Job name
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 8 # Number of cpu cores
#SBATCH --mem=50GB # Memory - Use up to 2GB per requested CPU as a rule of thumb
#sbatch --time=0-10:00:00 # Time limit in the form days-hours:minutes:seconds
#SBATCH --exclude=novasearchdl  # Exclude nodes with these names (comma-separated)
# SBATCH --partition=students # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1  # Possible values: gpu:hubgpu:<count> or gpu:nvidia_a100-pcie-40gb:<count> or gpu:nvidia_a100-sxm4-40gb:<count>

# Setup anaconda
eval "$(conda shell.bash hook)"

conda activate mistral # activate desired environment

python -u prune_model_ckpt.py