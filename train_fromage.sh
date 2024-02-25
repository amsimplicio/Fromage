#!/bin/bash
#SBATCH --job-name=fromage # Job name
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 4 # Number of cpu cores
#SBATCH --mem=200GB # Memory - Use up to 2GB per requested CPU as a rule of thumb
#sbatch --time=2-10:00:00 # Time limit in the form days-hours:minutes:seconds
# SBATCH --exclude=novasearchdl  # Exclude nodes with these names (comma-separated)
# SBATCH --partition=students # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:3  # Possible values: gpu:hubgpu:<count> or gpu:nvidia_a100-pcie-40gb:<count> or gpu:nvidia_a100-sxm4-40gb:<count>
# Setup anaconda
eval "$(conda shell.bash hook)"

# conda activate fromage # activate desired environment
conda activate mistral
python -u main.py \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dataset=cc3m  --val-dataset=cc3m \
    --opt-version='mistralai/Mistral-7B-Instruct-v0.1' --visual-model='openai/clip-vit-large-patch14' \
    --exp_name='Mistral' --image-dir='/storagebk/datasets'  --log-base-dir='runs/' \
    --learning-rate=0.0003 --precision='bf16'  --print-freq=10 \
    --batch-size=180  --val-batch-size=100 \
    --workers=4 --project='Train_Mistral' --run-name='original_bs' #--resume='runs/fromage_exp_42/ckpt_best.pth.tar'

