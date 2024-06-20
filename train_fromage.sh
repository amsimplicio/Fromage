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
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1  # Possible values: gpu:hubgpu:<count> or gpu:nvidia_a100-pcie-40gb:<count> or gpu:nvidia_a100-sxm4-40gb:<count>
# Setup anaconda
eval "$(conda shell.bash hook)"

# conda activate fromage # activate desired environment
conda activate fromage
#conda activate requeijao
randport=$(shuf -i8000-9999 -n1) 
python -u main.py \
    --world-size 1 --rank 0 --dist-url "tcp://127.0.0.1:${randport}" \
    --dataset=cc3m --val-dataset=cc3m \
    --opt-version='google/gemma-2b-it' --visual-model='openai/clip-vit-large-patch14' \
    --exp_name='Gemma' --image-dir='/storagebk/datasets'  --log-base-dir='runs/' \
    --learning-rate=0.0003 --precision='bf16'  --print-freq=10 \
    --batch-size=90 --val-batch-size=100 --grad-accumulation-steps=2 \
    --workers=4 --project='Train_Gemma' --run-name='layer-2' --cls-layer=-2 

