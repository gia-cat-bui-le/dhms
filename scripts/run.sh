#!/bin/bash
#SBATCH --job-name=baseline-inbetweening
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="baseline-inbetweening/scripts/log_out/training_baseline_inbetween/dhms.out"
#SBATCH --error="baseline-inbetweening/scripts/log_out/training_baseline_inbetween/dhms.err"
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH -w gpu01

source miniconda3/etc/profile.d/conda.sh

conda activate dhms

cd baseline-inbetweening

python3 train_diffusion.py --save_dir baseline-inbetweening/save --dataset aistpp --eval_during_training --inference_dir data/baseline-inbetweening