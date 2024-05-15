#!/bin/bash
#SBATCH --job-name=baseline-sinmdm
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/baseline-sinmdm/scripts/log_out/training_baseline_inbetween/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/baseline-sinmdm/scripts/log_out/training_baseline_inbetween/dhms.err"
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH -w gpu01

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

conda activate dhms

cd /media/nhdang/Vy_Cat/baseline-sinmdm

python3 train_diffusion.py --save_dir /raid/nhdang/Vy/dhms/baseline-sinmdm --dataset aistpp --eval_during_training --inference_dir /raid/nhdang/Vy/databaseline-sinmdm