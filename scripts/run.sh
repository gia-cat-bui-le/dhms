#!/bin/bash
#SBATCH --job-name=job_baseline
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/dhms-baseline/scripts/log_out/training_baseline_1.0/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/dhms-baseline/scripts/log_out/training_baseline_1.0/dhms.err"
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH -w gpu01

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

conda activate dhms

cd /media/nhdang/Vy_Cat/dhms-baseline

python3 train_diffusion.py --save_dir /raid/nhdang/Vy/dhms/pcmdm_baseline_1.0 --dataset aistpp --eval_during_training --refine --inference_dir /raid/nhdang/Vy/data/evaluation_baseline_1.0