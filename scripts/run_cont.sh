#!/bin/bash
#SBATCH --job-name=job
#SBATCH --ntasks=1
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/dhms-baseline/scripts/log_out/training_baseline_cont/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/dhms-baseline/scripts/log_out/training_baseline_cont/dhms.err"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=40
#SBATCH -w gpu05

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

cd /media/nhdang/Vy_Cat/dhms

conda activate dhms

python3 train_diffusion.py --save_dir /raid/nhdang/Vy/dhms/pcmdm_baseline_cont --dataset aistpp --hist_frames 30 --overwrite --resume_checkpoint /raid/nhdang/Vy/dhms/pcmdm_cont/model000110000.pt --resume_step