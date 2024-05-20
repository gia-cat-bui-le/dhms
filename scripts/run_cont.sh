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

source miniconda3/etc/profile.d/conda.sh

conda activate dhms

cd baseline-inbetweening

python3 train_diffusion.py --save_dir baseline-inbetweening/save --dataset aistpp --hist_frames 30 --overwrite --resume_checkpoint /raid/nhdang/Vy/dhms/pcmdm_cont/model000110000.pt --resume_step