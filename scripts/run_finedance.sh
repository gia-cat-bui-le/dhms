#!/bin/bash
#SBATCH --job-name=dhms
#SBATCH --ntasks=1
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/dhms/scripts/log_out/training_finedance/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/dhms/scripts/log_out/training_finedance/dhms.err"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=64
#SBATCH -w gpu05

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

cd /media/nhdang/Vy_Cat/dhms

conda activate dhms

python3 train_diffusion.py --save_dir ./save/pcmdm_finedance --dataset finedance --hist_frames 75
