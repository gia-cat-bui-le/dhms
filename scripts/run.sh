#!/bin/bash
#SBATCH --job-name=job
#SBATCH --ntasks=1
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/newPCMDM/scripts/log_out/training/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/newPCMDM/scripts/log_out/training/dhms.err"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH -w gpu05

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

cd /media/nhdang/Vy_Cat/dhms

conda activate dhms

python3 train_diffusion.py --save_dir ./save/pcmdm --dataset aistpp --hist_frames 75
