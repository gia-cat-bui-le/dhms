#!/bin/bash
#SBATCH --job-name=dhms
#SBATCH --ntasks=1
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/dhms/scripts/log_out/create_dataset/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/dhms/scripts/log_out/create_dataset/dhms.err"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH -w gpu05

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

cd /media/nhdang/Vy_Cat/dhms/data_loaders/d2m

conda activate dhms

# python3 create_dataset.py --extract-baseline --dataset_name aistpp
python3 create_dataset.py --extract-baseline --dataset_name finedance
