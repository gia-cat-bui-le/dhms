#!/bin/bash
#SBATCH --job-name=dhms
#SBATCH --ntasks=1
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/dhms/scripts/log_out/generate/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/dhms/scripts/log_out/generate/dhms.err"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH -w gpu05

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

cd /media/nhdang/Vy_Cat/dhms/

conda activate dhms

python3 generate.py --model_path ./save/pcmdm_cont/model000100000.pt --dataset aistpp
# python3 ./evaluation/fid.py