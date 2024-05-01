#!/bin/bash
#SBATCH --job-name=job_finedance
#SBATCH --ntasks=1
#SBATCH --gpus=1             # total number of GPUs
#SBATCH --output="/media/nhdang/Vy_Cat/dhms-baseline/scripts/log_out/training_baseline_finedance/dhms.out"
#SBATCH --error="/media/nhdang/Vy_Cat/dhms-baseline/scripts/log_out/training_baseline_finedance/dhms.err"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH -w gpu01

source /media/nhdang/hieunmt/miniconda3/etc/profile.d/conda.sh

conda activate dhms

# cd /media/nhdang/Vy_Cat/dhms/data_loaders/d2m

# python3 create_dataset.py --dataset_name finedance

cd /media/nhdang/Vy_Cat/dhms

python3 train_diffusion.py --save_dir /raid/nhdang/Vy/dhms/pcmdm_baseline_finedance --dataset finedance --eval_during_training --refine --inference_dir /raid/nhdang/Vy/data/evaluation_baseline_finedance