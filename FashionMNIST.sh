#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -t 0-08:00:00
#SBATCH --mem=40G
#SBATCH --array=0-126
PATH="$HOME/.conda/envs/:$PATH"
module load Anaconda3/2022.10
module load cuDNN/8.7.0.84-CUDA-11.8.0
source activate pytorch
python PhyKAN_FMNIST_HPC.py $SLURM_ARRAY_TASK_ID
