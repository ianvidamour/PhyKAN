#!/bin/bash
#SBATCH -t 3-08:00:00
#SBATCH --mem=40G
#SBATCH --array=0-24
PATH="$HOME/.conda/envs/:$PATH"
module load Anaconda3/2022.10
source activate pytorch
python KAN_CartPole_RL.py $SLURM_ARRAY_TASK_ID
