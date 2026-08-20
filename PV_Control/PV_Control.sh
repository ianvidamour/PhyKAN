#!/bin/bash
#SBATCH -t 3-08:00:00
#SBATCH --mem=40G
#SBATCH --array=0-120
PATH="$HOME/.conda/envs/:$PATH"
module load Anaconda3/2022.10
source activate pytorch
python KAN_PV_Control_continuous_HPC.py $SLURM_ARRAY_TASK_ID
