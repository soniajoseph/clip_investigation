#!/bin/bash
#SBATCH --array=0-9
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128Gb
#SBATCH --time=50:00
###SBATCH --ntasks=16
#SBATCH --output=sbatch_out/automate_ablations.%A.%a.out
#SBATCH --error=sbatch_err/automate_ablations.%A.%a.err
#SBATCH --job-name=automate_ablations

module load anaconda/3
module load cuda/11.7
module load libffi

source /home/mila/s/sonia.joseph/ViT-Planetarium/env/bin/activate

python automate_ablations.py --layer_num $SLURM_ARRAY_TASK_ID --layer_type "fc2"