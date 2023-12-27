#!/bin/bash
#SBATCH --array=0-9
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=120Gb
#SBATCH --time=3:30:00
###SBATCH --ntasks=16
#SBATCH --output=sbatch_out/get_max_min_images.%A.%a.out
#SBATCH --error=sbatch_err/get_max_min_images.%A.%a.err
#SBATCH --job-name=get_max_min_images

module load anaconda/3
module load cuda/11.7
module load libffi

source /home/mila/s/sonia.joseph/ViT-Planetarium/env/bin/activate
python get_maximally_activating_images.py --layer_num $SLURM_ARRAY_TASK_ID 