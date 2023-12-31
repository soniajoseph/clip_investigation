#!/bin/bash
#SBATCH --array=2,3,4,5,6,8,9
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24Gb
#SBATCH --time=2:00:00
###SBATCH --ntasks=16
#SBATCH --output=get_activations.%A.%a.out
#SBATCH --error=get_activations.%A.%a.err
#SBATCH --job-name=get_activations

module load anaconda/3
module load cuda/11.7
module load libffi

source /home/mila/s/sonia.joseph/ViT-Planetarium/env/bin/activate

python get_activations.py --layer_num $SLURM_ARRAY_TASK_ID --module_name 'fc2'