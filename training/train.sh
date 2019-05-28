#!/usr/bin/bash
#SBATCH -A neuralcbisddsm
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=14GB

module load plgrid/tools/python/3.6.5
export LD_LIBRARY_PATH=/net/people/<username>/cudnn/cuda/lib64:$LD_LIBRARY_PATH
module load plgrid/apps/cuda/9.0
python3 $SCRATCH<path>/training/train.py
