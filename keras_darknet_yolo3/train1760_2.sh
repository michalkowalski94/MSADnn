#!/usr/bin/bash
#SBATCH -A neuralcbisddsm
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5GB

export LD_LIBRARY_PATH=/net/people/plgmkowalski94/cudnn/cuda/lib64:$LD_LIBRARY_PATH
module load plgrid/apps/cuda/9.0
module load plgrid/libs/tensorflow-gpu/1.10.0-python-3.6
python3 $SCRATCH/master/new/keras-yolo3/1760train_2.py
