#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=test-env
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=00:05:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=8G

# setup python env
module load CUDA
module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chess_ml

nvidia-smi
python -c 'import torch; print("torch cuda version: {}".format(torch.version.cuda))'
python -c 'import torch; print("torch available: {}".format(torch.cuda.is_available()))'
