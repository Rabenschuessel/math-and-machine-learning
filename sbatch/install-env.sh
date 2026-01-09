#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=install-env
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=01:00:00
#SBATCH --mem=8G

# setup python env
module purge
module load CUDA
module load Anaconda3
eval "$(conda shell.bash hook)"
conda env create -f environment.yml -y
conda activate chess_ml

nvidia-smi
python -c 'import torch; print("torch cuda version: {}".format(torch.version.cuda))'
python -c 'import torch; print("torch available: {}".format(torch.cuda.is_available()))'
