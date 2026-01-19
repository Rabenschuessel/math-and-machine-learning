#!/bin/bash
#SBATCH -o ./log/%x-%j.out
#SBATCH -e ./log/%x-%j.err
#SBATCH --job-name=imitation-learning
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=30:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=32G

# setup python env
module purge
module load Anaconda3
module load CUDA
eval "$(conda shell.bash hook)"
conda activate chess_ml


# start imitation 
python -m chess_ml.train.imitation "$@" 
