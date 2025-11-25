#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=TestPytorch
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=01:30:00
#SBATCH --gpus=rtx2080ti:1

# setup python env
module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chess_ml


# start immitation 
python -m chess_ml.train.immitation \
	--experiment 1 \
	--epochs 1
