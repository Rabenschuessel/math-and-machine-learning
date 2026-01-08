#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=imitation-learning
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=15:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=8G

# setup python env
module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chess_ml


# start imitation 
python -m chess_ml.train.imitation "$@" \
	--experiment-name 0 \
	--epochs 10
