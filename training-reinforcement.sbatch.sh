#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=ReinforcementLearningChess
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=15:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=4G

# setup python env
module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chess_ml


# start immitation 
python -m chess_ml.train.reinforcement "$@" \
	--experiment 0 
