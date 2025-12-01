#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=data-prep
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=02:00:00
#SBATCH --mem=8G

# setup python env
module purge
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chess_ml


# start immitation 
python -m chess_ml.data.transform \
	-i data/lichess_puzzle_transformed.csv \
	-o data/lichess_puzzle_labeled.csv
