#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=data-prep
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# setup python env
module purge
module load CUDA
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chess_ml



# Download Datasets
echo "Downloading lichess puzzles"
curl -L -o data/lichess_puzzles.zip\
  https://www.kaggle.com/api/v1/datasets/download/tianmin/lichess-chess-puzzle-dataset
unzip data/lichess_puzzles.zip -d data

echo "Downloading gm games"
curl -L -o data/gm_games.zip \
  https://www.kaggle.com/api/v1/datasets/download/dimitrioskourtikakis/gm-games-chesscom
unzip data/gm_games.zip -d data


# Transform Datasets
echo "Transforming lichess puzzles"
python -m chess_ml.data.transform_puzzles \
	-i data/lichess_puzzle_transformed.csv \
	-o data/lichess_puzzle_labeled.csv

echo "Transforming gm games"
python -m chess_ml.data.transform_games \
	-i data/GM_games_dataset.csv \
	-o data/gm_games_labeled.csv
