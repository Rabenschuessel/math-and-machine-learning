#!/bin/bash
#SBATCH -o ./log/%x.out
#SBATCH -e ./log/%x.err
#SBATCH --job-name=reinforcement-learning
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=16G


#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export KMP_AFFINITY=disabled


# setup python env
module purge
module load Anaconda3
modul avail
eval "$(conda shell.bash hook)"
conda activate chess_ml
# Change to the script directory
#cd "$(dirname "$0")"

# Generate unique experiment name using SLURM job ID
EXPERIMENT_NAME="job-${SLURM_JOB_ID}"

# start reinforcement learning with unique folder per submission
python -m chess_ml.train.reinforcement "$@" \
	--experiment "${EXPERIMENT_NAME}" 
