#!/bin/bash

architecture='linear'
rewards=('win' 'material')
reward_name=$(IFS=_; printf '%s' "${rewards[*]}")

# install environment when not existing
if ! conda env list | grep -q chess_ml; then 
	echo "Installing Conda Environment"
	dep_env=$(sbatch sbatch/install-env.sh)
fi


# install environment if needed
if [ ! -f data/gm_games_labeled.csv -a ! -f data/lichess_puzzle_labeled.csv ]; then
	echo "Downloading Datasets"
	dep_env=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} sbatch/data-preparation.sh)
fi

# for arc in $architectures; do 

# only pretrain if not already done
dep_pz10=$dep_env
dep_pz20=$dep_env
if [ ! -f logs/im/$architecture-20-epochs/models/checkpoint-best.pth ]; then
	echo "Train Imitation Learning"

	# pretrain with 10 epochs
	dep_pz10=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} \
		--parsable sbatch/imitation-training.sh \
		-n $architecture-10-epochs \
		-e 10 \
		-a $architecture)

	# pretrain with 20 epochs
	dep_pz20=$(sbatch --dependency=afterok:$dep_pz10 --parsable sbatch/imitation-training.sh \
		-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
		-n $architecture-20-epochs \
		-e 10 \
		-a $architecture)
fi

# for reward in $rewards; do
# for reward in $rewards; do

# reinforcement learning with newly initialized model
sbatch ${dep_env:+--dependency=afterok:$dep_env} \
	sbatch/reinforcement-training.sh \
	-n $architecture-untrained-$rewards_name \
	-a $architecture \
	-r $rewards


# use model after 10 epochs
sbatch ${dep_pz10:+--dependency=afterok:$dep_pz10} \
	sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-10-$rewards_name \
	-a $architecture \
	-r $rewards


# use model after 20 epochs
sbatch ${dep_pz20:+--dependency=afterok:$dep_pz20} \
	sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-20-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-20-$rewards_name \
	-a $architecture \
	-r $rewards
