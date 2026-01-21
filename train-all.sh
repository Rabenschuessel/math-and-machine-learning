#!/bin/bash

architecture='linear'
rewards=('r_0' 'r_1' 'r_2')
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


for arc in "$architectures[@]"; do 
	
	# only pretrain if not already done
	dep_pz_10=$dep_env
	dep_pz_20=$dep_env
	if [ ! -f logs/im/$architecture-pz-20-epochs/models/checkpoint-best.pth ]; then
		echo "Train Puzzle Imitation Learning"

		# pretrain with 10 epochs
		dep_pz_10=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} --parsable \
		  sbatch/imitation-training.sh \
			-d ./data/lichess_puzzle_labeled.csv \
			-n $architecture-pz-10-epochs \
			-e 10 \
			-a $architecture)

		# pretrain with 20 epochs
		dep_pz_20=$(sbatch --dependency=afterok:$dep_pz_10 --parsable sbatch/imitation-training.sh \
			-m logs/im/$architecture-pz-10-epochs/models/checkpoint-best.pth \
			-d ./data/lichess_puzzle_labeled.csv \
			-n $architecture-pz-20-epochs \
			-e 10 \
			-a $architecture)
	fi

	# only pretrain if not already done
	dep_gm_10=$dep_env
	dep_gm_20=$dep_env
	if [ ! -f logs/im/$architecture-gm-20-epochs/models/checkpoint-best.pth ]; then
		echo "Train GM Imitation Learning"

		# pretrain with 10 epochs
		dep_gm_10=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} --parsable \
		  sbatch/imitation-training.sh \
			-d ./data/gm_games_labeled.csv \
			-n $architecture-gm-10-epochs \
			-e 10 \
			-a $architecture)

		# pretrain with 20 epochs
		dep_gm_20=$(sbatch --dependency=afterok:$dep_gm_10 --parsable sbatch/imitation-training.sh \
			-m logs/im/$architecture-gm-10-epochs/models/checkpoint-best.pth \
			-d ./data/gm_games_labeled.csv \
			-n $architecture-gm-20-epochs \
			-e 10 \
			-a $architecture)
	fi

	# Train for different sets of rewards
	for reward in "${rewards[@]}"; do
		echo "Queing Rewards: $reward"
		
		# reinforcement learning with newly initialized model
		sbatch ${dep_env:+--dependency=afterok:$dep_env} \
			sbatch/reinforcement-training.sh \
			-n $architecture-untrained-$rewards_name \
			-a $architecture \
			-r $rewards

		# PUZZLE GAMES 
		# use model after 10 epochs
		sbatch ${dep_pz_10:+--dependency=afterok:$dep_pz_10} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$architecture-pz-10-epochs/models/checkpoint-best.pth \
			-n $architecture-pretrained-pz-10-$rewards_name \
			-a $architecture \
			-r $rewards

		# use model after 20 epochs
		sbatch ${dep_pz_20:+--dependency=afterok:$dep_pz_20} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$architecture-pz-20-epochs/models/checkpoint-best.pth \
			-n $architecture-pretrained-pz-20-$rewards_name \
			-a $architecture \
			-r $rewards

		# GM GAMES 
		# use model after 10 epochs
		sbatch ${dep_gm_10:+--dependency=afterok:$dep_gm_10} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$architecture-gm-10-epochs/models/checkpoint-best.pth \
			-n $architecture-pretrained-gm-10-$rewards_name \
			-a $architecture \
			-r $rewards

		# use model after 20 epochs
		sbatch ${dep_gm_20:+--dependency=afterok:$dep_gm_20} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$architecture-gm-20-epochs/models/checkpoint-best.pth \
			-n $architecture-pretrained-gm-20-$rewards_name \
			-a $architecture \
			-r $rewards

	done
done
