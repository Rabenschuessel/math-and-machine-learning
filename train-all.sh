#!/bin/bash
# Use DEBUG=1 for short run 

architectures=('linear' 'cnn' 'resnet')
rewards=('r_0' 'r_1' 'r_2')
reward_name=$(IFS=_; printf '%s' "${rewards[*]}")

# install environment when not existing
if ! conda env list | grep -q chess_ml; then 
	echo "Installing Conda Environment"
	set -x
	dep_env=$(sbatch sbatch/install-env.sh)
	set +x
fi


# install environment if needed
if [ ! -f data/gm_games_labeled.csv -a ! -f data/lichess_puzzle_labeled.csv ]; then
	echo "Downloading Datasets"
	set -x
	dep_env=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} sbatch/data-preparation.sh)
	set +x
fi


for arc in "${architectures[@]}"; do 
	
	# only pretrain if not already done
	dep_pz_1=$dep_env
	dep_pz_10=$dep_env
	if [ ! -f logs/im/$arc-pz-10-epochs/models/checkpoint-best.pth ]; then
		echo "Train Puzzle Imitation Learning"
		set -x

		# pretrain with 1 epochs
		dep_pz_1=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} --parsable \
		  sbatch/imitation-training.sh \
			-d ./data/lichess_puzzle_labeled.csv \
			-n $arc-pz-1-epochs \
			-e ${DEBUG:-1} \
			-a $arc)

		# pretrain with 10 epochs
		dep_pz_10=$(sbatch --dependency=afterok:$dep_pz_1 --parsable sbatch/imitation-training.sh \
			-m logs/im/$arc-pz-1-epochs/models/checkpoint-best.pth \
			-d ./data/lichess_puzzle_labeled.csv \
			-n $arc-pz-10-epochs \
			-e ${DEBUG:-1} \
			-a $arc)

		set +x
	fi

	# only pretrain if not already done
	dep_gm_1=$dep_env
	dep_gm_10=$dep_env
	if [ ! -f logs/im/$arc-gm-10-epochs/models/checkpoint-best.pth ]; then
		echo "Train GM Imitation Learning"
		set -x 

		# pretrain with 1 epochs
		dep_gm_1=$(sbatch ${dep_env:+--dependency=afterok:$dep_env} --parsable \
		  sbatch/imitation-training.sh \
			-d ./data/gm_games_labeled.csv \
			-n $arc-gm-1-epochs \
			-e ${DEBUG:-1} \
			-a $arc)

		# pretrain with 10 epochs
		dep_gm_10=$(sbatch --dependency=afterok:$dep_gm_1 --parsable sbatch/imitation-training.sh \
			-m logs/im/$arc-gm-1-epochs/models/checkpoint-best.pth \
			-d ./data/gm_games_labeled.csv \
			-n $arc-gm-10-epochs \
			-e ${DEBUG:-1} \
			-a $arc)

		set +x
	fi

	# Train for different sets of rewards
	for reward in "${rewards[@]}"; do
		echo "Queing Rewards: $reward"
		set -x
		
		# reinforcement learning with newly initialized model
		sbatch ${dep_env:+--dependency=afterok:$dep_env} \
			sbatch/reinforcement-training.sh \
			-n $arc-untrained-$reward \
			-a $arc \
			-r $reward ${DEBUG:+-b 1}

		# PUZZLE GAMES 
		# use model after 1 epochs
		sbatch ${dep_pz_1:+--dependency=afterok:$dep_pz_1} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$arc-pz-1-epochs/models/checkpoint-best.pth \
			-n $arc-pretrained-pz-1-$reward \
			-a $arc \
			-r $reward ${DEBUG:+-b 1}

		# use model after 10 epochs
		sbatch ${dep_pz_10:+--dependency=afterok:$dep_pz_10} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$arc-pz-10-epochs/models/checkpoint-best.pth \
			-n $arc-pretrained-pz-10-$reward \
			-a $arc \
			-r $reward ${DEBUG:+-b 1}

		# GM GAMES 
		# use model after 1 epochs
		sbatch ${dep_gm_1:+--dependency=afterok:$dep_gm_1} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$arc-gm-1-epochs/models/checkpoint-best.pth \
			-n $arc-pretrained-gm-1-$reward \
			-a $arc \
			-r $reward ${DEBUG:+-b 1}

		# use model after 10 epochs
		sbatch ${dep_gm_10:+--dependency=afterok:$dep_gm_10} \
			sbatch/reinforcement-training.sh \
			-m logs/im/$arc-gm-10-epochs/models/checkpoint-best.pth \
			-n $arc-pretrained-gm-10-$reward \
			-a $arc \
			-r $reward ${DEBUG:+-b 1}

		set +x
	done
done
