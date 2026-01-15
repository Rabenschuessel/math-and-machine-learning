#!/bin/bash

architecture='linear'
rewards=('win' 'material')
reward_name=$(IFS=_; printf '%s' "${rewards[*]}")

# install environment when not existing
if ! conda env list | grep -q chess_ml; then 
	echo "Installing Conda Environment"
	jid=$(sbatch sbatch/install-env.sh)
fi


# install environment if needed
if [ ! -f data/gm_games_labeled.csv -a ! -f data/lichess_puzzle_labeled.csv ]; then
	echo "Downloading Datasets"
	jid=$(sbatch ${jib:+--dependency=afterok:$jib} sbatch/data-preparation.sh)
fi


# only pretrain if not already done
if [ ! -f logs/im/$architecture-20-epochs/models/checkpoint-best.pth ]; then
	echo "Train Imitation Learning"

	# pretrain with 10 epochs
	imitation10_jib=$(sbatch ${jib:+--dependency=afterok:$jib} \
		--parsable sbatch/imitation-training.sh \
		-n $architecture-10-epochs \
		-e 10 \
		-a $architecture)

	# pretrain with 20 epochs
	imitation20_jib=$(sbatch --dependency=afterok:$imitation10_jib --parsable sbatch/imitation-training.sh \
		-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
		-n $architecture-20-epochs \
		-e 10 \
		-a $architecture)
fi


# use model after 10 epochs
sbatch ${imitation10_jib:+--dependency=afterok:$imitation10_jib} \
	${jib:+--dependency=afterok:$jib} \
	sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-10-$rewards_name \
	-a $architecture \
	-r $rewards


# use model after 20 epochs
sbatch ${imitation20_jib:+--dependency=afterok:$imitation20_jib} \
	${jib:+--dependency=afterok:$jib} \
	sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-20-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-20-$rewards_name \
	-a $architecture \
	-r $rewards


# reinforcement learning with newly initialized model
sbatch ${jib:+--dependency=afterok:$jib} \
	sbatch/reinforcement-training.sh \
	-n $architecture-untrained-$rewards_name \
	-a $architecture \
	-r $rewards
