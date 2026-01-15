#!/bin/bash

architecture='linear'
rewards=('win' 'material')
reward_name=$(IFS=_; printf '%s' "${rewards[*]}")

# only pretrain if not already done
if [ ! -f logs/im/$architecture-20-epochs/models/checkpoint-best.pth ]; then
	# pretrain models with puzzles for either 10 or 20 epochs
	imitation10_jib=$(sbatch --parsable sbatch/imitation-training.sh \
		-n $architecture-10-epochs \
		-e 10 \
		-a $architecture)
	imitation20_jib=$(sbatch --dependency=afterok:$imitation10_jib --parsable sbatch/imitation-training.sh \
		-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
		-n $architecture-20-epochs \
		-e 10 \
		-a $architecture)
fi


# use model after 10 epochs
sbatch ${imitation10_jib:+--dependency=afterok:$imitation10_jib} \
	sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-10-$rewards_name \
	-a $architecture \
	-r $rewards

# use model after 20 epochs
sbatch ${imitation20_jib:+--dependency=afterok:$imitation20_jib} \
	sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-20-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-20-$rewards_name \
	-a $architecture \
	-r $rewards


# reinforcement learning with newly initialized model
sbatch sbatch/reinforcement-training.sh \
	-n $architecture-untrained-$rewards_name \
	-a $architecture \
	-r $rewards
