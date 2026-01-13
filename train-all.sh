#!/bin/bash

architecture='linear'
rewards='win'

# pretrain moels with puzzles for either 10 or 20 epochs
imitation10_jib=$(sbatch --parsable sbatch/imitation-training.sh \
	-n $architecture-10-epochs \
	-e 10 \
	-a $architecture)
imitation20_jib=$(sbatch --dependency=afterok:$imitation10_jib --parsable sbatch/imitation-training.sh \
	-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
	-n $architecture-20-epochs \
	-e 10 \
	-a $architecture)


# use model after 10 epochs
sbatch --dependency=afterok:$imitation10_jib sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-10-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-$rewards \
	-a $architecture \
	-r win

# use model after 20 epochs
sbatch --dependency=afterok:$imitation20_jib sbatch/reinforcement-training.sh \
	-m logs/im/$architecture-20-epochs/models/checkpoint-best.pth \
	-n $architecture-pretrained-$rewards \
	-a $architecture \
	-r win


# reinforcement learning with newly initialized model
sbatch sbatch/reinforcement-training.sh \
	-n $architecture-untrained-$rewards \
	-a $architecture \
	-r $rewards
