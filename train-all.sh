#!/bin/bash


# reinforcement model pretrained on puzzles
imitation_jib=$(sbatch --parsable sbatch/imitation-training.sh
	-n puzzles
	-a resnet
)
sbatch --dependency=afterok:$imitation_jib sbatch/reinforcement-training.sh \
	-m logs/im/experiment-puzzles/models/checkpoint-best.pth \
	-n resnet-pretrained-win
	-a resnet
	-r win

# reinforcement learning with newly initialized model
sbatch sbatch/reinforcement-training.sh \
	-n resnet-untrained-win \ 
	-a resnet
	-r win
