#!/bin/bash


# reinforcement model pretrained on puzzles
imitation_jib=$(sbatch --parsable sbatch/imitation-training.sh)
sbatch --dependency=afterok:$imitation_jib sbatch/reinforcement-training.sh \
	-m logs/im/experiment-0/models/checkpoint-best.pth \
	-n resnet-pretrained-win
	-r win

# reinforcement learning with newly initialized model
sbatch sbatch/reinforcement-training.sh \
	-n resnet-untrained-win \ 
	-r win
