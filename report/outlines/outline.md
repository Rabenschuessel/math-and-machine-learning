# Exploration of Reward Shaping and Imitation Learning in Chess

## Goal 

We want to investigate whether imitation learning can reduce the need for rewards shaping, even when the data is heavily biased. 
Specifically when data for imitation learning is heavily biased and only covers a subset of the desired agent behavior. 


We chose chess as environment, 
For a chess agent to be successful, it needs to perform well in strategy and tactics: 

- Strategy: there are various viable moves 
- Tactics: accomplishing immediate gain. Tactics are often defined as a short sequence of moves with only one viable move in each position 

We chose chess as environment, as widely available data in form of chess puzzles is heavily biased towards specific behavior.

# Data 

- puzzles 
    puzzles are biased towards tactics, and even there only contain positives. 

    Also only contain positions where tactical motive exists: 
    4k3/8/8/1b2q3/8/8/3PP3/4K3 b - - 0 1: qe2 checkmate
    4k3/8/8/4q3/8/8/3PP3/4K3 b - - 0 1: qe2 loses game
    Only checkmate represented in puzzle dataset -> model would probably learn qe2


- grand master games
    Grand master games on the other hand contain a mix of tactics and strategic positions. 
    While this reduces the biases, grand master games are not completely without bias. 
    Checkmates are underrepresented in grandmaster games, as they often resign moves ahead, where it is obvious to them that the position is lost. 
    
    


## Prior Works 

Existing methods used for chess. 
We want to just do reinforcement learning for simplicity.


## Methods

1. Reinforcement learning
2. Imitation learning with lichess puzzles
3. Imitation learning with gm games

Compare the performance of imitation learning from biased puzzles 
against pure RL and against less biased gm games


### Reward Functions

1. simple rewards: win/draw/loss
2. chess principles 


### Pretraining 


## Model Architecture 
### Inputs
### Outputs


# Findings 



