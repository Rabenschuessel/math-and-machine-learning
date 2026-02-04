---
title:
- Exploration of Reward Shaping and Imitation Learning in Chess
author:
- Jakob Lambert-Hartmann
date:
- 2025
graphics: true
theme: metropolis
suppress-figure-numbering: true
header-includes: 
    - \usepackage{tikz}
    - \usetikzlibrary{shapes.geometric, arrows, positioning, calc}
---


## Introduction 

\begin{tikzpicture}[
    node distance=0.5cm,
    box/.style={draw, rounded corners, minimum width=2.2cm, minimum height=1.2cm, align=center}
]

% Nodes
\node[box] (input) {\includegraphics[height=0.4\textheight]{./images/position.png}};
\node[box, right=of input] (model) {Model};
\node[box, right=of model] (output) {\includegraphics[height=0.4\textheight]{./images/move.png}};

% Arrows
\draw[->, thick] (input) -- (model);
\draw[->, thick] (model) -- (output);

\end{tikzpicture}

## Reinforcement Learning


\begin{center}
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1.2cm, align=center},
    >=latex
]
\node[box] (agent) {Agent};
\node[box, below=2cm of agent] (env) {Environment};
\draw[->, thick] (agent) -- ++(2.5,0) |- node[pos=0.25, right]{action} (env);
\draw[->, thick] (env) -- ++(-2,0) |- node[pos=0.25, right]{observation} (agent);
\draw[->, thick] (env) -- ++(-2.5, 0) |- node[pos=0.25, left]{reward} (agent);
\end{tikzpicture}
\end{center}

## REINFORCE 

- **Goal:** Directly optimize a stochastic policy $\pi_\theta(a \mid s)$ by maximizing the expected return  
  $$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} R_t \right] $$

- **Core idea:** Increase the probability of actions that lead to high returns  
  $$ \nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right] $$

- **Algorithm:**
  - Sample trajectories using the current policy $\pi_\theta$
  - Compute the return $G_t$ for each time step
  - Update parameters via gradient ascent

## Problems with rewards

- Sparse rewards $\rightarrow$ poor performance
- Reward Shaping $\rightarrow$ difficult: 
    - Missaligned rewards
    - Reward hacking 
    - Prioritizing wrong rewards



## Imitation Learning: Behavior Cloning

- Supervised learning
- Agent learns expert demonstrations 
- e.g. puzzles, grandmaster games

**Problems:**

- Demonstration sample bias $\rightarrow$ poor performance

## Combination

1. Pretrain: Behavior Cloning
2. Finetune: Reward Based Reinforcement Learning

**Can both approaches stabelize each other?**

- Behavior Cloning: Stabelize RL with sparse rewards
- Reinforcement Learning: Compensates for data bias 


## Lets Find Out

- Train different models: 
    - Architecture: Fully connected, cnn, residual block
    - Pretraining Data: Heavily biased $\rightarrow$ less biased
    - Rewards Sets: Sparse $\rightarrow$ Dense



# Methods

## Architecture

**Input:**

- tensor `shape = (8,8,12)`: 
    - $8 \cdot 8$ squares
    - 12 pieces: $\{Pawn, Rook, Knight, Bishop, Queen, King\} \times \{White, Black\}$


**Output:**

- Move distribution: from square $\rightarrow$ to square
- **But:** Not all moves are legal $\rightarrow$ mask illegal moves

## Masking

\includegraphics{./images/chess/legal_moves.pdf}

- mask illegal moves
- masked logits $\rightarrow$ move distribution
- sample from move distribution 

## Architecture

- Fully Connected NN
    - 3 hidden layers: 512 neurons
- Convolutional NN
    - increasing number of channels:  
    - $12 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 32$
- Residual Block NN
    - 20 blocks with 64 channels



<!-- \includegraphics{./figures/architecture/fc_net.pdf} -->
<!-- \includegraphics{./figures/architecture/res_net.pdf} -->
<!-- \includegraphics{./figures/architecture/conv_net.pdf} -->


## Behavior Cloning

A good model needs to master strategy and tactics\
(important to understand data bias): 

::: columns

:::: {.column width=45%}
**Strategy:**
\begin{center}
\includegraphics{./images/chess/strategy.pdf}
\end{center}
::::

:::: {.column width=45%}
**Tactics:**
\begin{center}
\includegraphics{./images/chess/tactics.pdf}
\end{center}
::::
:::


## Dataset: Chess Puzzles 

::: {.columns align=center }

:::: {.column width=50%}
- lichess puzzle dataset
- 6 million positions 
- **bias:** 
    - no strategic positions
    - no positions without tactic 

::::
:::: {.column width=45%}
\begin{center}
\includegraphics{./images/chess/puzzle_bias.pdf}
\end{center}
::::
:::


## Dataset: GM Games

::: {.columns align=center }

:::: {.column width=50%}
\centering
- chess.com master games 
- balance tactics/strategy
- **bias:** 
    - high level play
    - many resignations
- **other problem:**
    - multiple moves possible

::::

:::: {.column width=45%}
\begin{center}
\includegraphics{./images/chess/gm_bias.pdf}
\end{center}
::::
:::

## Behavior Cloning

- Train model with one of both datasets
    - Save best after 1 epoch
    - Save best after 10 epoch


## Reinforcement Learning: Reward Shaping

- Reward sets: 
    1. Sparse: only reward on win/loss
    2. Medium: adding rewards material advantage, center control, king safety 
    2. Dense: adding rewards outer center control, castling, pawn promotion, blunder prevention, checks 

**Selfplay:** 1000 batches with 16 games each (REINFORCE) \
$\rightarrow$ total of 16000 games per color per model




## Total Models



::: {.columns align=center }

:::: {.column width=50%}
- Architecture: 
    - fully connected
    - cnn
    - residual block
- Pretraining: 
    - untrained
    - puzzle data: 1 epoch
    - puzzle data: 10 epoch
    - gm data: 1 epoch
    - gm data: 10 epoch
- Reward Sets: 
    - none
    - sparse
    - medium
    - dense
::::

:::: {.column width=5%}
$\rightarrow$  
::::

:::: {.column width=40%}
57 models  
::::
:::

# Results

## Behavior Cloning


|dataset      | epochs| fc   | cnn |resblock |
|-------------|-------|------|-----|---------|
|puzzle       |1      | 62.0%|69.9%|74.6%    |
|             |10     | 65.8%|73.3%|81.7%    |
|grand masters|1      | 31.0%|38.5%|41.5%    |
|             |10     | 31.8%|39.6%|43.7%    |

- Puzzle better generalizing
- **But** master games: different labels, same position


## Evaluate 

- For each architecture: play tournament with 1000 games per matchup

## Reward Shaping

## Best Model

| architecture  | train  | rewards   | points% |
|---------------|--------|-----------|---------|
| cnn           | gm_10  | r_0       | 67.4    |
| fc            | pz_1   | r_0       | 17.6    |
| resnet        | pz_10  | none      | 65.0    |


[well... it's better than random (probably)](https://lichess.org/cuRJkeJ1#82)

