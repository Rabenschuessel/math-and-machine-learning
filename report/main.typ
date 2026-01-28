#import "@preview/basic-report:0.3.0": *
#import "@preview/subpar:0.2.2"
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/board-n-pieces:0.9.0": *

#show: codly-init.with()
#codly(zebra-fill: none)

#show: it => basic-report(
  doc-title: "Exploration of Rewards Shaping and Immitation Learning in Chess",
  author: "Jakob Lambert-Hartmann",
  language: "en",
  compact-mode: false,
  show-outline: false,
  heading-font: "Fira Math",
  it
)

#set text(size: 11pt, font: "Fira Sans")
#set page(margin: (x: 2.5cm, top:4cm, bottom:3cm))
#set page(footer:  grid.cell(colspan: 2, line(length: 100%, stroke: 0.5pt)))
#set par(justify: true)
#set heading(numbering: none)
#show heading: set text(weight: "bold")
#show heading.where(level: 4): it => box(it.body) 

#outline(depth: 3)
#pagebreak()

= Introduction 


// basic rl terms
Reinforcment learning (RL) is a field of machine 
learning concerned with desicion-making. 
It is used in scenarios where an inteligent agent 
takes actions in an environment in order to maximize a reward 
(e.g. a chess agent learning to maximize the change of wining). 
The actions chosen by the agent (e.g. chess move) are determined by a policy $pi$. 
Whenever the agent takes an action, it receives a reward from the environment. 
Winning a game for instance will yield a positive reward, losing a negative reward, 
and otherwise the reward will be zero.
Over the course of training the agent adapts its policy in order to maximize rewards.

// sparse rewards and reward shaping
If rewards are sparse, meaning most rewards are zero,
it is difficult for an agent to learn a good policy, 
as there will be too little feedback for the model to learn. 
Wining a game of chess for instance is a sparse reward. 
It is difficult for an agent to learn chess with win as only reward, 
as it is very unlikely for an agent to randomly win a game. 
To help agents learn policies faster reward shaping 
introduces further rewards that align with the original goal 
and help to nudge the model in the right direction. 
Reward shaping however introduces a new set of problems. 
If the newly introduced rewards are not aligning with the ultimate goal for instance, 
the model may choose to prioritize those and ignore the real goal (reward hacking). 
@reward-shaping

// Imitation learning
Imitation learning is a paradigm in reinforcement learning, 
in which an agent learns a policy through a set of expert demonstrations, 
rather than an explicit reward function. 
Behavior cloning is a specific imitation learning technique, 
which utilizes supervised learning to learn this policy. 
However, expert demonstrations used for behavior cloning are often
not uniformly sampled from the state space leading to poor performance. 
@stanford-imitation-learning

// combination of both and goal of project
One Idea for mitigating the drawbacks of both reward based reinforcement learning
and behavior cloning would be to combine the two approaches.
Behavior cloning as a pretraining could reduce the need for reward shaping, 
while finetuning with reward based reinforcement learning 
could iron out the sampling bias from the expert demonstrations. 
In this document we investigate the interaction between pretraining 
with behavior cloning and finetuning with reward shaping
using different sets of expert demonstrations and reward functions. 
We will apply this to the domain of chess,
as available data is well split into different sampling biases.




= Methods

We trained models with different configurations, 
namely the neural network architecture, 
the datasets used for behavior cloning, 
and the reward functions used for reinforcement learning. 
In this section we discuss details of each aspect
before evaluating our results in the next chapter.


== Neural Network Architecture

We trained models with three different architectures: 
a fully connected neural network (see @fig-fc-net), 
a convolutional neural network (see @fig-conv-net), 
and a residual block network (see @fig-res-net).

While the hidden layers of the agent differ,
they all share a common input and output representation.
The network *input* contains the curent position,
is represented by a piece encoding 
(12 pieces: pawn, rook, knight, bishop, queen, king for both black and white)
for each square on the chess board ($8 times 8$). 
resulting in an input tensor $x in RR^(8 times 8 times 12)$. 

The *output* consists of two tensors representing from which square to move
$m_"from"$ to which square $m_"to"$. 
A joint probability is created by computing from-to logits $M = m_"from"m_"to"^T$, 
masking out illegal moves (see @fig-legal-moves), and computing the softmax
$pi(b) = sigma("mask"(m_"from"m_"to"^T))$.

#include "figures/chess/legal_moves.typ"





== Behavior Cloning

For an agent to be successful in Chess it needs to master both strategies and tactics. 
*Tactics* defines moves that achieve short term gains. 
This may be capturing opponent pieces or checkmating the opponent king. 
*Strategy* on the other hand defines moves with a long term benefit. 
This could be positioning a piece to put pressure on the opponents king, 
which may lead to a checkmate in 20-30 moves.
See @fig-tactics-vs-strategy for example positions. 

#include "figures/chess/tactics_vs_strategy.typ"

For behavior cloning we chose two datasets, where one is biased towards strategic,
and the other towards tactical positions,
which we will discuss in in more detail in the following sections. 
For both datasets we saved the model after one epoch and the best model out of 
10 epochs.


=== Puzzle Data

Players that want to improve their tactical perception can train on puzzles. 
A chess puzzle contains a position with a tactical motif (e.g. checkmate,...). 
A player then needs to play 4-5 moves to prove their understanding of the motif. 

There are many published datasets containing chess puzzles, 
such as the Lichess puzzle dataset @lichess-puzzle-dataset. 
Positions in these datasets are highly biased in that they solely contain tactical positions, 
underrepresenting strategic motifs.
Furthermore, positions in such datasets always contain a tactical motiv. 
The lack of negatives may may lead the model to be tactically overconfident, 
leading to false positives (see @fig-puzzle-bias).

#include "figures/chess/puzzle_bias.typ"

=== Grandmaster Data

Grandmaster games are overall less biased than the puzzle dataset. 
They cover entire games, and therefore contain positions
with tactical and positions with strategic motifs.
However, they still are biased in that they only represent top level play. 
Furthermore, grandmasters are able to recognize a losing position earlier than
club level players. As such, they often resign multiple moves before 
an unavoidable checkmate (see @gm-games-dataset). 
This underrepresentation of checkmates, 
may make it difficult for our model to convert a wining position into a checkmate
(see @fig-gm-bias).

#include "figures/chess/gm_bias.typ"




== Rewards Shaping

Models trained with reinforcement learning through selfplay. 
The models played 16 games before updating based with REINFORCE.  // TODO cite
This was performed for 1000 iterations,
resultsing in 16000 games played both as white and as black. 
After each chosen action a set of reward function was evaluated. 

We used three sets of reward functions ranging form sparse to dense: 

// TODO more information 
/ Sparse: The sparse reward consisted of only one reward function, 
  only a nonzero reward at the end of the game.
  A positive reward was awarded for a win, whereas a los returned a negative reward. 
/ Medium: The reward set of medium sparsity returned 
  material, control center, win, king safety
/ Dense: The dense rewardset returned 
  control outer center, control center, castling, promoting, blunder prevention,
  material, king safety, give check, win


In chess a win yields one point, a loss zero, and a draw half a point. 
To compare the models we summed points for each matchup (two models playing 500 games).
We normalized the point values to the range $[0,1]$, 
which represents the portion of available points the model received. 






= Results

To evaluate the model performance,
we let the models play against each other. 
Each model played 500 games as white and 500 games as black
against each other model with the same architecture 
(i.e. cnn models only played cnns, etc.). 

The models trained with supervised training performed much better on the puzzle
test set.
However a good accuracy in the masters game is difficult, 
as a position may have multiple viable moves, and training datapoints. 

#include "figures/analysis/imitation-learning.typ"

#include "figures/analysis/analysis-plots.typ"



= Furter Work 

// remove duplicate positions with different labels in GM dataset



#pagebreak()
#bibliography("bib.yaml")



#pagebreak()
= Appendix

#include "figures/architecture/conv_net.typ"
#include "figures/architecture/res_net.typ"
#include "figures/architecture/fc_net.typ"
