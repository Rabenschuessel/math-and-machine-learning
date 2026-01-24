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

In this Report we investigate imitating learning as a form of pretraining. 
Specifically, we explore the effectiveness of pretraining with imitation 
learning with heavily biased data. 
We explore this for the game of chess,
because of the availability of both heavily biased data (puzzle datasets), 
and less biased data (grandmaster games). 

To isolate the effect of the bias in pretraining data, 
we train models with different architectures and reward functions. 


= Background
== Imitation Learning
== Reinforcement Learning


= Properties of Chess

For an agent to be successful in Chess it needs to master strategies and tactics. 
*Tactics* defines moves that achieve short term gains. 
This may be capturing opponent pieces or checkmating the opponent king. 
*Strategy* on the other hand defines moves with a long term benefit. 
This could be positioning a piece to put pressure on the opponents king, 
which may lead to a checkmate in 20-30 moves.
See @fig-tactics-vs-strategy for example positions. 
// source: https://www.chess.com/blog/APOSTOLISVAS/strategy-vs-tactics-the-difference-between-them

#include "figures/tactics_vs_strategy.typ"


== Puzzle Data

Players that want to improve their tactical perception can train on puzzles. 
A chess puzzle contains a position with a tactical motif (e.g. checkmate,...). 
A player then needs to play 4-5 moves to prove their understanding of the motif. 

There are many published datasets containing chess puzzles, such as //TODO 
These datasets are highly biased in that they solely contain tactical positions, 
underrepresenting strategic motifs.
Furthermore, positions in such datasets always contain a tactical motiv. 
The lack of negatives may may lead the model to be tactically overconfident, 
leading to false positives (see @fig-puzzle-bias).

#include "figures/puzzle_bias.typ"

== Grandmaster Data

Grandmaster games on the other hand are less biased than the puzzle dataset. 
They cover entire games, and therefore contain positions
with tactical motives, and strategic motives

#include "figures/gm_bias.typ"


= Methods

== Architecture

We train agents with three different architectures: a Feedforward neural network, 
a Convolutional neural network, and a Resnet. 
While the networks differ in their hidden layers, they all share a common input and output dimension: 

The network *input* contains the curent position.
The input contains a piece encoding ($12$ pieces: pawn, rook, knight, bishop, queen, king for both black and white)
for each square on the chess board ($8 times 8$), resulting in an input tensor of shape $8 times 8 times 12$. 

The *output* is used to generate a move distribution. 
It returns two vectors $m_"from" in RR^(64 times 1)$, 
and $m_"to" in RR^(1 times 64)$. 
A move matrix is then created from $M = m_("from")m_("to") in RR^(64 times 64)$. 
Illegal moves are masked, and move distribution is created.


== Imitation Pretraining



== Reward Functions



= Findings

