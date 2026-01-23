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

=== Input/Output

== Imitation Pretraining

== Reward Functions



= Findings

