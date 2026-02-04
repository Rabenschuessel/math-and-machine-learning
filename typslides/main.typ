#import "@preview/metropolyst:0.1.0": *
#import "@preview/neural-netz:0.3.0": draw-network

#show: metropolyst-theme.with(
  accent-color: rgb("#23373b"),
  aspect-ratio: "4-3",
  config-info(
    title: [Exploring Interaction Between Behavior Cloning and Reward Shaping in Chess],
    author: [Jakob Lambert-Hartmann],
    date: datetime.today(),
  ),
)

#title-slide()


== Introduction 

#include "figures/architecture/conv_net_big.typ"



== Reinforcement Learning


#slide(composer: (3fr, 3fr))[
=== Reinforcement Learning 

#include "figures/rl.typ"

=== Problems

- Sparse rewards #sym.arrow poor performance
- Reward Shaping #sym.arrow difficult
  @comprehensive-reward-shaping: 
    - Missaligned rewards
    - Reward hacking 
    - Prioritizing wrong rewards

][
=== Imitation Learning: Behavior Cloning

#include "figures/imitation.typ"

- Supervised learning
- Agent learns expert demonstrations 
- e.g. puzzles, grandmaster games

=== Problems

- Demonstration sample bias #sym.arrow poor performance
  @stanford-imitation-learning

]







== Combination

=== Combine both approaches
1. Pretrain: Behavior Cloning
2. Finetune: Reward Based Reinforcement Learning

=== Can both approaches stabelize each other?

- Behavior Cloning: Stabelize RL with sparse rewards
- Reinforcement Learning: Compensates for data bias 

=== Lets find out!

- Train different models: 
    - Architecture: Fully connected, cnn, residual block
    - Pretraining Data: Heavily biased #sym.arrow less biased
    - Rewards Sets: Sparse #sym.arrow Dense



= Architecture

== Architecture

#slide(composer: (3fr, 2fr))[
=== FCNN

- 3 hidden layers: 512 neurons
=== CNN

- increasing number of channels:  
- $12 arrow 32 arrow 64 arrow 128 arrow 256 arrow 32$
=== ResNet

- 20 blocks with 64 channels
][
  #include "figures/architecture/fc_net.typ"
  #include "figures/architecture/conv_net.typ"
  #include "figures/architecture/res_net.typ"
]


== Architecture
#slide(composer: (3fr, 1.5fr))[
=== Input
- tensor `shape = (8,8,12)`: 
    - $8 times 8$ squares
    - 12 pieces: ${"White", "Black"} times$
        ${"Pawn", "Rook", "Knight", "Bishop", "Queen", "King"}$
=== Output
- Move distribution: from square #sym.arrow to square
- *But:* Not all moves are legal #sym.arrow mask illegal moves
][
#include "figures/chess/legal_moves.typ"
]









= Imitation Learning

== Background: Chess

=== Strategy vs Tactics

#include "figures/chess/tactics_vs_strategy.typ"


== Imitation Learning: Behavior Cloning

#slide(composer: (2fr, 2fr))[
  #align(bottom)[
=== Lichess Puzzles @lichess-puzzle-dataset
- 6 million positions 
- tactical positions 

=== Bias 
- no strategic positions
- no positions without tactic 
#include "figures/chess/puzzle_bias.typ"
]][
  #align(bottom)[
=== GM games @gm-games-dataset
- chess.com master games 
- balance tactics/strategy

=== Bias & Problems
- high level play
- many resignations
- multiple moves possible

#include "figures/chess/gm_bias.typ"
]]



= Reinforcement Learning

== Reinforcement Learning: Reward Shaping

=== Reward sets
0. *Sparse:* rewards only for true goal
  - reward for win/loss/draw
1. *Medium:* more rewards, but might misalign with goals
  - control center, material win, king safety
2. *Dense:* highest risk of misalignement
  - control outer center, control center, castling, promoting, blunder prevention, material, king safety, give check, win





== Total Models

=== Training 
- *Architecture:* ${"FCNN", "CNN", "ResNet"}$
- *Pretraining:*  ${"Puzzles", "Masters"} times {"1 Epoch", "10 Epochs"}, {"Untrained"}$
- *Reward Set:*   ${"R_0", "R_1", "R_2"}$
  - 1000 batches with 16 games each (REINFORCE) 
#sym.arrow 57 models 








= Results



// == Overal Results
//
// #include "figures/analysis/tt/results.typ"
//
// - No optimal training strategy visible 
// - Sparse / No rewards seem to perform better
//
// === Let's take a closer look!

== Behavior Cloning: Training Accuracy 

#include "figures/analysis/bc/training-accuracy.typ"

- Puzzle better generalizing
- *But* master games: different labels, same position

== Behavior Cloning

=== Evaluation 

- All models plays 1000 games against all other model (same architecture)
- Normalized tournament score (win 1 point, draw 0.5 points, loss 0 points)


// #slide(composer: (2fr, 2fr))[
  === BC model with RL 
  #include "figures/analysis/bc/mean-rl.typ"
// ][
  - Models improve significantly with pretraining
// ]



== Reward Shaping

#align(center)[
#stack(
  dir: ltr, 
  spacing: 2cm,
[

  === RL models with Pretraining
  #include "figures/analysis/rl/mean-pretrain.typ"
],[

  === RL models without Pretraining
  #include "figures/analysis/rl/mean-no-train.typ"
])]
- Depending on the model RL hurts performance
- Sparse rewards generally better


== Conclusion

- We could not find any training strategy across architectures
- Overall denser rewards hurt model performance

=== Possible Improvements
- *Data Prep:* Unify data in Masters games
- *Architecture:* Hyperparameter tuning on pretrain accuracy
- *BC:* Mix of master games and Puzzles 
- *RL:* Longer training
- *RL:* Iterative reward shaping


== How does it really play?

#align(center)[
Well....
#link("https://lichess.org/cuRJkeJ1#0")
]



== References
#bibliography("bib.yaml", title: none)
