#import "@preview/board-n-pieces:0.9.0": *

#figure(
  stack(
    dir: ltr,
    spacing: 0.5cm,
    board(
      fen("4k3/8/8/4q3/8/8/3PP3/4K3 b - - 0 1"),
      display-numbers: true, 
      square-size: 0.8cm, 
      stroke: 0.8pt + black,
    ),
    board(
      fen("4k3/8/8/1b2q3/8/8/3PP3/4K3 b - - 0 1"),
      display-numbers: true, 
      square-size: 0.8cm, 
      arrows: ("e5 e2",),
      stroke: 0.8pt + black,
    ),

  ),
  caption: [
    both positions are similar except for the black bishop on d5.
    In the position with the Bishop (right) black can win the game in one 
    move by checkmating with the queen on e2. 
    playing queen e2 in the other position will lose the game. 
    ]
) <fig-puzzle-bias> 
