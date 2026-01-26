#import "@preview/board-n-pieces:0.9.0": *

#figure(
  stack(
    dir: ltr,
    spacing: 0.5cm,
    board(
      fen("r2q1rk1/p3bppp/2n1p3/1p1pN3/2pP4/1PP3P1/P3PPBP/R1BQR1K1 b - - 2 15"),  
      display-numbers: true, 
      square-size: 0.8cm, 
      arrows: ("e5 c6",),
      stroke: 0.8pt + black,
    ),
    board(
      fen("rnbq1rk1/3nbppp/p2p4/4p3/1p2P3/2NP1N1P/BPP2PP1/R1BQR1K1 w - - 1 12"),
      display-numbers: true, 
      square-size: 0.8cm, 
      arrows: ("c3 d5",),
      stroke: 0.8pt + black,
    ),

  ),
  caption: [
    The position on the left has a tactical motif,
    where the black knight can be captured by white, as it is not defended. 
    On the other hand the right position has a strategic motiv, 
    where the knight is attacked and has to retreat. 
    There are multiple available squares, but d5 is the best option. 
    There is no immediate gain, however in the long run the knight will 
    be able to exert more pressure on the opponent from this square. 
    ]
) <fig-tactics-vs-strategy> 
