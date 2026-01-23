
#import "@preview/board-n-pieces:0.9.0": *

#figure(
  stack(
    dir: ltr,
    spacing: 0.5cm,
    board(
      fen("8/Q6p/6p1/5p2/5P2/2p3P1/3r3P/2K1k3 b - - 3 44"),
      display-numbers: true, 
      square-size: 0.8cm, 
      stroke: 0.8pt + black,
    ),
  ),
  caption: [
    This is the last position of a game by the two former world champions
    Garry Kasparov and Veselin Topalov.
    Black resigns in this completely lost position. 
    This is obvious to a grandmaster, in our case however, 
    it will lead to an underrepresentation of checkmates, 
    and will make it difficult for a model to convert completely wining positions.
    ]
) <fig-gm-bias> 

// source: https://www.chess.com/news/view/25-years-ago-kasparov-topalov-1999
