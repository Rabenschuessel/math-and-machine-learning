#import "@preview/board-n-pieces:0.9.0": *
#let board = board.with(
  white-square-fill: rgb("#d2eeea"),
  black-square-fill: rgb("#567f96"),
  arrow-fill: green,
)

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
    board(
      fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
      arrows: (
        "c2 c4",
        "d2 d4",
        "e2 e4",
        "g2 g3",
        "g1 f3",
      ),
      display-numbers: true, 
      square-size: 0.8cm, 
      stroke: 0.8pt + black,
    ),
  ),
  caption: [
    The position on the left shows the last position of a game
    by the two former world champions Garry Kasparov and Veselin Topalov.
    Black resigns in this completely lost position without waiting for a checkmate. 
    The position on the right shows different first moves played by masters, 
    showing that there is more than one viable move per position. 
    ]
) <fig-gm-bias> 

// source: https://www.chess.com/news/view/25-years-ago-kasparov-topalov-1999
