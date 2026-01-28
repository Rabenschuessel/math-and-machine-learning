#import "@preview/board-n-pieces:0.9.0": *
#set text(white)
#set page(width: 7cm, height: 16cm, margin: 0pt)
#set align(center)

#board(
  fen("4k3/8/8/1b2q3/8/8/3PP3/4K3 b - - 0 1"),
  // display-numbers: true, 
  square-size: 0.8cm, 
  arrows: ("e5 e2",),
  stroke: 0.8pt + black,
)

#board(
  fen("4k3/8/8/4q3/8/8/3PP3/4K3 b - - 0 1"),
  // display-numbers: true, 
  square-size: 0.8cm, 
  stroke: 0.8pt + black,
)
