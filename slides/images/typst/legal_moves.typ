#import "@preview/board-n-pieces:0.9.0": *
// #set text(color.hsl(30deg, 50%, 60%))
#set text(white)

#board(fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    display-numbers: true, 
    arrows: (
      "a2 a3", "a2 a4",
      "b2 b3", "b2 b4",
      "c2 c3", "c2 c4",
      "d2 d3", "d2 d4",
      "e2 e3", "e2 e4",
      "f2 f3", "f2 f4",
      "g2 g3", "g2 g4",
      "h2 h3", "h2 h4",
      "b1 a3", "b1 c3",
      "g1 f3", "g1 h3",
    ),
    square-size: 1.5cm, 
    stroke: 0.8pt + black,
  )

#board(fen("4k3/8/8/8/8/8/3PP3/4K3 b - - 0 1"),
  display-numbers: true, 
  arrows: (
    "d2 d3", "d2 d4",
    "e2 e3", "e2 e4",
    "e1 d1", 
    "e1 f1", 
    "e1 f2", 
  ),
  square-size: 1.5cm, 
  stroke: 0.8pt + black,
)
