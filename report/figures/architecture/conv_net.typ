#import "@preview/board-n-pieces:0.9.0": *
#import "@preview/neural-netz:0.3.0": draw-network
#let board = board.with(
  white-square-fill: rgb("#d2eeea"),
  black-square-fill: rgb("#567f96"),
  arrow-fill: green,
)

#let layers = (12, 32, 64, 128, 256, 32)
#figure(draw-network(
  ((
    type: "input", 
    image: board(fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")),
    height: 6,
    depth: 6,
    label: "position",
  ),) + 
  for channel in layers {
  ((
    type: "conv", 
    widths:   (channel / 256, ),
    channels: (channel, 64),
    height: 6,
    depth: 6,
    label: "conv",
  ),) }+ 
  ((
    type: "fc", 
    channels: (32 * 64, ),
    height: 6,
    depth: 0.25,
    label: "fc",
  ),
  (
    type: "input", 
    image: board(
      fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
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
    ),
    height: 6,
    depth: 8,
    label: "move distribution",
  ), ), 
  scale: 65%,
  palette: "cold"
), 
  caption: [
    Convolutional neural network 
  ] 
) <fig-conv-net>

