#import "@preview/board-n-pieces:0.9.0": *
#let board = board.with(
  white-square-fill: rgb("#d2eeea"),
  black-square-fill: rgb("#567f96"),
  arrow-fill: green,
  square-size: 0.7cm,
)

#stack(
  dir: ltr,
  spacing: 0.5cm,
  board(
    fen("4k3/8/8/4q3/8/8/3PP3/4K3 b - - 0 1"),
    stroke: 0.8pt + black,
  ),
  board(
    fen("4k3/8/8/1b2q3/8/8/3PP3/4K3 b - - 0 1"),
    arrows: ("e5 e2",),
    stroke: 0.8pt + black,
  ),

)
