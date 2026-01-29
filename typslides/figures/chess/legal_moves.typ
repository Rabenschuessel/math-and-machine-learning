#import "@preview/board-n-pieces:0.9.0": *
#let board = board.with(
  white-square-fill: rgb("#d2eeea"),
  black-square-fill: rgb("#567f96"),
  arrow-fill: green,
  display-numbers: false, 
  square-size: 1cm, 
)

#board(
  fen("4k3/8/8/8/8/8/3PP3/4K3 b - - 0 1"),
  arrows: (
    "d2 d3", "d2 d4",
    "e2 e3", "e2 e4",
    "e1 d1", 
    "e1 f1", 
    "e1 f2", 
  ),
  stroke: 0.8pt + black,
)

// #figure(
//   stack(
//     dir: ttb,
//     spacing: 0.5cm,
//     // board(
//     //   fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
//     //   arrows: (
//     //     "a2 a3", "a2 a4",
//     //     "b2 b3", "b2 b4",
//     //     "c2 c3", "c2 c4",
//     //     "d2 d3", "d2 d4",
//     //     "e2 e3", "e2 e4",
//     //     "f2 f3", "f2 f4",
//     //     "g2 g3", "g2 g4",
//     //     "h2 h3", "h2 h4",
//     //     "b1 a3", "b1 c3",
//     //     "g1 f3", "g1 h3",
//     //   ),
//     //   stroke: 0.8pt + black,
//     // ),
//     board(
//       fen("4k3/8/8/8/8/8/3PP3/4K3 b - - 0 1"),
//       arrows: (
//         "d2 d3", "d2 d4",
//         "e2 e3", "e2 e4",
//         "e1 d1", 
//         "e1 f1", 
//         "e1 f2", 
//       ),
//       stroke: 0.8pt + black,
//     ),
//   ),
// ) <fig-legal-moves> 
