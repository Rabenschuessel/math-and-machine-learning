#import "@preview/board-n-pieces:0.9.0": *
#import "@preview/neural-netz:0.3.0": draw-network

#let nblocks = 20 
#figure(draw-network((
  (
    type: "input", 
    image: board(fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")),
    height: 6,
    depth: 6,
  ),
  (
    type: "conv", 
    widths:   (0.5, ),
    channels: (64, 64),
    height: 6,
    depth: 6,
    label: " convolution",
  ),
  (
    type: "convres", 
    widths:   (for _ in range(nblocks) { (0.5, ) } ),
    channels: (for _ in range(nblocks) { (64, ) } + (64, ) ),
    height: 6,
    depth: 6,
    label: str(nblocks) + " resblocks",
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
  ),
),
  // show-legend: true,
),
caption: [
    Architecture of the residual block model used.
  ]
) <fig-res-net>
