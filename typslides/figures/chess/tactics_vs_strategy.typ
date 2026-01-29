#import "@preview/board-n-pieces:0.9.0": *
#let board = board.with(
  white-square-fill: rgb("#d2eeea"),
  black-square-fill: rgb("#567f96"),
  arrow-fill: green,
)

#stack(
    dir: ttb,
    spacing: 0.5cm,
    stack(dir: ltr, spacing: 1cm,
    board(
      fen("r2q1rk1/p3bppp/2n1p3/1p1pN3/2pP4/1PP3P1/P3PPBP/R1BQR1K1 b - - 2 15"),  
      square-size: 0.8cm, 
      arrows: ("e5 c6",),
      stroke: 0.8pt + black,
    ),
    align(top)[
      === Tactics

      - Short term moves 
          - winning a piece
          - checkmate
          - etc.
    ]
  
    ),
    stack(dir: ltr, spacing: 1cm,
    board(
      fen("rnbq1rk1/3nbppp/p2p4/4p3/1p2P3/2NP1N1P/BPP2PP1/R1BQR1K1 w - - 1 12"),
      square-size: 0.8cm, 
      arrows: ("c3 d5",),
      stroke: 0.8pt + black,
    ),
    align(top)[
      === Strategy

      - Long term move: 
        - Good piece positioning
        - Pressure on oponent King
        - Control of center
        - etc.
    ])

  )
