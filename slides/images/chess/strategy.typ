#set text(white)
#set page(width: 7cm, height: 8cm, margin: 0pt)
#set align(center)
#import "@preview/board-n-pieces:0.9.0": *

#board(
      fen("rnbq1rk1/3nbppp/p2p4/4p3/1p2P3/2NP1N1P/BPP2PP1/R1BQR1K1 w - - 1 12"),
      // display-numbers: true, 
      square-size: 0.8cm, 
      arrows: ("c3 d5",),
      stroke: 0.8pt + black,
    )
