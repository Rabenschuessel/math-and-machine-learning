import chess 
import chess.svg

fen = '5k2/r1p2rp1/p3R3/1p6/1P6/P5K1/5P2/7R w - - 8 49' 
board = chess.Board(fen)

arr = chess.svg.Arrow(chess.H1, chess.H8)
svg = chess.svg.board(board, arrows=[arr])


with open("matein1.svg", "w") as f: 
    f.write(svg)





import chess 
import chess.svg

board = chess.Board()
board.push(chess.Move.from_uci("a2a4"))
board.push(chess.Move.from_uci("d7d5"))
svg   = chess.svg.board(board, lastmove=chess.Move.from_uci("d7d5"))


with open(".svg", "w") as f: 
    f.write(svg)



fen = 'rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4'
board = chess.Board(fen)
arr   = [
    chess.svg.Arrow(chess.C4, chess.D5),
    chess.svg.Arrow(chess.C1, chess.G5),
    chess.svg.Arrow(chess.G1, chess.F3),
]
svg   = chess.svg.board(board, arrows=arr)


with open("strategy.svg", "w") as f: 
    f.write(svg)
