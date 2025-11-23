
import torch
import chess
import chess.svg
import cairosvg
from chess_ml.env.Environment import Environment
from chess_ml.env.Rewards import attack_center
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward


model = ChessFeedForward([512, 512, 512])
torch.save(model.state_dict(), "models/model.pth")

m2 = ChessFeedForward([512, 512, 512])
m2.load_state_dict(torch.load("models/model.pth"))


# model.load_state_dict(torch.load("models/model.pth"))
# env   = Environment([attack_center])
# board      = env.reset()
#


import importlib 
from chess import Board, Move
import chess_ml.env.Rewards as Rewards



importlib.reload(Rewards)
board   = Board()
rewards = [Rewards.attack_center, Rewards.early_development, Rewards.win]
moves   = ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3"]
for move in moves: 
    m    = Move.from_uci(move)
    board.push(m)
    eval = [reward(board, m) for reward in rewards]
    print("\nevalutation of move: {}\n  attacking center: {}\n  early piece development: {}\n  win: {}"
          .format(m, *eval))




import chess
from chess import Board, Move
board = Board()
moves = ["f3","e6","g4","Qh4#"]
for move in moves: 
    board.push_san(move)


import chess.pgn
game = chess.pgn.Game.from_board(board)
with open("game.pgn", "w") as f:
    print(game, file=f)


