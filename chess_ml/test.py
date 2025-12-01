
import torch
import chess
import chess.svg
import cairosvg
from torch.distributions import one_hot_categorical
from chess_ml.env.Environment import Environment
from chess_ml.env.Rewards import attack_center
from chess_ml.model import Convolution
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




import chess
import torch
from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.model import ChessNN
model    = ChessFeedForward([512, 512, 512])
loss_fn  = torch.nn.CrossEntropyLoss()
board    = chess.Board()
fen      = board.fen()
fen      = [fen, fen, fen]
moves    = ['a2a4','a2a3', 'b2b4']




import chess
import pandas as pd
df = pd.read_csv('./data/lichess_puzzle_transformed.csv', nrows=5000)


df["Moves"].str.split()



mask = df["Moves"].transform( lambda r: 
        all([
            chess.Move.from_uci(m).promotion in [None, chess.QUEEN] 
            for m in r.split()
        ]))

mask  = moves.apply(lambda r: not all([
    m.promotion is None or
    m.promotion is chess.QUEEN
    for m in r
]))

df[mask]["Moves"]


up = df[~df["Themes"].str.contains("underPromotion")]

len(up)




a = chess.Move.from_uci("a2b2")
print(a.promotion in [None, chess.PieceType])

a = chess.Move.from_uci("b7b8r")
print(a.promotion in [None, chess.QUEEN])

a = chess.Move.from_uci("a2b8q")
print(a.promotion in [None, chess.QUEEN])




import importlib 
import chess
import chess_ml.model.Convolution as Convolution
from chess_ml.model import ChessNN 

importlib.reload(Convolution)
model = Convolution.ChessCNN()
board = chess.Board()

t = ChessNN.board_to_tensor(board).unsqueeze(0)
model(t).shape

model.predict(board, epsilon=1)




model.predict(board, epsilon=0)
