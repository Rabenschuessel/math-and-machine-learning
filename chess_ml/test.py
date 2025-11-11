
import torch
import chess
import chess.svg
import cairosvg
from chess_ml.env.Environment import Environment
from chess_ml.env.Rewards import attack_center
from chess_ml.model.FeedForward import ChessFeedForward


model = ChessFeedForward([512, 512, 512])
torch.save(model.state_dict(), "models/model.pth")

m2 = ChessFeedForward([512, 512, 512])
m2.load_state_dict(torch.load("models/model.pth"))


# model.load_state_dict(torch.load("models/model.pth"))
# env   = Environment([attack_center])
# board      = env.reset()
#
