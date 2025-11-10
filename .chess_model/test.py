from torch.distributions import Categorical
import model 
import cairosvg
import chess_util 
from importlib import reload
reload(model)
reload(chess_util)

import torch
import chess
import chess.svg
from model import ChessModel
from chess_util import *

board = chess.Board()
model = ChessModel().to("cpu")



game = 2
unflip = False
i = 0
while not board.is_game_over():
    print("move {}".format(i))
    i = i + 1
    # flip colors so that network 'only plays white'
    if board.turn == chess.BLACK: 
        board = board.mirror()
        unflip = True

    x      = tensor_from_position(board)
    y      = model(x)
    move,_ = sample_move(board, y)

    board.push(move)

    if unflip: 
        unflip = False
        board = board.mirror()


    svg = chess.svg.board(board)
    cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                     write_to="games/game-{:02d}-move-{:03d}.png".format(game, i))

