from torch.distributions.utils import logits_to_probs
import model 
import chess_util 
from rewards import *
from importlib import reload
reload(model)
reload(chess_util)

import torch
from torch import optim
from torch.distributions import Categorical
import chess
from model import ChessModel
from chess_util import *






def train(board : chess.Board,
          model : ChessModel,
          optimizer, number_of_games=10):
    '''
    selfplay a number of games and 
    '''

    for game in range(number_of_games):
        board.reset()
        log_probs_black = []
        rewards_black   = []
        log_probs_white = []
        rewards_white   = []
        turn = board.turn

        while not board.is_game_over():
            # get the action
            x         = tensor_from_position(board)
            y         = model(x)
            move, log = sample_move(board, y)
            log = log.unsqueeze(0)

            # step and evaluate
            board.push(move)
            reward = compute_reward(board)

            # append to the current player
            if turn == chess.WHITE: 
                log_probs_white.append(log)
                rewards_white.append(reward)
            if turn == chess.BLACK: 
                log_probs_black.append(log)
                rewards_black.append(reward)

            # flip after every move 
            # so from the model perspective it is always white to play
            board = board.mirror()
            turn  = not turn 


        # white rewards
        policy_loss = []
        discounted_rewards_white = compute_discounted_rewards(rewards_white)
        for log_prob, Gt in zip(log_probs_white, discounted_rewards_white):
            policy_loss.append(-log_prob * Gt)

        # black rewards 
        discounted_rewards_black = compute_discounted_rewards(rewards_black)
        for log_prob, Gt in zip(log_probs_black, discounted_rewards_black):
            policy_loss.append(-log_prob * Gt)

        print("accumulated rewards: \nwhite: {}\nblack: {}".format(sum(rewards_white), sum(rewards_black)))

        # apply rewards
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()



board = chess.Board()
model = ChessModel().to("cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-2)

train(board, model, optimizer)


