from  chess_ml.env.Environment import Environment
import torchrl
import torch 
import chess
from chess import Move
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward



board  = chess.Board()
boards = [board, board, board]
model  = ChessFeedForward()



color = chess.WHITE
log_probs_white = []
rewards_white   = []
log_probs_black = []
rewards_black   = []

envs   = [Environment() for i in range(8)]
boards = [env.reset() for env in envs]
done   = [False]

while not all(done): 
    moves, log_probs = model.predict(boards)
    boards, rewards, done = zip(*[env.step(move) for env, move in zip(envs, moves)])

    if color is chess.WHITE: 
        log_probs_white.append(log_probs)
        rewards_white.append(rewards)
    else: 
        log_probs_black.append(log_probs)
        rewards_black.append(rewards)

    color = not color 
    print(sum(done))

# rewards_white = torch.stack(rewards_white)
log_probs_white = torch.stack(log_probs_white)
#
# rewards_black = torch.stack(rewards_black)
log_probs_black = torch.stack(log_probs_black)

print("rewards white: {}".format(rewards_white))
print("log probs white: {}".format(log_probs_white))

print("rewards black: {}".format(rewards_black))
print("log probs black: {}".format(log_probs_black))




################################################################################
### test
################################################################################
import torch 
from torchrl.objectives.value.functional import reward2go



rewards = torch.ones((3, 10))
done    = torch.zeros((3, 10), dtype=torch.bool)
done[[0, 1, 2], [1, 2, 3]] = True


reward2go(rewards, done, 0.9, time_dim=-1)
