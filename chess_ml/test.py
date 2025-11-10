import chess
from chess_ml.env.Environment import Environment
from chess_ml.env.Rewards import attack_center
from chess_ml.model.FeedForward import ChessFeedForward


model = ChessFeedForward()

env = Environment([attack_center])

board      = env.reset()
move, prob = model.predict(board)
print(move)

board, reward = env.step(move)
print(reward)





