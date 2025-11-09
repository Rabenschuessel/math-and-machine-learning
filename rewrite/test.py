import chess
from Environment import Environment
from model.FeedForward import ChessFeedForward
from Rewards import attack_center


model = ChessFeedForward()

env = Environment([attack_center])

board      = env.reset()
move, prob = model.predict(board)
print(move)

board, reward = env.step(move)
print(reward)


