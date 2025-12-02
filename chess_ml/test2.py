import torch
from chess_ml.env import Rewards
from  chess_ml.env.Environment import Environment
from chess import Move, Board



rewards = Rewards.ALL
moves1   = ["f2f3", 'e2e3', 'g2g4', 'd1h5', "a2a3"]
env1     = Environment(rewards)

moves2  = ["e2e4", "f2f3", "d2d4", "g2g4", "d1h5"]
env2     = Environment(rewards)


for move1, move2 in zip(moves1, moves2): 
    print("new iteration: \nenv1: {}\nenv2: {}".format(env1.reward_hist, env2.reward_hist))
    move1 = Move.from_uci(move1)
    env1.step(move1)

    move2 = Move.from_uci(move2)
    env2.step(move2)

len(env2.reward_hist)
len(env1.reward_hist)



rewards_white1, rewards_black1 = env1.get_rewards()
rewards_white2, rewards_black2 = env2.get_rewards()

rewards_white = [rewards_white1, rewards_white2]
print(torch.tensor(rewards_white))

torch.tensor(rewards_white).abs().mean(dim=(0,1))

