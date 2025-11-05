import torch
import chess 

def compute_discounted_rewards(rewards, gamma=0.99):
    ''' 
    compute discounted rewards over one game
    '''
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
    return discounted_rewards


def compute_reward(board: chess.Board): 
    if board.is_game_over(): 
        res = board.result()
        if res == "1-0": 
            return 1.0
        if res == "1/2-1/2": 
            return 0.0
        if res == "0-1": 
            return -1.0
    return 0
