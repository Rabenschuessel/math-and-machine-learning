'''
This module provides a training procedure for reinforcement learning

    python -m chess_ml.train.reinforcement
'''


import argparse
import logging
import chess
import chess.pgn
import torch 
import pandas as pd
from torch import optim
from torch import device
from tqdm import tqdm 
from pathlib import Path
from typing import Union

from chess_ml.env import Rewards
from chess_ml.env.Environment import Environment
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward



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


################################################################################
#### Train
################################################################################
def train(model : ChessNN,
          env   : Environment,
          optimizer, 
          number_of_games=10, 
          log_dir="log/rl/",
          device:Union[str,device]="cpu"
          ):
    '''
    selfplay a number of games and 
    '''

    for game_number in tqdm(range(number_of_games),
                      total=number_of_games,
                      desc ="Selfplay Games",
                      unit ="Game"):


        logging.info("\nGame: {}".format(game_number))
        over            = False
        board           = env.reset()
        log_probs_black = []
        rewards_black   = []
        log_probs_white = []
        rewards_white   = []
        turn            = board.turn

        while not over:
            # get the action
            move, log = model.predict(board)
            log       = log.unsqueeze(0)

            # step and evaluate
            board, reward, over = env.step(move)

            # append to the current player and flip turn
            if turn == chess.WHITE: 
                log_probs_white.append(log)
                rewards_white.append(reward)
            if turn == chess.BLACK: 
                log_probs_black.append(log)
                rewards_black.append(reward)
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

        # logging 
        logging.info("  Accumulated rewards: \n    white: {}\n    black: {}"
              .format(sum(rewards_white), sum(rewards_black)))

        # save pgn to logs
        game = env.get_game()
        game.headers["Round"] = str(game_number)
        game.headers["Accumulated-Rewards-White"] = str(sum(rewards_white))
        game.headers["Accumulated-Rewards-Black"] = str(sum(rewards_black))
        path = Path(log_dir) / "games"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "game-{:06d}.pgn".format(game_number), "w") as f:
            print(game, file=f)
        (pd.DataFrame(env.reward_log)
            .to_csv(path / "game-{:06d}.rewards".format(game_number)))


        # apply rewards
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()


################################################################################
#### Main
################################################################################
def main(experiment=1,
         number_of_games=100, 
         model_path=None): 
    log_dir    = Path("logs/rl/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )


    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env       = Environment(rewards=Rewards.all_rewards)
    model     = ChessFeedForward([512, 512, 512])
    if model_path is not None: 
        model.load_state_dict(torch.load(model_path))
    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train(model, env, optimizer, number_of_games=number_of_games, log_dir=log_dir)

    print("Save Model")
    models_dir = Path(log_dir/"models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"final-model.pth")



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        prog="immitation learning", 
        description="transform chess puzzle dataset")
    parser.add_argument('-g', '--games' , default=1000, type=int)
    parser.add_argument('-e', '--experiment', default=1, type=int)
    parser.add_argument('-m', '--model', default=None)
    args = parser.parse_args()

    main(experiment=args.experiment, number_of_games=args.games, model_path=args.model)
