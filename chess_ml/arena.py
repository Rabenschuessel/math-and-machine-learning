import argparse
from collections import Counter
import torch
import chess
import chess.svg
import logging
from pathlib import Path
from tqdm import tqdm
from chess_ml.env.Environment import Environment
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.model.Convolution import ChessCNN




def pit(model1, model2, envs, log_dir):
    color = chess.WHITE
    boards = [env.reset() for env in envs]
    done   = [False]

    with tqdm(total=len(envs), desc="Games", unit="Games") as pbar: 
        while not all(done): 
            if color is chess.WHITE: 
                moves, log_probs = model1.predict(boards)
            else: 
                moves, log_probs = model2.predict(boards)
            boards, done = zip(*[env.step(move) for env, move in zip(envs, moves)])

            color = not color 
            pbar.update(sum(done) - pbar.n)

    # Logging 
    for gamenr, env in enumerate(envs): 
        game = env.get_game()
        with open(log_dir / "game-{:06d}.pgn".format(gamenr), "w") as f:
            print(game, file=f)

    return (Counter([env._board.result() for env in envs]))





def main(path1, path2, experiment, games):
    log_dir    = Path("logs/arena/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )
    games_dir = Path(log_dir/"models")
    games_dir.mkdir(parents=True, exist_ok=True)

    # m1 = ChessFeedForward([512, 512, 512])
    m1 = ChessCNN()
    if path1 is not None: 
        state = torch.load(path1, map_location="cpu")
        m1.load_state_dict(state)
    m1.eval()

    # m2 = ChessFeedForward([512, 512, 512])
    m2 = ChessCNN()
    if path2 is not None: 
        state = torch.load(path2, map_location="cpu")
        m2.load_state_dict(state)
    m2.eval()

    with torch.no_grad():
        envs = [Environment() for i in range(games)]
        results = pit(m1, m2, envs, log_dir)
        print(results)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pit models against each other")
    parser.add_argument('-1', '--model1', default=None)
    parser.add_argument('-2', '--model2', default=None)
    parser.add_argument('-e', '--experiment', default=0, type=int)
    parser.add_argument('-g', '--games', default=100, type=int)
    args = parser.parse_args()
    main(path1=args.model1,
         path2=args.model2,
         games=args.games,
         experiment=args.experiment)



ChessNN()
