import pickle
import argparse
import logging
from pathlib import Path 
from tqdm import tqdm
import torch 
import chess
from pathlib import Path
from collections import Counter

import chess
import torch
from tqdm import tqdm

from chess_ml.env import Rewards
from chess_ml.env.Environment import Environment
from chess_ml.model.Convolution import ChessCNN
from chess_ml.model.ResBlock import ChessResBlock
from chess_ml.model.FeedForward import ChessFeedForward

arc2class = {
    'linear': ChessFeedForward,
    'cnn': ChessCNN,
    'resnet': ChessResBlock
}

def save_game(env, log_dir, gamenr): 
    game = env.get_game()
    with open(log_dir / f"game--{gamenr:06d}.pgn", "w") as f:
        print(game, file=f)


def p(model1, model2, ngames, batch_size, log_dir): 
    log_dir.mkdir(parents=True, exist_ok=True)
    envs = [Environment() for i in range(batch_size)]
    result = Counter()
    color           = chess.WHITE
    boards          = [env.reset() for env in envs]
    finished_games  = 0

    with tqdm(total=ngames, desc="Games", unit="Games") as pbar:
        while True: 
            if color == chess.WHITE: 
                moves, _ = model1.predict(boards)
            else: 
                moves, _ = model2.predict(boards)

            # step all moves
            color = not color
            boards = []
            for env, move in zip(envs, moves): 
                board, done = env.step(move)

                # reset if next turn white to play
                if done and color == chess.WHITE: 
                    save_game(env, log_dir, finished_games)
                    result += Counter([env._board.result()])

                    board = env.reset()
                    pbar.update(finished_games - pbar.n)
                    finished_games += 1
                    if finished_games >= ngames: 
                        return result

                # push to new boards
                boards.append(board)




def map_model(m) -> dict[str, str | Path | None]: 
    exp_path = m
    parents  = str(exp_path).split('/')
    exp_name = [p for p in parents if any(a in p for a in arc2class.keys())][-1]
    exp_arc  = None
    for k in arc2class.keys(): 
        if k in exp_name: 
            exp_arc = k
    return dict(architecture=exp_arc, name=exp_name, path=exp_path)


def main(path, filter_arc=None, max_models=None, ngames=100, batch_size=20):
    log_dir    = Path("logs/tournament/")
    if filter_arc is not None: 
        log_dir = log_dir / filter_arc
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = dict()
    print(f'using device: {device}')

    path = Path(path)
    models = [p for p in path.rglob('models/')]
    models = [sorted(p.iterdir())[-1] for p in models if len(list(p.iterdir())) > 0]
    models = list(map(map_model, models))
    if filter_arc is not None: 
        models = list(filter(lambda x: x['architecture'] == filter_arc, models))
    if max_models is not None: 
        models = models[:max_models]
    print("found models: ")
    print([m['name'] for m in models])

    with tqdm(total=len(models)*(len(models)-1), desc="Matchups", unit="Matchups") as pbar:
        i = 0 
        for m1 in models: 
            model1 = arc2class[m1['architecture']]().to(device)
            state  = torch.load(m1['path'], map_location=device)
            model1.load_state_dict(state)
            model1.eval()

            for m2 in [m for m in models if m != m1]: 
                model2 = arc2class[m2['architecture']]().to(device)
                state  = torch.load(m2['path'], map_location=device)
                model2.load_state_dict(state)
                model2.eval()

                match_name = f"{m1['name']}__vs__{m2['name']}"
                tqdm.write(match_name)
                with torch.no_grad():
                    matchup_results = p(model1, model2, ngames, batch_size, log_dir / match_name)
                    results[match_name] = results.get(match_name, Counter()) + matchup_results
                    tqdm.write(str(matchup_results))

                i += 1
                pbar.update(i - pbar.n)

    with open(log_dir / 'results.pkl', 'wb') as f: 
        pickle.dump(results, f)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pit models against each other")
    parser.add_argument('-p', '--path', default=".")
    parser.add_argument('-n', '--ngames', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-f', '--filter', default=None)
    parser.add_argument('-m', '--max-models', default=None, type=int)
    args = parser.parse_args()
    main(path=args.path, 
         ngames=args.ngames,
         batch_size=args.batch_size,
         max_models=args.max_models, 
         filter_arc=args.filter)
