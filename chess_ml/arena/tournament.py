import argparse
import pickle
import torch
import logging
import pandas as pd
from tqdm import tqdm 
from pathlib import Path
from collections import Counter
from chess_ml.env import Rewards
from chess_ml.env.Environment import Environment
from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.model.Convolution import ChessCNN
from chess_ml.model.ResBlock import ChessResBlock
from chess_ml.arena.arena import pit


arc2class = {
    'linear': ChessFeedForward,
    'cnn': ChessCNN,
    'resnet': ChessResBlock
}

def map_model(m) -> dict[str, str | Path | None]: 
    exp_path = m
    parents  = str(exp_path).split('/')
    exp_name = [p for p in parents if any(a in p for a in arc2class.keys())][-1]
    exp_arc  = None
    for k in arc2class.keys(): 
        if k in exp_name: 
            exp_arc = k
    return dict(architecture=exp_arc, name=exp_name, path=exp_path)


def main(path, filter_arc=None, max_models=None, ngames=100):
    log_dir    = Path("logs/tournament/")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    envs    = [Environment(Rewards.ALL) for i in range(ngames)]
    results = dict()

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

    df = pd.DataFrame(0, index=[m['name'] for m in models], columns=[m['name'] for m in models])
    df.index.name   = 'White'
    df.columns.name = 'Black'

    with tqdm(total=len(models)**2, desc="Matchups", unit="Matchups") as pbar:
        i = 0 
        for m1 in models: 
            model1 = arc2class[m1['architecture']]().to(device)
            state  = torch.load(m1['path'], map_location=device)
            model1.load_state_dict(state)
            
            for m2 in models: 
                model2 = arc2class[m2['architecture']]().to(device)
                state  = torch.load(m2['path'], map_location=device)
                model2.load_state_dict(state)

                match_name = f"{m1['name']}__vs__{m2['name']}"
                tqdm.write(match_name)
                with torch.no_grad():
                    matchup_results = pit(model1, model2, envs, log_dir / match_name)
                    results[match_name] = results.get(match_name, Counter()) + matchup_results
                    tqdm.write(str(matchup_results))

                i += 1
                pbar.update(i - pbar.n)


    with open(log_dir / 'results.pkl', 'wb') as f: 
        pickle.dump(results, f)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pit models against each other")
    parser.add_argument('-p', '--path', default=".")
    parser.add_argument('-g', '--games', default=1000, type=int)
    parser.add_argument('-f', '--filter', default=None)
    parser.add_argument('-m', '--max-models', default=None, type=int)
    args = parser.parse_args()
    main(path=args.path, 
         ngames=args.games,
         max_models=args.max_models, 
         filter_arc=args.filter)

