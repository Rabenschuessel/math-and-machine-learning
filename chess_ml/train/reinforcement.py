import textwrap
import argparse
import logging
from pathlib import Path 
from chess_ml.env import Rewards
from  chess_ml.env.Environment import Environment
from tqdm import tqdm
import torch 
import chess
import xarray as xr
from collections import Counter
from torchrl.objectives.value.functional import reward2go
from chess import Move
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.Convolution import ChessCNN
from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.model.ResBlock import ChessResBlock



def log_batch(path, envs, rewards_white, rewards_black, batch_nr): 
    # Saving white rewards
    path = Path(path) / "games" / "batch-{:04d}".format(batch_nr)
    path.mkdir(parents=True, exist_ok=True)

    (xr.concat([ 
        xr.Dataset(
            data_vars={
                r.__name__: (["game", "turn"],t.cpu().numpy()[:,:,i])
                for i, r in enumerate(envs[0]._rewards)
            }, 
            coords=dict(
                game=("game", range(t.cpu().numpy().shape[0])),
                turn=("turn", range(t.cpu().numpy().shape[1])),
            )
        )
        for t in [rewards_black, rewards_white]], dim='color', join='outer')
     .assign_coords(color=[chess.BLACK, chess.WHITE])
     .to_netcdf(path / "rewards.nc"))


    # Saving games as pgn
    for gamenr, env in enumerate(envs): 
        game = env.get_game()
        with open(path / "game-{:06d}.pgn".format(gamenr), "w") as f:
            print(game, file=f)



def train_batch(model, optim, envs, log_dir, batch_nr, gamma):
    color = chess.WHITE
    log_probs_white = []
    done_white      = []

    log_probs_black = []
    done_black      = []

    boards = [env.reset() for env in envs]
    done   = [False]

    with tqdm(total=len(envs), desc="Games", unit="Games") as pbar: 
        while not all(done): 
            moves, log_probs = model.predict(boards)
            boards, done = zip(*[env.step(move) for env, move in zip(envs, moves)])

            if color is chess.WHITE: 
                log_probs_white.append(log_probs)
                done_white.append(torch.tensor(done))
            else: 
                log_probs_black.append(log_probs)
                done_black.append(torch.tensor(done))

            color = not color 
            pbar.update(sum(done) - pbar.n)

    # transform to torch tensors
    rewards_white, rewards_black = zip(*[env.get_rewards() for env in envs])
    rewards_white   = torch.tensor(rewards_white)
    log_probs_white = torch.stack(log_probs_white)
    done_white      = torch.stack(done_white)
    rewards_black   = torch.tensor(rewards_black)
    log_probs_black = torch.stack(log_probs_black)
    done_black      = torch.stack(done_black)

    log_batch(log_dir, envs, rewards_white, rewards_black, batch_nr)

    # compute loss
    rewards_white = rewards_white.sum(dim=-1).permute(1, 0)
    rewards_black = rewards_black.sum(dim=-1).permute(1, 0)
    rewards_white = reward2go(rewards_white, done_white, gamma)
    rewards_black = reward2go(rewards_black, done_black, gamma)
    loss_white    = (- rewards_white * log_probs_white).sum()
    loss_black    = (- rewards_black * log_probs_black).sum()
    loss          = loss_white + loss_black

    # optimize
    optim.zero_grad()
    loss.backward()
    optim.step()

    tqdm.write("Batch Summary:")
    tqdm.write("loss: {}".format(loss.item()))
    tqdm.write("results: {}".format(str(Counter([env._board.result() for env in envs]))))
    tqdm.write("mean game length: {}".format(sum([len(env._board.move_stack) for env in envs])/len(envs)))



def train(model, optim, batches, batch_size, env_params, log_dir, gamma): 
    models_dir = Path(log_dir/"models")
    models_dir.mkdir(parents=True, exist_ok=True)

    envs = [Environment(**env_params) for i in range(batch_size)]

    for batch in tqdm(range(batches), desc="Batches", unit="Batches"): 
        train_batch(model, optim, envs, log_dir, batch, gamma)

        if batch % 10 == 0: 
            tqdm.write("Save Checkpoint")
            torch.save(model.state_dict(), models_dir / f"checkpoint-{batch}.pth")


    ds = [xr.open_dataset(entry / 'rewards.nc') 
            for entry in (log_dir/"games").iterdir() 
            if entry.is_dir() and 'batch' in entry.name]
    ds = xr.concat(ds, dim='batch', join='outer')
    ds = ds.assign_coords(batch=range(ds.sizes['batch']))
    ds.to_netcdf(log_dir / "rewards.nc")




def main(*, model_path, experiment, architecture, batches, batch_size, gamma, rewards): 
    name2reward = {r.__name__:r for r in Rewards.ALL}
    env_params  = {"rewards": [name2reward[r] for r in rewards]}
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create a new log dir in 'logs/rl/experiment-<name>/<x>' where x starts from 0
    log_dir     = Path("logs/rl/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted([int(x.name) for x in log_dir.iterdir() if x.is_dir() and x.name.isdigit()])
    new         = 0 if len(experiments) == 0 else (int(experiments[-1]) + 1)
    log_dir     = (log_dir / "{:03d}".format(new))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,      
        format='%(message)s'  
    )
    with open(log_dir / 'hparams.txt', 'w') as f:
        f.write(textwrap.dedent(f"""\
                model_path: {model_path}
                architecture: {architecture}
                batches: {batches}
                batch_size: {batch_size}
                gamma: {gamma}
                rewards: {rewards}"""))

    logging.info('loading model architecture')
    if architecture == 'linear': 
        model = ChessFeedForward()
    elif architecture == 'cnn': 
        model = ChessCNN()
    else: 
        model = ChessResBlock()

    if model_path is not None: 
        logging.info('loading model weights')
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)


    logging.info('creating optimizer')
    optim = torch.optim.Adam(model.parameters())
    logging.info('train model')
    train(model, optim, batches, batch_size, env_params, log_dir, gamma)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reinforcement learning", 
        description="transform chess puzzle dataset")
    parser.add_argument('-b', '--batches' , default=1000, type=int)
    parser.add_argument('-g', '--batch_size' , default=16, type=int)
    parser.add_argument('-n', '--experiment-name', default=0)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('-a', '--architecture', choices=['linear', 'cnn', 'resnet'], default='resnet')
    parser.add_argument('-r', '--rewards', choices=[r.__name__ for r in Rewards.ALL], nargs="+")
    args = parser.parse_args()

    main(experiment=args.experiment_name,
         batches=args.batches,
         batch_size=args.batch_size,
         model_path=args.model,
         gamma=args.gamma, 
         architecture=args.architecture, 
         rewards=args.rewards)




