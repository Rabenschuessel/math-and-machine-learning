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
    # Logging reward values
    tqdm.write("reward order: {}".format([r.__name__ for r in envs[0]._rewards]))
    tqdm.write("mean white reward values: {}".format(rewards_white.abs().sum(dim=(0,1))))
    tqdm.write("mean black reward values: {}".format(rewards_black.abs().sum(dim=(0,1))))

    # Saving white rewards
    path = Path(path) / "games" / "batch-{:04d}".format(batch_nr)
    path.mkdir(parents=True, exist_ok=True)
    # xr.DataArray(
    #     rewards_white.cpu().numpy(), 
    #     dims=['game', 'time', 'reward'], 
    #     coords={ 'reward': [r.__name__ for r in envs[0]._rewards] }
    # ).to_netcdf(path / "rewards_white.nc")
    #
    # xr.DataArray(
    #     rewards_black.cpu().numpy(), 
    #     dims=['game', 'time', 'reward'], 
    #     coords={ 'reward': [r.__name__ for r in envs[0]._rewards] }
    # ).to_netcdf(path / "rewards_black.nc")

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

    tqdm.write("loss: {}".format(loss.item()))
    tqdm.write(str(Counter([env._board.result() for env in envs])))



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




def main(model_path, experiment, batches, batch_size, gamma): 
    env_params = {"rewards": Rewards.ALL}
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir    = Path("logs/rl/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )

    model = ChessCNN()
    if model_path is not None: 
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)


    optim = torch.optim.Adam(model.parameters())
    train(model, optim, batches, batch_size, env_params, log_dir, gamma)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reinforcement learning", 
        description="transform chess puzzle dataset")
    parser.add_argument('-b', '--batches' , default=1000, type=int)
    parser.add_argument('-g', '--batch_size' , default=32, type=int)
    parser.add_argument('-n', '--experiment-name', default=0)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--gamma', default=0.9, type=float)
    args = parser.parse_args()

    main(experiment=args.experiment_name,
         batches=args.batches,
         batch_size=args.batch_size,
         model_path=args.model,
         gamma=args.gamma)




