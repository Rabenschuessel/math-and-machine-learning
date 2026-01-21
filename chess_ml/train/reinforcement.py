import textwrap
import argparse
import logging
from pathlib import Path 
from tqdm import tqdm
import torch 
import chess
import xarray as xr
from pathlib import Path
from collections import Counter

import chess
import torch
import xarray as xr
from tqdm import tqdm
from torchrl.objectives.value.functional import reward2go

from chess_ml.env import Rewards
from chess_ml.env.PositionSampler import get_position_sampler
from chess_ml.env.Environment import Environment
from chess_ml.model.Convolution import ChessCNN
from chess_ml.model.ResBlock import ChessResBlock
from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.logging import setup_logging, CSVLogger, summarize_batch_stats, init_csv_logger, save_rewards_and_games


def train_batch(model,
                optim,
                envs,
                log_dir: Path,
                batch_nr: int,
                gamma: float,
                csv_logger: CSVLogger,
                save_artifacts_every: int = 0):

    """Run one batch of self-play games, compute policy gradient loss, optimize model,
    and log metrics and artifacts.
    """

    color           = chess.WHITE
    log_probs_white = []
    done_white      = []
    log_probs_black = []
    done_black      = []
    boards          = [env.reset() for env in envs]
    done            = [False] * len(envs)

    with tqdm(total=len(envs), desc="Games", unit="Games") as pbar:
        while not all(done):
            moves, log_probs = model.predict(boards)
            boards, done = zip(*[env.step(move) for env, move in zip(envs, moves)])

            done_tensor = torch.tensor(done, dtype=torch.bool)

            if color == chess.WHITE:
                log_probs_white.append(log_probs)
                done_white.append(done_tensor)
                color = chess.BLACK
            else:
                log_probs_black.append(log_probs)
                done_black.append(done_tensor)
                color = chess.WHITE

            pbar.update(sum(done) - pbar.n)

    rewards_white, rewards_black = zip(*[env.get_rewards() for env in envs])
    rewards_white   = torch.tensor(rewards_white)
    rewards_black   = torch.tensor(rewards_black)
    stats = summarize_batch_stats(envs, rewards_white, rewards_black)

    # compute loss
    loss = 0
    if len(log_probs_white) > 0:
        log_probs_white = torch.stack(log_probs_white)
        done_white      = torch.stack(done_white)
        rewards_white   = rewards_white.sum(dim=-1).permute(1, 0)
        rewards_white   = reward2go(rewards_white, done_white, gamma)
        loss_white      = (- rewards_white * log_probs_white).sum()
        loss += loss_white
    if len(log_probs_black) > 0:
        log_probs_black = torch.stack(log_probs_black)
        done_black      = torch.stack(done_black)
        rewards_black   = rewards_black.sum(dim=-1).permute(1, 0)
        rewards_black   = reward2go(rewards_black, done_black, gamma)
        loss_black      = (- rewards_black * log_probs_black).sum()
        loss += loss_black


    optim.zero_grad()
    if loss != 0:
        loss.backward()
    optim.step()

    tqdm.write("Batch Summary:")
    tqdm.write("loss: {}".format(loss.item()))
    tqdm.write("results: {}".format(str(Counter([env._board.result() for env in envs]))))
    tqdm.write("mean game length: {}".format(sum([len(env._board.move_stack) for env in envs])/len(envs)))
    row = {
        "batch": batch_nr,
        "loss": float(loss.item()),
        "loss_white": float(loss_white.item()),
        "loss_black": float(loss_black.item()),
        "mean_len": stats["mean_len"],
        "w_wins": stats["w_wins"],
        "b_wins": stats["b_wins"],
        "draws": stats["draws"],
    }

    for i, name in enumerate(stats["reward_names"]):
        row[f"white_mean_{name}"] = float(stats["white_mean"][i])
        row[f"black_mean_{name}"] = float(stats["black_mean"][i])
        row[f"white_std_{name}"] = float(stats["white_std"][i])
        row[f"black_std_{name}"] = float(stats["black_std"][i])

    tqdm.write(
        f"Batch {batch_nr:04d} | loss={loss.item():.3f} "
        f"(W={loss_white.item():.3f}, B={loss_black.item():.3f}) | "
        f"len={stats['mean_len']:.1f} | "
        f"1-0={stats['w_wins']} 0-1={stats['b_wins']} 1/2-1/2={stats['draws']}")

    logging.info(
        f"batch={batch_nr} loss={loss.item():.6f} loss_w={loss_white.item():.6f} loss_b={loss_black.item():.6f} "
        f"mean_len={stats['mean_len']:.2f} w_wins={stats['w_wins']} b_wins={stats['b_wins']} draws={stats['draws']} "
        # f"return_w_mean={returns_white.mean().item():.6f} return_b_mean={returns_black.mean().item():.6f}"
    )

    csv_logger.log(row)

    if save_artifacts_every > 0 and (batch_nr % save_artifacts_every == 0):
        save_rewards_and_games(log_dir, envs, rewards_white, rewards_black, batch_nr)


def train(model, optim, batches, batch_size, env_params, log_dir, csv_logger, gamma): 
    models_dir = Path(log_dir/"models")
    models_dir.mkdir(parents=True, exist_ok=True)

    envs = [Environment(**env_params) for i in range(batch_size)]

    for batch in tqdm(range(batches), desc="Batches", unit="Batches"): 
        train_batch(model, optim, envs, log_dir, batch, gamma, csv_logger)

        if batch % 10 == 0: 
            tqdm.write("Save Checkpoint")
            torch.save(model.state_dict(), models_dir / f"checkpoint-{batch}.pth")


    ds = [xr.open_dataset(entry / 'rewards.nc') 
            for entry in (log_dir/"games").iterdir() 
            if entry.is_dir() and 'batch' in entry.name]
    ds = xr.concat(ds, dim='batch', join='outer')
    ds = ds.assign_coords(batch=range(ds.sizes['batch']))
    ds.to_netcdf(log_dir / "rewards.nc")




def main(*, model_path, experiment, architecture, batches, batch_size, gamma, rewards, position_type='standard', positions_file=None): 
    name2reward = {r.__name__:r for r in Rewards.ALL}
    rew = set()
    for r in rewards: 
        if r in name2reward.keys(): 
            rew.add(name2reward[r])
        else: 
            rew.update(Rewards.reward_sets[r])
    env_params  = {"rewards": rew}
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(env_params)

    # Create position sampler
    if position_type == 'file':
        if positions_file is None:
            raise ValueError("--positions-file must be specified when using --position-type file")
        position_sampler = get_position_sampler(position_type, file_path=positions_file)
    else:
        position_sampler = get_position_sampler(position_type)
    env_params["position_sampler"] = position_sampler

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
                rewards: {rewards}
                position_type: {position_type}
                positions_file: {positions_file}"""))
    csv_logger = init_csv_logger(log_dir, env_params['rewards'])

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

    logging.info("Train model")
    optim = torch.optim.Adam(model.parameters())
    train(model, optim, batches, batch_size, env_params, log_dir, csv_logger, gamma)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reinforcement learning", 
        description="transform chess puzzle dataset")

    parser.add_argument("--gamma", default=0.997, type=float)
    parser.add_argument('-a', '--architecture', choices=['linear', 'cnn', 'resnet'], default='resnet')
    parser.add_argument('-b', '--batches' , default=1000, type=int)
    parser.add_argument('-g', '--batch_size' , default=16, type=int)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-n', '--experiment-name', default=0)
    parser.add_argument('-r', '--rewards', choices=[r.__name__ for r in Rewards.ALL] + list(Rewards.reward_sets.keys()), nargs="+", default=[])
    parser.add_argument('--positions-file', default=None, help='Path to file containing FEN positions (required for --position-type file)')
    parser.add_argument('--position-type', default='standard')
    args = parser.parse_args()

    main(experiment=args.experiment_name,
         batches=args.batches,
         batch_size=args.batch_size,
         model_path=args.model,
         gamma=args.gamma, 
         architecture=args.architecture, 
         rewards=args.rewards,
         position_type=args.position_type,
         positions_file=args.positions_file)

