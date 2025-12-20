import argparse
import logging
from pathlib import Path
from collections import Counter

import chess
import torch
import xarray as xr
from tqdm import tqdm
from torchrl.objectives.value.functional import reward2go

from chess_ml.env import Rewards
from chess_ml.env.Environment import Environment
from chess_ml.model.Convolution import ChessCNN
from chess_ml.utils.csv_logger import CSVLogger


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure file-based logging for batch summaries and diagnostics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "log.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("chess_rl")


def init_csv_logger(log_dir: Path, reward_fns) -> CSVLogger:
    """
    Create a CSV logger that writes one row per batch for easy plotting/analysis.
    """
    reward_names = [r.__name__ for r in reward_fns]

    fieldnames = (
        ["batch", "loss", "loss_white", "loss_black", "mean_len", "w_wins", "b_wins", "draws"]
        + [f"white_mean_{n}" for n in reward_names]
        + [f"black_mean_{n}" for n in reward_names]
        + [f"white_std_{n}" for n in reward_names]
        + [f"black_std_{n}" for n in reward_names]
    )

    return CSVLogger(path=log_dir / "metrics.csv", fieldnames=fieldnames)


def save_rewards_and_games(log_dir: Path, envs, rewards_white: torch.Tensor, rewards_black: torch.Tensor, batch_nr: int):
    """
    Save per-component rewards (NetCDF) and sample games (PGN) for later inspection.
    """
    games_dir = Path(log_dir) / "games" / f"batch-{batch_nr:04d}"
    games_dir.mkdir(parents=True, exist_ok=True)

    reward_fns = envs[0]._rewards

    ds = xr.concat(
        [
            xr.Dataset(
                data_vars={
                    r.__name__: (["game", "turn"], t.cpu().numpy()[:, :, i])
                    for i, r in enumerate(reward_fns)
                },
                coords=dict(
                    game=("game", range(t.shape[0])),
                    turn=("turn", range(t.shape[1])),
                ),
            )
            for t in [rewards_black, rewards_white]
        ],
        dim="color",
        join="outer",
    ).assign_coords(color=[chess.BLACK, chess.WHITE])

    ds.to_netcdf(games_dir / "rewards.nc")

    for gamenr, env in enumerate(envs):
        game = env.get_game()
        with open(games_dir / f"game-{gamenr:06d}.pgn", "w") as f:
            print(game, file=f)


def summarize_batch_stats(envs, rewards_white: torch.Tensor, rewards_black: torch.Tensor):
    """
    Produce human-readable batch statistics for console and structured logs.
    """
    reward_names = [r.__name__ for r in envs[0]._rewards]

    white_mean = rewards_white.mean(dim=(0, 1)).cpu().numpy()
    black_mean = rewards_black.mean(dim=(0, 1)).cpu().numpy()
    white_std = rewards_white.std(dim=(0, 1)).cpu().numpy()
    black_std = rewards_black.std(dim=(0, 1)).cpu().numpy()

    results = [env._board.result() for env in envs]
    cnt = Counter(results)

    stats = {
        "reward_names": reward_names,
        "white_mean": white_mean,
        "black_mean": black_mean,
        "white_std": white_std,
        "black_std": black_std,
        "w_wins": cnt.get("1-0", 0),
        "b_wins": cnt.get("0-1", 0),
        "draws": cnt.get("1/2-1/2", 0),
        "mean_len": float(rewards_white.shape[1]),
    }
    return stats


def compute_policy_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    normalize_advantage: bool = True,
):
    """
    Compute a REINFORCE-style policy gradient loss using reward-to-go.

    Shapes:
        log_probs: [T, B]
        rewards:   [T, B]
        done:      [T, B]
    """
    returns = reward2go(rewards, done, gamma)

    if normalize_advantage:
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        adv = returns

    loss = (-adv * log_probs).sum()
    return loss, returns


def train_batch(
    model,
    optim,
    envs,
    log_dir: Path,
    batch_nr: int,
    gamma: float,
    csv_logger: CSVLogger,
    logger: logging.Logger,
    save_artifacts_every: int = 10,
    normalize_advantage: bool = True,
):
    """
    Run one batch of self-play games, compute policy gradient loss, optimize model,
    and log metrics and artifacts.
    """
    color = chess.WHITE

    log_probs_white = []
    done_white = []

    log_probs_black = []
    done_black = []

    boards = [env.reset() for env in envs]
    done = [False] * len(envs)

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
    rewards_white = torch.tensor(rewards_white, dtype=torch.float32)
    rewards_black = torch.tensor(rewards_black, dtype=torch.float32)

    log_probs_white = torch.stack(log_probs_white) if len(log_probs_white) > 0 else torch.empty((0, len(envs)))
    log_probs_black = torch.stack(log_probs_black) if len(log_probs_black) > 0 else torch.empty((0, len(envs)))

    done_white = torch.stack(done_white) if len(done_white) > 0 else torch.empty((0, len(envs)), dtype=torch.bool)
    done_black = torch.stack(done_black) if len(done_black) > 0 else torch.empty((0, len(envs)), dtype=torch.bool)

    stats = summarize_batch_stats(envs, rewards_white, rewards_black)

    rewards_white_scalar = rewards_white.sum(dim=-1).permute(1, 0)
    rewards_black_scalar = rewards_black.sum(dim=-1).permute(1, 0)

    if rewards_white_scalar.shape != log_probs_white.shape or rewards_white_scalar.shape != done_white.shape:
        raise RuntimeError(
            "Shape mismatch for WHITE. This usually indicates reward/logprob timeline misalignment.\n"
            f"rewards_white_scalar: {tuple(rewards_white_scalar.shape)}\n"
            f"log_probs_white:      {tuple(log_probs_white.shape)}\n"
            f"done_white:           {tuple(done_white.shape)}"
        )

    if rewards_black_scalar.shape != log_probs_black.shape or rewards_black_scalar.shape != done_black.shape:
        raise RuntimeError(
            "Shape mismatch for BLACK. This usually indicates reward/logprob timeline misalignment.\n"
            f"rewards_black_scalar: {tuple(rewards_black_scalar.shape)}\n"
            f"log_probs_black:      {tuple(log_probs_black.shape)}\n"
            f"done_black:           {tuple(done_black.shape)}"
        )

    loss_white, returns_white = compute_policy_loss(
        log_probs=log_probs_white,
        rewards=rewards_white_scalar,
        done=done_white,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
    )
    loss_black, returns_black = compute_policy_loss(
        log_probs=log_probs_black,
        rewards=rewards_black_scalar,
        done=done_black,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
    )

    loss = loss_white + loss_black

    optim.zero_grad()
    loss.backward()
    optim.step()

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

    csv_logger.log(row)

    tqdm.write(
        f"Batch {batch_nr:04d} | loss={loss.item():.3f} "
        f"(W={loss_white.item():.3f}, B={loss_black.item():.3f}) | "
        f"len={stats['mean_len']:.1f} | "
        f"1-0={stats['w_wins']} 0-1={stats['b_wins']} 1/2-1/2={stats['draws']}"
    )

    logger.info(
        f"batch={batch_nr} loss={loss.item():.6f} loss_w={loss_white.item():.6f} loss_b={loss_black.item():.6f} "
        f"mean_len={stats['mean_len']:.2f} w_wins={stats['w_wins']} b_wins={stats['b_wins']} draws={stats['draws']} "
        f"return_w_mean={returns_white.mean().item():.6f} return_b_mean={returns_black.mean().item():.6f}"
    )

    if save_artifacts_every > 0 and (batch_nr % save_artifacts_every == 0):
        save_rewards_and_games(log_dir, envs, rewards_white, rewards_black, batch_nr)


def train(
    model,
    optim,
    experiment: int,
    batches: int,
    batch_size: int,
    env_params: dict,
    log_dir: Path,
    models_dir: Path,
    gamma: float,
    save_checkpoint_every: int = 10,
    save_artifacts_every: int = 10,
    normalize_advantage: bool = True,
):
    """
    Train the model using self-play and policy gradient updates.
    """
    checkpoints_dir = log_dir / "models"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    envs = [Environment(**env_params) for _ in range(batch_size)]
    logger = logging.getLogger("chess_rl")
    csv_logger = init_csv_logger(log_dir, Rewards.ALL)

    for batch in tqdm(range(batches), desc="Batches", unit="Batches"):
        train_batch(
            model=model,
            optim=optim,
            envs=envs,
            log_dir=log_dir,
            batch_nr=batch,
            gamma=gamma,
            csv_logger=csv_logger,
            logger=logger,
            save_artifacts_every=save_artifacts_every,
            normalize_advantage=normalize_advantage,
        )

        if save_checkpoint_every > 0 and (batch % save_checkpoint_every == 0):
            torch.save(model.state_dict(), checkpoints_dir / f"checkpoint-{batch}.pth")
            tqdm.write("Saved checkpoint")

    games_root = log_dir / "games"
    if games_root.exists():
        ds = [
            xr.open_dataset(entry / "rewards.nc")
            for entry in games_root.iterdir()
            if entry.is_dir() and "batch" in entry.name and (entry / "rewards.nc").exists()
        ]
        if len(ds) > 0:
            ds = xr.concat(ds, dim="batch", join="outer")
            ds = ds.assign_coords(batch=range(ds.sizes["batch"]))
            ds.to_netcdf(log_dir / "rewards.nc")
    
    torch.save(model.state_dict(), models_dir / f"trained-{experiment}.pth")
    tqdm.write("Saved final model")


def main(model_path, experiment, batches, batch_size, gamma):
    """
    Entry point: create environment(s), model, optimizer, and start training.
    """
    env_params = {"rewards": Rewards.ALL}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path(f"logs/rl/experiment-{experiment}")
    logger = setup_logging(log_dir)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model = ChessCNN().to(device)
    if model_path is not None:
        state = torch.load(models_dir / model_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded model checkpoint: {model_path}")

    optim = torch.optim.Adam(model.parameters())

    train(
        model=model,
        optim=optim,
        experiment=experiment,
        batches=batches,
        batch_size=batch_size,
        env_params=env_params,
        log_dir=log_dir,
        models_dir=models_dir,
        gamma=gamma,
        save_checkpoint_every=10,
        save_artifacts_every=10,
        normalize_advantage=True,
    )

    tqdm.write("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reinforcement learning",
        description="Self-play reinforcement learning for chess",
    )
    parser.add_argument("-b", "--batches", default=1000, type=int)
    parser.add_argument("-g", "--batch_size", default=32, type=int)
    parser.add_argument("-n", "--experiment-name", default=0)
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("--gamma", default=0.997, type=float)

    args = parser.parse_args()

    main(
        experiment=args.experiment_name,
        batches=args.batches,
        batch_size=args.batch_size,
        model_path=args.model,
        gamma=args.gamma,
    )
