from torchrl.objectives.value.functional import reward2go
from collections import Counter
import xarray as xr
import torch 
import logging
from pathlib import Path
from chess_ml.logging.csv_logger import CSVLogger
import chess

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

