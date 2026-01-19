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
    Configure a dedicated file logger for this run.

    Why this is explicit (instead of logging.basicConfig):
    - basicConfig is a no-op if logging was configured earlier (common in notebooks / imports).
    - We want deterministic behavior: always write to log_dir/log.log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.log"

    logger = logging.getLogger("chess_rl")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate logs via root logger

    # Remove existing handlers to avoid duplicated lines if the script is re-run
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)

    logger.addHandler(fh)
    return logger


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


def save_rewards_and_games(
    log_dir: Path,
    envs,
    rewards_white: torch.Tensor,
    rewards_black: torch.Tensor,
    batch_nr: int,
):
    """
    Save per-component rewards (NetCDF) and sample games (PGN) for later inspection.

    Expected shapes:
        rewards_white: [B, T, K]
        rewards_black: [B, T, K]
    """
    games_dir = Path(log_dir) / "games" / f"batch-{batch_nr:04d}"
    games_dir.mkdir(parents=True, exist_ok=True)

    reward_fns = envs[0]._rewards

    ds = xr.concat(
        [
            xr.Dataset(
                data_vars={
                    r.__name__: (["game", "turn"], t.detach().cpu().numpy()[:, :, i])
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


def summarize_batch_stats(
    envs,
    rewards_white: torch.Tensor,
    rewards_black: torch.Tensor,
):
    """
    Compute batch-level statistics for logging.

    Parameters
    ----------
    envs : list[Environment]
        Environments after finished self-play games.
    rewards_white : torch.Tensor
        Tensor of shape [B, T_w, K] containing white rewards
        (padded with zeros where games are shorter).
    rewards_black : torch.Tensor
        Tensor of shape [B, T_b, K] containing black rewards
        (padded with zeros where games are shorter).

    Returns
    -------
    dict
        Dictionary with reward means/stds and game outcome statistics.
    """

    # Reward component names (order must match reward tensors)
    reward_names = [r.__name__ for r in envs[0]._rewards]

    # Aggregate rewards over batch and time
    # Shape: [K]
    white_mean = rewards_white.mean(dim=(0, 1)).detach().cpu().numpy()
    black_mean = rewards_black.mean(dim=(0, 1)).detach().cpu().numpy()
    white_std = rewards_white.std(dim=(0, 1)).detach().cpu().numpy()
    black_std = rewards_black.std(dim=(0, 1)).detach().cpu().numpy()

    # Game outcomes
    results = [env._board.result() for env in envs]
    cnt = Counter(results)

    # Actual game lengths (number of half-moves played)
    lengths = [len(env._board.move_stack) for env in envs]
    mean_len = float(sum(lengths) / max(1, len(lengths)))

    return {
        "reward_names": reward_names,
        "white_mean": white_mean,
        "black_mean": black_mean,
        "white_std": white_std,
        "black_std": black_std,
        "w_wins": cnt.get("1-0", 0),
        "b_wins": cnt.get("0-1", 0),
        "draws": cnt.get("1/2-1/2", 0),
        "mean_len": mean_len,
    }


def compute_policy_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    normalize_advantage: bool = True,
):
    """
    Compute a REINFORCE-style policy gradient loss using reward-to-go.

    Key contracts:
    - `done` is the terminal flag AFTER the environment step (needed by reward2go).
    - `valid` indicates whether the action at this timestep is real
      (env was NOT done BEFORE the step). This keeps the terminal move valid
      and masks out phantom timesteps after termination.

    Shapes:
        log_probs: [T, B]
        rewards:   [T, B]
        done:      [T, B]  (done AFTER the step)
        valid:     [T, B]  (1.0 if env active BEFORE the step, else 0.0)
    """
    valid = valid.float()
    valid_count = valid.sum()

    # If there are no valid timesteps, return a clean zero loss
    if valid_count.item() == 0:
        returns = reward2go(rewards, done, gamma)
        zero = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
        return zero, returns

    # Defensive: mask everything to guarantee invalid timesteps are inert
    log_probs = log_probs * valid
    rewards = rewards * valid

    # reward2go needs "done after step" to stop return accumulation at terminals
    returns = reward2go(rewards, done, gamma)

    if normalize_advantage:
        # Normalize only across valid timesteps to avoid padding/terminal distortion
        valid_idx = valid.bool()
        valid_returns = returns[valid_idx]

        mean = valid_returns.mean()
        std = valid_returns.std(unbiased=False).clamp_min(1e-8)

        adv = torch.zeros_like(returns)
        adv[valid_idx] = (valid_returns - mean) / std
    else:
        adv = returns * valid

    loss = (-adv * log_probs).sum() / (valid_count + 1e-8)
    return loss, returns


def _pad_time(t: torch.Tensor, target_T: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad or truncate a [T, K] tensor along time dimension to target_T.

    This is required because games have different lengths, but we need a
    rectangular batch tensor for training.
    """
    if t.dim() != 2:
        raise ValueError(f"Expected [T, K] tensor, got shape {tuple(t.shape)}")

    T, K = t.shape
    if T == target_T:
        return t
    if T > target_T:
        return t[:target_T, :]
    pad_len = target_T - T
    pad = torch.full((pad_len, K), pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)


def _stack_rewards_from_envs(envs, device: torch.device, target_T_white: int, target_T_black: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect rewards from envs and return padded tensors on the requested device.

    Expected env.get_rewards() output per env:
        (white_rewards, black_rewards) each shaped [T, K] where T may differ per env.

    Returns:
        rewards_white: [B, target_T_white, K]
        rewards_black: [B, target_T_black, K]
    """
    rw_list = []
    rb_list = []

    for env in envs:
        rw, rb = env.get_rewards()

        rw_t = torch.as_tensor(rw, dtype=torch.float32, device=device)
        rb_t = torch.as_tensor(rb, dtype=torch.float32, device=device)

        rw_t = _pad_time(rw_t, target_T_white, pad_value=0.0)
        rb_t = _pad_time(rb_t, target_T_black, pad_value=0.0)

        rw_list.append(rw_t)
        rb_list.append(rb_t)

    rewards_white = torch.stack(rw_list, dim=0)  # [B, T, K]
    rewards_black = torch.stack(rb_list, dim=0)  # [B, T, K]
    return rewards_white, rewards_black


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
    Run one batch of self-play games, compute policy gradient loss, update the model,
    and log metrics and artifacts.

    Correctness properties:
    - Never call env.step() on environments that are already done.
    - The terminal (game-ending) move is INCLUDED in the gradient update.
    - No phantom gradients from timesteps after termination.
    - reward2go receives 'done AFTER step', while masking uses 'done BEFORE step'.
    """
    model_device = next(model.parameters()).device
    color = chess.WHITE

    log_probs_white: list[torch.Tensor] = []
    done_white: list[torch.Tensor] = []
    valid_white: list[torch.Tensor] = []

    log_probs_black: list[torch.Tensor] = []
    done_black: list[torch.Tensor] = []
    valid_black: list[torch.Tensor] = []

    boards = [env.reset() for env in envs]
    done = [False] * len(envs)

    with tqdm(total=len(envs), desc="Games", unit="Games") as pbar:
        prev_done_count = 0

        while not all(done):
            moves, log_probs = model.predict(boards)

            if not isinstance(log_probs, torch.Tensor):
                log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=model_device)
            else:
                log_probs = log_probs.to(model_device)

            # Snapshot: which envs were already terminated BEFORE executing this step?
            done_prev = torch.tensor(done, dtype=torch.bool, device=model_device)

            new_boards = list(boards)
            new_done = list(done)

            # Step only active environments
            for i, (env, mv) in enumerate(zip(envs, moves)):
                if not done[i]:
                    board_next, is_done = env.step(mv)
                    new_boards[i] = board_next
                    new_done[i] = bool(is_done)

            boards = new_boards
            done = new_done

            # Done flags AFTER the step (this is what reward2go expects)
            done_after = torch.tensor(done, dtype=torch.bool, device=model_device)

            # Valid mask: 1.0 if action was taken this timestep, else 0.0
            valid_mask = (~done_prev).float()

            # Debug sanity: valid_mask should be exactly the inverse of done_prev
            if not torch.equal(valid_mask.bool(), (~done_prev)):
                raise RuntimeError("valid_mask / done_prev inconsistency detected.")

            log_probs_masked = log_probs * valid_mask

            if color == chess.WHITE:
                log_probs_white.append(log_probs_masked)
                done_white.append(done_after)
                valid_white.append(valid_mask)
                color = chess.BLACK
            else:
                log_probs_black.append(log_probs_masked)
                done_black.append(done_after)
                valid_black.append(valid_mask)
                color = chess.WHITE

            done_count = sum(done)
            pbar.update(done_count - prev_done_count)
            prev_done_count = done_count

    B = len(envs)

    log_probs_white_t = torch.stack(log_probs_white, dim=0) if log_probs_white else torch.empty((0, B), device=model_device)
    log_probs_black_t = torch.stack(log_probs_black, dim=0) if log_probs_black else torch.empty((0, B), device=model_device)

    done_white_t = torch.stack(done_white, dim=0) if done_white else torch.empty((0, B), dtype=torch.bool, device=model_device)
    done_black_t = torch.stack(done_black, dim=0) if done_black else torch.empty((0, B), dtype=torch.bool, device=model_device)

    valid_white_t = torch.stack(valid_white, dim=0) if valid_white else torch.empty((0, B), device=model_device)
    valid_black_t = torch.stack(valid_black, dim=0) if valid_black else torch.empty((0, B), device=model_device)

    target_T_white = int(log_probs_white_t.shape[0])
    target_T_black = int(log_probs_black_t.shape[0])

    rewards_white, rewards_black = _stack_rewards_from_envs(
        envs,
        device=model_device,
        target_T_white=target_T_white,
        target_T_black=target_T_black,
    )

    stats = summarize_batch_stats(envs, rewards_white, rewards_black)

    rewards_white_scalar = rewards_white.sum(dim=-1).permute(1, 0)
    rewards_black_scalar = rewards_black.sum(dim=-1).permute(1, 0)

    if rewards_white_scalar.shape != log_probs_white_t.shape or rewards_white_scalar.shape != done_white_t.shape:
        raise RuntimeError(
            "Shape mismatch for WHITE. Reward/logprob timeline misalignment.\n"
            f"rewards_white_scalar: {tuple(rewards_white_scalar.shape)}\n"
            f"log_probs_white:      {tuple(log_probs_white_t.shape)}\n"
            f"done_white:           {tuple(done_white_t.shape)}"
        )
    if rewards_black_scalar.shape != log_probs_black_t.shape or rewards_black_scalar.shape != done_black_t.shape:
        raise RuntimeError(
            "Shape mismatch for BLACK. Reward/logprob timeline misalignment.\n"
            f"rewards_black_scalar: {tuple(rewards_black_scalar.shape)}\n"
            f"log_probs_black:      {tuple(log_probs_black_t.shape)}\n"
            f"done_black:           {tuple(done_black_t.shape)}"
        )

    loss_white, returns_white = compute_policy_loss(
        log_probs=log_probs_white_t,
        rewards=rewards_white_scalar,
        done=done_white_t,
        valid=valid_white_t,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
    )
    loss_black, returns_black = compute_policy_loss(
        log_probs=log_probs_black_t,
        rewards=rewards_black_scalar,
        done=done_black_t,
        valid=valid_black_t,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
    )

    loss = loss_white + loss_black

    optim.zero_grad(set_to_none=True)
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
    rewards: list,
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

    env_params = {"rewards": rewards}
    envs = [Environment(**env_params) for _ in range(batch_size)]
    logger = logging.getLogger("chess_rl")
    csv_logger = init_csv_logger(log_dir, rewards)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rewards = Rewards.JUST_WIN

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
        rewards=rewards,
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
    parser.add_argument("-n", "--experiment-name", default=0, type=int)
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
