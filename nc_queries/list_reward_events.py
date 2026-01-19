#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def iter_batch_dirs(games_root: Path):
    """
    Yield batch directories in sorted order: batch-0000, batch-0001, ...
    """
    for d in sorted(games_root.glob("batch-*")):
        if d.is_dir():
            yield d


def color_label(c):
    # In your saved rewards.nc you used chess.BLACK / chess.WHITE (0/1).
    return "black" if int(c) == 0 else "white" if int(c) == 1 else f"color_{int(c)}"


def load_nonzero_events(batch_rewards_nc: Path, reward_filter: list[str] | None) -> pd.DataFrame:
    """
    Load a single batch rewards.nc and return a DataFrame with one row per non-zero reward event:
        batch_dir, reward, color, game, turn, value
    """
    ds = xr.open_dataset(batch_rewards_nc)

    reward_vars = list(ds.data_vars.keys())
    if not reward_vars:
        return pd.DataFrame(columns=["batch", "reward", "color", "game", "turn", "value"])

    if reward_filter is not None:
        missing = [r for r in reward_filter if r not in reward_vars]
        if missing:
            raise ValueError(
                f"Rewards not found in {batch_rewards_nc}: {missing}. Available: {reward_vars}"
            )
        reward_vars = reward_filter

    # Expect dims: color, game, turn
    for dim in ("color", "game", "turn"):
        if dim not in ds.dims and dim not in ds.coords:
            raise RuntimeError(
                f"{batch_rewards_nc}: expected '{dim}' in dataset. dims={list(ds.dims)} coords={list(ds.coords)}"
            )

    batch_name = batch_rewards_nc.parent.name  # batch-XXXX

    rows = []
    colors = ds["color"].values

    for rv in reward_vars:
        da = ds[rv].transpose("color", "game", "turn")  # stable order
        arr = da.values  # [C, G, T]

        # Find all non-zero entries
        nz = np.nonzero(arr)  # tuple of arrays (c_idx, g_idx, t_idx)
        if len(nz[0]) == 0:
            continue

        c_idx, g_idx, t_idx = nz
        vals = arr[c_idx, g_idx, t_idx]

        for i in range(len(vals)):
            c_val = colors[int(c_idx[i])]
            rows.append(
                {
                    "batch": batch_name,
                    "reward": rv,
                    "color": color_label(c_val),
                    "color_id": int(c_val),
                    "game": int(g_idx[i]),
                    "turn": int(t_idx[i]),
                    "value": float(vals[i]),
                }
            )

    return pd.DataFrame(rows)


def main(
    games_root: Path,
    rewards: list[str] | None,
    out_csv: Path | None,
    show_limit: int | None,
    group_by_game: bool,
):
    if not games_root.exists():
        raise FileNotFoundError(f"games_root not found: {games_root}")

    all_rows = []
    for batch_dir in iter_batch_dirs(games_root):
        rewards_nc = batch_dir / "rewards.nc"
        if not rewards_nc.exists():
            continue
        df = load_nonzero_events(rewards_nc, rewards)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No non-zero reward events found.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # Sort nicely
    df_all = df_all.sort_values(["batch", "game", "turn", "color_id", "reward"]).reset_index(drop=True)

    # Optional: save to CSV
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")

    # Display
    if group_by_game:
        # Print per game as a block (more readable)
        # NOTE: this can be huge if you have many games; use --limit if needed
        grouped = df_all.groupby(["batch", "game"], sort=False)
        printed = 0
        for (b, g), sub in grouped:
            print(f"\n=== {b} | game {g} ===")
            print(sub[["turn", "color", "reward", "value"]].to_string(index=False))
            printed += len(sub)
            if show_limit is not None and printed >= show_limit:
                print(f"\n[Stopped after printing ~{show_limit} rows due to --limit]")
                break
    else:
        # Print as one table
        if show_limit is not None:
            print(df_all.head(show_limit).to_string(index=False))
            print(f"\n[Showing first {show_limit} rows. Use --group-by-game for nicer per-game view.]")
        else:
            print(df_all.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="List ALL turns with non-zero rewards across all batches (database view)."
    )
    p.add_argument(
        "games_root",
        type=Path,
        help="Path to games root, e.g. logs/rl/experiment-0/games",
    )
    p.add_argument(
        "--rewards",
        nargs="*",
        default=None,
        help="Optional: only include these reward variable names (e.g. JUST_WIN). Default: all rewards in file.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional: write all events to a CSV file.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit how many rows are printed (printing everything can be huge).",
    )
    p.add_argument(
        "--group-by-game",
        action="store_true",
        help="Print in blocks per (batch, game) instead of one big table.",
    )

    args = p.parse_args()

    main(
        games_root=args.games_root,
        rewards=args.rewards,
        out_csv=args.out_csv,
        show_limit=args.limit,
        group_by_game=args.group_by_game,
    )
