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
import torch.nn.functional as F
from chess import Move
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.Convolution import ChessCNN
from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.model.ResBlock import ChessResBlock



def log_batch(path, envs, rewards_tensor, batch_nr): 
    # Logging reward values
    tqdm.write("mean reward values: {}".format(rewards_tensor.abs().mean()))

    # Saving rewards
    path = Path(path) / "games" / "batch-{:04d}".format(batch_nr)
    path.mkdir(parents=True, exist_ok=True)

    # rewards_tensor is shape (T, B) - time steps x batch size
    ds = xr.Dataset(
        data_vars={
            'reward': (["turn", "game"], rewards_tensor.cpu().numpy())
        },
        coords=dict(
            game=("game", range(rewards_tensor.shape[1])),
            turn=("turn", range(rewards_tensor.shape[0])),
        )
    )
    ds.to_netcdf(path / "rewards.nc")


    # Saving games as pgn
    for gamenr, env in enumerate(envs): 
        game = env.get_game()
        with open(path / "game-{:06d}.pgn".format(gamenr), "w") as f:
            print(game, file=f)



def train_batch(model, optim, envs, log_dir, batch_nr, gamma, epsilon=0.1):
    color = chess.WHITE
    values_all  = []
    dones_all   = []
    rewards_all = []

    device = next(model.parameters()).device
    boards = [env.reset() for env in envs]
    done   = [False] * len(envs)

    with tqdm(total=len(envs), desc="Games", unit="Games") as pbar: 
        while not all(done): 
            # Pure value-based RL: get state values (with gradients for training)
            values = model.predict_values(boards)
            
            # Epsilon-greedy action selection using successor-state values (vectorized)
            moves = [None] * len(boards)
            # collect successor boards for exploitation decisions
            succ_boards = []
            succ_owner = []  # (game_index)
            succ_moves = []  # corresponding Move

            for i, board in enumerate(boards):
                # If the game is already finished, append dummy move (env.step ignores it)
                if done[i]:
                    moves[i] = Move(0, 0)
                    continue

                legal = list(board.legal_moves)
                if len(legal) == 0:
                    moves[i] = Move(0, 0)
                    continue

                if torch.rand(1).item() < epsilon:
                    # exploration: random legal move
                    moves[i] = legal[torch.randint(0, len(legal), (1,)).item()]
                else:
                    # exploitation: evaluate successor states later in batch
                    for m in legal:
                        nb = board.copy()
                        nb.push(m)
                        # ensure successor is represented with white to play
                        if nb.turn != chess.WHITE:
                            nb = nb.mirror()
                        succ_boards.append(nb)
                        succ_owner.append(i)
                        succ_moves.append(m)

            # evaluate all successor states in a single forward pass
            if len(succ_boards) > 0:
                succ_tensor = model.boards_to_tensor(succ_boards).to(device)
                with torch.no_grad():
                    out = model(succ_tensor)
                    if isinstance(out, (tuple, list)):
                        _, succ_values = out
                    else:
                        succ_values = out

                # succ_values is shape (N, ) or (N) depending on model; ensure 1D tensor
                succ_values = succ_values.view(-1).cpu()

                # pick best successor per game
                from collections import defaultdict
                best_idx = defaultdict(lambda: (None, float('-inf')))
                for idx, owner in enumerate(succ_owner):
                    val = float(succ_values[idx].item())
                    if val > best_idx[owner][1]:
                        best_idx[owner] = (idx, val)

                for owner, (sidx, _) in best_idx.items():
                    moves[owner] = succ_moves[sidx]
            
            boards, done = zip(*[env.step(move) for env, move in zip(envs, moves)])

            values_all.append(values.to(device))
            dones_all.append(torch.tensor(done, device=device))

            pbar.update(sum(done) - pbar.n)

    # transform to torch tensors
    all_rewards = [env.get_rewards() for env in envs]
    device = next(model.parameters()).device
    rewards_tensor = torch.tensor([r for r, _ in all_rewards], device=device)
    
    values_all  = torch.stack(values_all).to(device)
    dones_all   = torch.stack(dones_all).to(device)
    
    # sum reward components and transpose to (T, B)
    rewards_tensor = rewards_tensor.sum(dim=-1).permute(1, 0)
    
    # values_all and rewards_tensor should have same T dimension
    # If values_all is longer, trim it
    T_rewards = rewards_tensor.shape[0]
    if values_all.shape[0] > T_rewards:
        values_all = values_all[:T_rewards]
        dones_all = dones_all[:T_rewards]

    log_batch(log_dir, envs, rewards_tensor, batch_nr)
    #batch td(lambda) 
    # helper to compute lambda-returns (forward view)
    def lambda_returns(rewards, values, dones, gamma, lam):
        T, B = rewards.shape
        G = torch.zeros_like(rewards)
        G_next = torch.zeros(B, device=rewards.device)
        for t in range(T - 1, -1, -1):
            if t + 1 < T:
                V_tp1 = values[t + 1]
            else:
                V_tp1 = torch.zeros_like(G_next)
            done_mask = dones[t].to(torch.bool)
            V_tp1 = V_tp1 * (~done_mask)
            G_next = G_next * (~done_mask)
            G_t = rewards[t] + gamma * ((1 - lam) * V_tp1 + lam * G_next)
            G[t] = G_t
            G_next = G_t
        return G

    lam = getattr(model, "lambda_", 0.8)

    # compute TD(lambda) returns
    returns = lambda_returns(rewards_tensor, values_all, dones_all, gamma, lam)

    # value function loss (MSE to TD(lambda) returns) - pure value-based RL
    loss = F.mse_loss(values_all, returns, reduction='mean')

    # Debug: log value predictions vs returns
    debug_log_path = "debug_value_log.txt"
    with open(debug_log_path, "a") as debug_log:
        debug_log.write(f"[BATCH] Values mean: {values_all.mean().item():.4f}, Returns mean: {returns.mean().item():.4f}\n")
        debug_log.write(f"[BATCH] Values max: {values_all.max().item():.4f}, Returns max: {returns.max().item():.4f}\n")
        debug_log.write(f"[BATCH] Values min: {values_all.min().item():.4f}, Returns min: {returns.min().item():.4f}\n")
        debug_log.write(f"[BATCH] Loss: {loss.item():.4f}\n\n")

    # optimize
    optim.zero_grad()
    loss.backward()
    optim.step()

    tqdm.write("loss: {}".format(loss.item()))
    tqdm.write(str(Counter([env._board.result() for env in envs])))
    return loss.item()



def train(model, optim, batches, batch_size, env_params, log_dir, gamma, epsilon=0.1): 
    import csv
    models_dir = Path(log_dir/"models")
    models_dir.mkdir(parents=True, exist_ok=True)

    envs = [Environment(**env_params) for i in range(batch_size)]

    # Prepare CSV logging (create file and header)
    loss_log_path = Path(log_dir) / "loss_and_results.csv"
    with open(loss_log_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch", "loss", "win", "draw", "loss"])


    for batch in tqdm(range(batches), desc="Batches", unit="Batches"): 
        # Run training batch and get loss
        loss = train_batch(model, optim, envs, log_dir, batch, gamma, epsilon)

        # After train_batch, collect loss and results for logging
        from collections import Counter
        results = [env._board.result() for env in envs]
        win = results.count('1-0')
        draw = results.count('1/2-1/2')
        loss_count = results.count('0-1')
        with open(loss_log_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([batch, loss, win, draw, loss_count])

        if batch % 10 == 0: 
            tqdm.write("Save Checkpoint")
            torch.save(model.state_dict(), models_dir / f"checkpoint-{batch}.pth")

    ds = [xr.open_dataset(entry / 'rewards.nc') 
            for entry in (log_dir/"games").iterdir() 
            if entry.is_dir() and 'batch' in entry.name]
    ds = xr.concat(ds, dim='batch', join='outer')
    ds = ds.assign_coords(batch=range(ds.sizes['batch']))
    ds.to_netcdf(log_dir / "rewards.nc")




def main(model_path, experiment, batches, batch_size, gamma, lam, epsilon=0.1, starting_fen=None):
    env_params = {"rewards": Rewards.ALL}
    if starting_fen:
        env_params["starting_fen"] = starting_fen
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir    = Path("logs/rl/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )

    model = ChessCNN()
    model = model.to(device)
    # attach TD(lambda) hyperparameter to model for easy access
    model.lambda_ = lam
    if model_path is not None: 
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)


    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, optim, batches, batch_size, env_params, log_dir, gamma, epsilon)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reinforcement learning", 
        description="transform chess puzzle dataset")
    parser.add_argument('-b', '--batches' , default=1000, type=int)
    parser.add_argument('-g', '--batch_size' , default=32, type=int)
    parser.add_argument('-n', '--experiment-name', default=0)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--lam', default=0.8, type=float, help='TD(lambda) parameter')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Epsilon-greedy exploration parameter')
    parser.add_argument('--starting_fen', default=None, type=str, help='Custom starting position in FEN')
    args = parser.parse_args()

    main(experiment=args.experiment_name,
         batches=args.batches,
         batch_size=args.batch_size,
        model_path=args.model,
        gamma=args.gamma,
        lam=args.lam,
        epsilon=args.epsilon,
        starting_fen=args.starting_fen)




