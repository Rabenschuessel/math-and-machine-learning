import textwrap
import argparse
import logging
from pathlib import Path 
from chess_ml.env import Rewards
from chess_ml.env.Environment import Environment
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
from multiprocessing import Pool
import numpy as np
import gc
import math


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, board, parent=None, cpuct=1.0):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.cpuct = cpuct
        
    def ucb_score(self):
        """Calculate Upper Confidence Bound score for exploration"""
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / self.visit_count
        exploration = self.cpuct * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration
    
    def select_child(self):
        """Select child with highest UCB score"""
        return max(self.children.values(), key=lambda x: x.ucb_score())
    
    def expand(self, model, device):
        """Expand node by creating children for all legal moves"""
        if self.board.is_game_over():
            return None
        
        legal_moves = list(self.board.legal_moves)
        
        # Create children for each legal move
        for move in legal_moves:
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(child_board, parent=self, cpuct=self.cpuct)
        
        return self.select_child() if self.children else None
    
    def backup(self, value):
        """Backpropagate the value up the tree"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # Flip value for next player
            value = -value
            node = node.parent
    
    @staticmethod
    def board_to_tensor(board, device):
        """Convert chess board to tensor representation (12 planes for all pieces)"""
        board_state = torch.zeros((1, 12, 8, 8), device=device, dtype=torch.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                # Piece index: 0-5 = white pieces (pawn, knight, bishop, rook, queen, king)
                #             6-11 = black pieces
                piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                board_state[0, piece_idx, rank, file] = 1.0
        return board_state


class MonteCarlo:
    """Monte Carlo Tree Search engine"""
    
    def __init__(self, root, model, device, num_simulations=100, cpuct=1.0):
        self.root = root
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.cpuct = cpuct
    
    def simulate(self):
        """Run one MCTS simulation"""
        node = self.root
        
        # Selection & Expansion
        while not node.board.is_game_over() and len(node.children) > 0:
            node = node.select_child()
        
        if not node.board.is_game_over() and node.visit_count > 0:
            node = node.expand(self.model, self.device)
            if node is None:
                return
        
        # Evaluation
        value = self._evaluate_position(node.board)
        
        # Backup
        node.backup(value)
    
    def _evaluate_position(self, board):
        """Evaluate a position: 1 if current player can win, -1 if losing, 0 if draw"""
        if board.is_checkmate():
            return 1.0  # Current player loses (checkmate is bad)
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0  # Draw
        
        # Use neural network for position evaluation
        with torch.no_grad():
            board_tensor = MCTSNode.board_to_tensor(board, self.device)
            try:
                # Try to get value prediction (adapt to your model interface)
                if hasattr(self.model, 'predict_value'):
                    value_pred = self.model.predict_value(board_tensor)
                else:
                    # Fallback: use first output of model
                    output = self.model(board_tensor)
                    if isinstance(output, tuple):
                        value_pred = output[1]  # Assume second output is value
                    else:
                        value_pred = output
                return value_pred.item()
            except:
                # Fallback to random evaluation
                return np.random.uniform(-1, 1)
    
    def get_policy_distribution(self, temperature=1.0):
        """Get policy distribution over root's children based on visit counts"""
        if not self.root.children:
            return None
        
        moves = list(self.root.children.keys())
        visits = np.array([self.root.children[m].visit_count for m in moves])
        
        if temperature == 0:
            # Deterministic: select move with most visits
            policy = np.zeros(len(moves), dtype=np.float32)
            policy[np.argmax(visits)] = 1.0
        else:
            # Stochastic: visits raised to 1/temperature
            visits_weighted = np.power(visits.astype(np.float32), 1.0 / temperature)
            policy = visits_weighted / visits_weighted.sum()
        
        return moves, policy
    
    def select_move(self, temperature=1.0):
        """Select best move based on MCTS statistics"""
        result = self.get_policy_distribution(temperature)
        if result is None:
            return None, None
        moves, policy = result
        best_idx = np.argmax(policy)
        return moves[best_idx], policy


def perform_mcts_episodes(args):
    """Run MCTS episodes in parallel"""
    (episodes, model_path, architecture, env_params, sim_steps, cpuct, sample_ratio, thread) = args
    
    # Setup device for this process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if architecture == 'linear': 
        model = ChessFeedForward()
    elif architecture == 'cnn': 
        model = ChessCNN()
    else: 
        model = ChessResBlock()
    
    if model_path is not None and Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    
    model = model.to(device)
    model.eval()
    
    # Buffers for training data
    state_buffer = []
    policy_buffer = []
    game_outcomes = []
    
    for episode in range(episodes):
        try:
            env = Environment(**env_params)
            board = env.reset()
            
            root = MCTSNode(board, parent=None, cpuct=cpuct)
            mcts = MonteCarlo(root, model, device, num_simulations=sim_steps, cpuct=cpuct)
            
            states = []
            policies = []
            
            # Play one game with MCTS
            ply = 0
            while not env._board.is_game_over() and ply < 200:  # Limit moves
                # Run MCTS simulations
                for _ in range(sim_steps):
                    mcts.simulate()
                
                # Select move
                move, policy = mcts.select_move(temperature=1.0 if np.random.random() < sample_ratio else 0.0)
                
                if move is None or policy is None:
                    break
                
                # Record state and policy before move
                states.append(env._board.copy())
                policies.append(policy)
                
                # Make move
                env.step(move)
                
                # Move root in MCTS tree
                if move in mcts.root.children:
                    mcts.root = mcts.root.children[move]
                    mcts.root.parent = None  # Detach from tree
                else:
                    mcts.root = MCTSNode(env._board, parent=None, cpuct=cpuct)
                
                ply += 1
                gc.collect()
            
            # Determine outcome
            result = env._board.result()
            if result == "1-0":
                outcome = 1.0  # White won
            elif result == "0-1":
                outcome = -1.0  # Black won
            else:
                outcome = 0.0  # Draw or unfinished
            
            # Store trajectories (from white's perspective)
            for i, (board_state, policy) in enumerate(zip(states, policies)):
                state_buffer.append(board_state)
                policy_buffer.append(policy)
                # Flip outcome every other move (alternating colors)
                move_outcome = outcome if i % 2 == 0 else -outcome
            
            game_outcomes.append(outcome)
            
            if (episode + 1) % max(1, episodes // 5) == 0:
                tqdm.write(f"Thread {thread}: Episode {episode + 1}/{episodes}")
        
        except Exception as e:
            tqdm.write(f"Thread {thread}: Error in episode {episode}: {str(e)}")
            continue
    
    return state_buffer, policy_buffer, game_outcomes


def train_mcts(model, optim, epochs, env_params, log_dir, num_threads=4, 
               episodes_per_thread=10, sim_steps=100, cpuct=1.0, sample_ratio=0.8, 
               architecture='resnet', batch_size=32):
    """Train with MCTS self-play"""
    models_dir = Path(log_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(epochs):
        tqdm.write(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        
        # Save model to checkpoint for workers
        model_path = models_dir / f"temp_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        
        # Collect data from parallel MCTS episodes
        state_buffer = []
        policy_buffer = []
        outcomes = []
        
        with Pool(num_threads) as p:
            results = p.map(perform_mcts_episodes, [
                (
                    episodes_per_thread,
                    str(model_path),
                    architecture,
                    env_params,
                    sim_steps,
                    cpuct,
                    sample_ratio,
                    thread
                ) for thread in range(num_threads)
            ])
        
        # Aggregate results
        for states, policies, game_outcomes in results:
            state_buffer.extend(states)
            policy_buffer.extend(policies)
            outcomes.extend(game_outcomes)
        
        if not state_buffer:
            tqdm.write("No samples collected, skipping training")
            continue
        
        tqdm.write(f"Collected {len(state_buffer)} training samples from {len(outcomes)} games")
        
        # Convert boards to tensors and train
        board_tensors = []
        for board in state_buffer:
            board_tensor = MCTSNode.board_to_tensor(board, device)
            board_tensors.append(board_tensor)
        
        board_tensors = torch.cat(board_tensors, dim=0)
        policy_targets = torch.tensor(np.array(policy_buffer), device=device, dtype=torch.float32)
        
        # Training loop
        model = model.to(device)
        num_batches = len(state_buffer) // batch_size
        total_loss = 0
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(state_buffer))
            
            board_batch = board_tensors[start:end]
            policy_batch = policy_targets[start:end]
            
            # Forward pass and loss depends on your model architecture
            try:
                output = model(board_batch)
                if isinstance(output, tuple):
                    policy_pred = output[0]
                else:
                    policy_pred = output
                
                # Adapt loss based on your policy representation
                if policy_pred.shape[-1] == policy_batch.shape[-1]:
                    loss = torch.nn.functional.mse_loss(policy_pred, policy_batch)
                else:
                    # If policy_pred is logits for specific moves
                    loss = torch.nn.functional.cross_entropy(policy_pred, policy_batch)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
            except Exception as e:
                tqdm.write(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            tqdm.write(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Log outcomes
        outcomes_array = np.array(outcomes)
        wins = (outcomes_array == 1).sum()
        draws = (outcomes_array == 0).sum()
        losses = (outcomes_array == -1).sum()
        tqdm.write(f"Results: W-D-L {wins}-{draws}-{losses}")
        
        # Checkpoint
        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), models_dir / f"checkpoint-{epoch}.pth")
            tqdm.write(f"Saved checkpoint at epoch {epoch + 1}")


def main_mcts(*, model_path, experiment, architecture, epochs, num_threads, 
              episodes_per_thread, sim_steps, cpuct, sample_ratio, rewards):
    """Main function for MCTS training"""
    name2reward = {r.__name__: r for r in Rewards.ALL}
    if rewards:
        env_params = {"rewards": [name2reward[r] for r in rewards]}
    else:
        env_params = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create log directory
    log_dir = Path("logs/rl/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted([int(x.name) for x in log_dir.iterdir() if x.is_dir()])
    new = 0 if len(experiments) == 0 else (int(experiments[-1]) + 1)
    log_dir = log_dir / "{:03d}".format(new)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Save hyperparameters
    with open(log_dir / 'hparams.txt', 'w') as f:
        f.write(textwrap.dedent(f"""\
                model_path: {model_path}
                architecture: {architecture}
                epochs: {epochs}
                num_threads: {num_threads}
                episodes_per_thread: {episodes_per_thread}
                sim_steps: {sim_steps}
                cpuct: {cpuct}
                sample_ratio: {sample_ratio}
                rewards: {rewards}"""))
    
    logging.info('Loading model architecture')
    if architecture == 'linear':
        model = ChessFeedForward()
    elif architecture == 'cnn':
        model = ChessCNN()
    else:
        model = ChessResBlock()
    
    if model_path is not None:
        logging.info('Loading model weights')
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    
    model = model.to(device)
    logging.info('Creating optimizer')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    logging.info('Starting MCTS training')
    train_mcts(model, optim, epochs, env_params, log_dir, num_threads=num_threads,
               episodes_per_thread=episodes_per_thread, sim_steps=sim_steps, 
               cpuct=cpuct, sample_ratio=sample_ratio, architecture=architecture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Chess MCTS RL",
        description="Train chess model with Monte Carlo Tree Search"
    )
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-t', '--threads', default=4, type=int, help='Number of parallel threads')
    parser.add_argument('-ep', '--episodes-per-thread', default=10, type=int, help='Episodes per thread')
    parser.add_argument('-s', '--sim-steps', default=100, type=int, help='MCTS simulations per move')
    parser.add_argument('-c', '--cpuct', default=1.0, type=float, help='CPUCT exploration parameter')
    parser.add_argument('-sr', '--sample-ratio', default=0.8, type=float, help='Ratio of stochastic move selection')
    parser.add_argument('-n', '--experiment-name', default='mcts', help='Experiment name')
    parser.add_argument('-m', '--model', default=None, help='Path to pretrained model')
    parser.add_argument('-a', '--architecture', choices=['linear', 'cnn', 'resnet'], default='resnet')
    parser.add_argument('-r', '--rewards', choices=[r.__name__ for r in Rewards.ALL], nargs="*")
    
    args = parser.parse_args()
    
    main_mcts(
        model_path=args.model,
        experiment=args.experiment_name,
        architecture=args.architecture,
        epochs=args.epochs,
        num_threads=args.threads,
        episodes_per_thread=args.episodes_per_thread,
        sim_steps=args.sim_steps,
        cpuct=args.cpuct,
        sample_ratio=args.sample_ratio,
        rewards=args.rewards if args.rewards else []
    )
