"""
Position Sampler for different types of chess positions
Supports: standard (starting position), random positions, endgames, and positions from file
"""

import random
import chess
from pathlib import Path
from abc import ABC, abstractmethod


class PositionSampler(ABC):
    """Abstract base class for position samplers"""
    
    @abstractmethod
    def sample(self) -> str:
        """Return a FEN string for a position"""
        pass


class StandardPositionSampler(PositionSampler):
    """Always returns the standard starting position"""
    
    def sample(self) -> str:
        return chess.STARTING_FEN


class EndgamePositionSampler(PositionSampler):
    """Returns endgame positions (few pieces remaining) by random play"""
    
    def __init__(self, max_pieces=8):
        """
        Parameters:
            max_pieces: target number of pieces (including kings) for endgame
        """
        self.max_pieces = max_pieces
    
    def sample(self) -> str:
        board = chess.Board()
        while True:
            # Count pieces (excluding kings)
            piece_count = sum(1 for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).piece_type != chess.KING)
            if piece_count <= self.max_pieces:
                return board.fen()
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return board.fen()
            
            board.push(random.choice(legal_moves))


class FilePositionSampler(PositionSampler):
    """Loads positions from a file (one FEN per line)"""
    
    def __init__(self, file_path):
        """
        Parameters:
            file_path: path to file containing FEN strings (one per line)
        """
        self.positions = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Position file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            for line in f:
                fen = line.strip()
                # Skip empty lines and comments
                if fen and not fen.startswith('#'):
                    self.positions.append(fen)
        
        if not self.positions:
            raise ValueError(f"No valid FEN positions found in {file_path}")
    
    def sample(self) -> str:
        return random.choice(self.positions)


def get_position_sampler(position_type='standard', **kwargs) -> PositionSampler:
    """
    Factory function to create a position sampler
    
    Parameters:
        position_type: 'standard', 'endgame', or 'file'
        **kwargs: additional arguments passed to sampler (e.g., file_path for 'file')
    
    Returns:
        PositionSampler instance
    """
    if position_type == 'standard':
        return StandardPositionSampler()
    elif position_type == 'endgame':
        return EndgamePositionSampler(max_pieces=kwargs.get('max_pieces', 8))
    elif position_type == 'file':
        return FilePositionSampler(kwargs.get('file_path'))
    else:
        raise ValueError(f"Unknown position type: {position_type}")
