import torch 
import numpy as np
import chess
from torch.distributions import Categorical

def tensor_from_position(board):
    '''
    board position to tensor that can be used as nn input

    Parameters: 
    - board: current board position 

    Returns: 
    - position encoding: torch tensor of shape (8, 8, 12) = rank, file, piece
        all values are flipped when it's black to play
    '''
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = (piece.piece_type - 1) + (6 if piece.color == board.turn else 0)
        tensor[row, col, idx] = 1

    return torch.from_numpy(tensor)



def move_mask(board): 
    '''
    possible moves to mask nn output

    Parameters: 
    - board: chessboard of current position 

    Return: 
    - legal move mask: torch tensor of shape (64,64,5) = from-to-promotion
    '''
    num_promotions = 5  # queen, rook, bishop, knight + None
    mask = np.zeros((64, 64, num_promotions), dtype=np.bool_)

    promo_map = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}

    for move in board.legal_moves:
        f, t = move.from_square, move.to_square
        mask[f, t, promo_map[move.promotion]] = True

    return torch.from_numpy(mask)




def move_distribution(board, pred): 
    '''
    get the masked move distribution

    Parameters: 
    - board: chessboard with current position
    - pred: torch tensor with NN output

    Return: 
    - pred distribution over legal moves: shape (64*64*5)

    '''

    mask    = move_mask(board).flatten()
    masked  = pred.masked_fill(~mask, float('-inf'))
    softmax = torch.softmax(masked, dim=0)
    return softmax



def mask_logits(board, pred): 
    '''
    get the masked move logits

    Parameters: 
    - board: chessboard with current position
    - pred: torch tensor with NN output

    Return: 
    - pred logits of legal moves: shape (64*64*5)

    '''
    mask    = move_mask(board).flatten()
    masked  = pred.masked_fill(~mask, float('-inf'))
    return masked



def sample_move(board, pred):
    '''
    sample a move from legal move distribution

    Parameters: 
    - board: chessboard with current position
    - pred: torch tensor with NN output

    Return: 
    - move in UCI notation (from square to square)
    - Categorial log distribution: 
    '''

    logits = mask_logits(board, pred)
    m      = Categorical(logits=logits)
    idx    = m.sample()
    move   = torch.unravel_index(idx, (64,64,5))
    return chess.Move(*move), m.log_prob(idx)
    
