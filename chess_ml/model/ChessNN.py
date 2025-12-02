import torch 
from typing import Sequence, Tuple
from torch import nn, Tensor
from torch.distributions import Categorical
from chess import BLACK, WHITE, Board, Move, square_file, square_rank
from collections.abc import Iterable
import math

from torch.types import Device



class ChessNN(nn.Module): 
    input_shape = (12, 8, 8)
    input_size  = math.prod(input_shape)

    output_shape = (64, 64)
    output_size = math.prod(output_shape)

    def __init__(self):
        '''Abstract base class for chess neural network. 

        Handles transformation from board to input tensor `boards_to_tensor()` 
        and from output tensor to legal move `tensor_to_move_distribution()`. 
        The `predict()` function takes a `Board` and returns a sampled legal `Move`. 

            import chess 
            from chess_ml.model import ChessFeedForward

            board     = chess.Board()
            model     = ChessFeedForward()
            move, log = model.predict(board)
            board.push(move)

        '''
        super().__init__()


    def predict(self, boards: Sequence[Board]|Board) -> Tuple[Sequence[Move], Tensor]: 
        '''Wrapper for forward which parses Board position and 
        returns legal move distribution and sampled move. 

        Handles parsing the position into a tensor that the model can work with. 
        Then forward is applied to predict a move. 
        A move distribution is generated from the model output and the legal moves in the position. 
        A move is sampled from this distribution and `predict` returns 
        a legal move and the distribution it was sampled from. 

        Parameters: 
            board: current position. Expects white to play 
            gamma: probability to explore and sample from uniform distribution
        Returns: 
            samples move: move sampled from output distribution. The move is guaranteed to be legal
            prob_dist: log probabilies of model output masked by move legality
        '''

        if isinstance(boards, Board):
            boards = [boards] 

        if BLACK in [b.turn for b in boards]:
            raise ValueError("Invalid Parameter: expects white to play")


        device  = next(self.parameters()).device
        input   = self.boards_to_tensor(boards).to(device)
        output  = self(input)
        distr   = self.tensor_to_move_distribution(output, boards, device)
        actions = distr.sample()

        move_idx = torch.unravel_index(actions, ChessNN.output_shape)
        moves    = [Move(*idx) for idx in zip(*move_idx)]
        log_prob = distr.log_prob(actions)
        return moves, log_prob


    @staticmethod
    def boards_to_tensor(boards: Sequence[Board]): 
        '''Transforms board into input tensor.  

        Parameters: 
            board (chess.Board): expects white to play
        Returns: 
            input tensor (torch.tensor): one hot encoding
        '''
        
        # indice list where pieces exist
        idx = [(b,
                (piece.piece_type - 1) + (6 if piece.color else 0),
                square_file(square), 
                square_rank(square)) 
                for b, board in enumerate(boards)
                for square,piece in board.piece_map().items()
        ]

        # create tensor
        t                   = torch.zeros(len(boards), *ChessNN.input_shape)
        t[tuple(zip(*idx))] = 1
        return t


    @staticmethod
    def tensor_to_move_distribution(tensor: Tensor, boards: Iterable[Board], device: Device) -> Categorical: 
        '''Transforms model output to legal move distribution. 

        Parameters: 
            tensor: model output tensor. 
            board: current position. Used to mask out illegal moves
        Returns: 
            move distribution: legal move distribution  
        '''
        mask   = torch.stack([ChessNN.move_mask(board) for board in boards]).to(device)
        logits = tensor.masked_fill(~mask, float('-inf'))
        distr  = Categorical(logits=logits)

        return distr


    @staticmethod
    def move_mask(board: Board) -> Tensor:
        '''Returns move mask for legal moves.

        Parameters: 
            board: current position
        Returns: 
            move mask: mask with 1 for legal move and 0 for illegal move 
        '''
        moves = [(m.from_square,m.to_square) for m in board.legal_moves]
        idx   = tuple(zip(*moves))
        mask      = torch.zeros(ChessNN.output_shape, dtype=torch.bool)
        mask[idx] = 1
        mask      = mask.flatten()

        return mask 


    @staticmethod
    def fen_to_tensor(fen): 
        '''
        transforms fen or fen iterable into batch 
        '''
        if isinstance(fen, str) or not isinstance(fen, Iterable):
            fen = [fen] 

        boards = [Board(f) for f in fen]
        tensor = ChessNN.boards_to_tensor(boards)
        return tensor


    @staticmethod
    def fen_to_mask(fen): 
        '''
        transforms fen or fen iterable into batch 
        '''
        if isinstance(fen, str) or not isinstance(fen, Iterable):
            fen = [fen] 

        tensors = [ChessNN.move_mask(Board(f)) for f in fen]
        return torch.stack(tensors, dim=0)


    @staticmethod
    def move_to_one_hot(moves): 
        '''
        generate labels from move (uci) iterable
        '''
        moves = [Move.from_uci(m) for m in moves]
        moves = [(i, m.from_square,m.to_square) for i, m in enumerate(moves)]
        idx   = tuple(zip(*moves))

        labels      = torch.zeros((len(moves), *ChessNN.output_shape))
        labels[idx] = 1
        labels      = labels.flatten(start_dim=1)
        return labels


    @staticmethod
    def move_to_labels(moves): 
        '''
        generate labels from move (uci) iterable
        '''
        return ChessNN.move_to_one_hot(moves).argmax(dim=1)

