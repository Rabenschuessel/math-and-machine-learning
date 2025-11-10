import torch 
from torch import nn, Tensor
from torch.distributions import Categorical
from chess import WHITE, Board, Move
import math

class ChessNN(nn.Module): 
    input_shape = (64, 12)
    input_size  = math.prod(input_shape)

    output_shape = (64, 64)
    output_size = math.prod(output_shape)

    def __init__(self):
        super().__init__()



    def predict(self, board: Board): 
        ''' 
        Wrapper around forward.
        Transforms board into input tensor, 
        and output tensor into move distribution (masked by legal moves). 

        Parameters: 
            board (chess.Board): 

        Returns: 
            samples move (chess.Move): 
            prob_dist (torch.tensor): log probabilies
        '''
        if board.turn is not WHITE: 
            raise ValueError("Invalid Parameter: expects white to play")

        input  = self.board_to_tensor(board)
        output = self(input)
        distr  = self.tensor_to_move_distribution(output, board)
        action = distr.sample()


        move_idx = torch.unravel_index(action, ChessNN.output_shape)
        move     = Move(*move_idx)
        log_prob = distr.log_prob(action)
        return move, log_prob



    @staticmethod
    def board_to_tensor(board: Board): 
        '''
        Transforms board into input tensor.  

        Parameters: 
            board (chess.Board): expects white to play

        Returns: 
            input tensor (torch.tensor): one hot encoding
        '''
        
        # indice list where pieces exist
        idx = [(square,
               (piece.piece_type - 1) + (6 if piece.color else 0))
               for square,piece in board.piece_map().items()]

        # create tensor
        t                   = torch.zeros(ChessNN.input_shape)
        t[tuple(zip(*idx))] = 1
        return t




    @staticmethod
    def tensor_to_move_distribution(tensor: Tensor, board: Board): 
        ''' 
        Transforms model output tensor into move distribution. 
        The move distribution is masked according to legal moves

        Parameters: 
            board (chess.Board): 

        Returns: 
            move distribution (torch.distributions.Categorical)
        '''
        mask   = ChessNN.move_mask(board)
        logits = tensor.masked_fill(~mask, float('-inf'))
        distr  = Categorical(logits=logits)

        return distr



    @staticmethod
    def move_mask(board: Board):
        ''' 
        move mask for leagl moves 

        Parameters: 
            board (chess.Board): 

        Returns: 
            move mask (torch.tensor)
        '''
        moves = [(m.from_square,m.to_square) for m in board.legal_moves]
        idx   = tuple(zip(*moves))
        mask      = torch.zeros(ChessNN.output_shape, dtype=torch.bool)
        mask[idx] = 1
        mask      = mask.flatten()

        return mask 


