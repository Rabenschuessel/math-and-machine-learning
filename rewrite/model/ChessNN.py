from torch import nn
from chess import WHITE, Board

class ChessNN(nn.Module): 
    input_size  = 64 * 12
    output_size = 64 * 64

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

        input  = self.board_to_tensor(board)
        output = self(input)
        masked = self.tensor_to_move_distribution(board)

        # TODO sample from distribution
        raise NotImplementedError()



    @staticmethod
    def board_to_tensor(board: Board): 
        '''
        Transforms board into input tensor.  

        Parameters: 
            board (chess.Board): expects white to play

        Returns: 
            input tensor (torch.tensor): one hot encoding
        '''

        if board.turn is not WHITE: 
            raise ValueError("Invalid Parameter: expects white to play")

        # TODO
        raise NotImplementedError()



    @staticmethod
    def tensor_to_move_distribution(board: Board): 
        ''' 
        Transforms model output tensor into move distribution. 
        The move distribution is masked according to legal moves

        Parameters: 
            board (chess.Board): 

        Returns: 
            move distribution (torch.tensor)
        '''

        # TODO 
        raise NotImplementedError()



    @staticmethod
    def move_mask(board: Board):
        ''' 
        move mask for leagl moves 

        Parameters: 
            board (chess.Board): 

        Returns: 
            move mask (torch.tensor)
        '''

        # TODO 
        raise NotImplementedError()
