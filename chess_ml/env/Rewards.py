from chess import Board, Move
import chess


def attack_center(board: Board, move: Move) -> float: 
    '''
    Parameters: 
        board: current position white to play
        metadata: turn number, etc. board is not guaranteed to be white to play

    Return: 
        evaluation for white in board
    '''
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    return sum([len(board.attackers(not board.turn, square)) 
                for square in center])



def early_development(board: Board, move: Move) -> float: 
    if (len(board.move_stack) < 20 
        and board.piece_at(move.to_square).piece_type in [chess.KNIGHT, chess.BISHOP] 
        and chess.square_rank(move.from_square) in [0, 7]): 
        return 1
    return 0



def win(board: Board, move: Move) -> float: 
    if (outcome := board.outcome()) is not None: 
        if outcome.winner is None: 
            return 0.5
        return float(board.turn != outcome.winner) 
    return 0.


all_rewards = [attack_center, early_development, win]
