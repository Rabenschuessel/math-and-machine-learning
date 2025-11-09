from chess import Board
import chess

def attack_center(board: Board) -> float: 
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    return sum([len(board.attackers(chess.WHITE, square)) 
                for square in center])

