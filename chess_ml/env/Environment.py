import logging
import chess
from chess import Board, Move


class Environment: 
    def __init__(self, rewards=[]):
        '''
        Environment acts as wrapper around `chess.Board` for reinforcement learning. 
        It has a state and returns a reward after each step. 

        Parameters: 
            rewards: set of activated reward functions
        '''
        self.board   = Board()
        self.rewards = rewards



    def reset(self) -> Board: 
        self.board.reset()
        return self.board



    def step(self, move: Move) -> tuple[Board, float, bool]: 
        '''
        Perform move and evaluate rewards. 
        Mirrors board afterwards so white is always playing
        '''
        # model returns white move, mirror move to black if black to play
        if self.board.turn == chess.BLACK:
            move = Environment.mirror_move(move)

        # default to queen promotion when nothing specified
        if (self.board.piece_at(move.from_square).piece_type == chess.PAWN 
                and chess.square_rank(move.to_square) in [0, 7]):
            move.promotion = chess.QUEEN
        

        self.board.push(move)
        board  = self.board if self.board.turn == chess.WHITE else self.board.mirror()
        reward = self.evaluate_rewards(move)

        return board, reward, self.board.is_game_over()
    


    def evaluate_rewards(self, move: Move) -> float: 
        # calculate rewards
        rewards     = [reward(self.board, move) for reward in self.rewards]
        acc_rewards = sum(rewards)

        # logging
        reward_dict = {reward_f.__name__: reward 
                        for reward_f, reward in zip(self.rewards, rewards)}
        logging.info("  Move: {}\n    Acc Rewards: {} \n    Rewards: {}"
                     .format(move, acc_rewards, reward_dict))

        return acc_rewards
    

    
    @staticmethod
    def mirror_move(move: chess.Move): 
        def mirror_square(sq: chess.Square): 
            return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

        return chess.Move(
                mirror_square(move.from_square),
                mirror_square(move.to_square)
        )
