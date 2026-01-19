
import logging
import chess
import chess.pgn
from collections import deque, deque
from chess import Board, Move 
from typing import Tuple
import chess_ml.env.Rewards as Rewards


class Environment: 
    def __init__(self, rewards=[]):
        '''
        Environment acts as wrapper around `chess.Board` for reinforcement learning. 
        It has a state and returns a reward after each step. 

        Parameters: 
            rewards: set of activated reward functions
        '''
        self._board   = Board()
        self._rewards = rewards
        self.reward_log = {r.__name__: [] for r in self._rewards}
        self.reward_log["sum"] = []

        self.mov_q = deque()
        self.pos_q = deque([Board()])
        self.reward_hist = []



    def reset(self) -> Board: 
        self.reward_log = {r.__name__: [] for r in self._rewards}
        self.reward_log["sum"] = []
        self._board.reset()

        self.mov_q = deque()
        self.pos_q = deque([Board()])
        self.reward_hist = []
        return self._board


    def get_game(self) -> chess.pgn.Game: 
        '''Returns the game so that it can be saved as pgn'''
        return chess.pgn.Game.from_board(self._board)


    def get_rewards(self): 
        # add reward for last move of the game
        if len(self.pos_q) > 0: 
            self.pos_q.clear()
            self.mov_q.clear()
            r = [0 if reward.__name__ != "win" 
                   else Rewards.WIN_VALUE for reward in self._rewards]
            self.reward_hist.append(r)
        # otherwise add zero rewards so that all games in batch have the same length
        else: 
            r = [0 for reward in self._rewards]
            self.reward_hist.append(r)

        return self.reward_hist[0::2], self.reward_hist[1::2]


    def step(self, move: Move) -> Tuple[Board, bool]: 
        '''
        Perform move and evaluate rewards. 
        Mirrors board afterwards so white is always playing
        '''
        # used for batch processing
        if self._board.is_game_over(): 
            board  = self._board if self._board.turn == chess.WHITE else self._board.mirror()
            # add reward for last move of the game
            if len(self.pos_q) > 0: 
                self.pos_q.clear()
                self.mov_q.clear()
                r = [0 if reward.__name__ != "win" 
                       else Rewards.WIN_VALUE for reward in self._rewards]
                self.reward_hist.append(r)
            # otherwise add zero rewards (for batch processing)
            else: 
                r = [0 for reward in self._rewards]
                self.reward_hist.append(r)
            return board, True

        # model returns white move, mirror move to black if black to play
        if self._board.turn == chess.BLACK:
            move = Environment.mirror_move(move)

        # default to queen promotion when nothing specified
        if (self._board.piece_at(move.from_square).piece_type == chess.PAWN 
                and chess.square_rank(move.to_square) in [0, 7]):
            move.promotion = chess.QUEEN
        

        self._board.push(move)
        board  = self._board if self._board.turn == chess.WHITE else self._board.mirror()
        self.evaluate_rewards(move)


        return board, self._board.is_game_over()
    


    def evaluate_rewards(self, move: Move): 
        self.pos_q.append(self._board.copy())
        self.mov_q.append(move)

        if len(self.pos_q) >= 3: 
            state  = self.pos_q.popleft()
            result = self.pos_q[1]
            move   = self.mov_q.popleft()

            rewards     = [reward(state, move, result) for reward in self._rewards]
            self.reward_hist.append(rewards)

        #
        # # calculate rewards
        #
        # # logging
        # reward_dict = {reward_f.__name__: reward 
        #                 for reward_f, reward in zip(self._rewards, rewards)}
        # logging.info("  Move: {}\n    Acc Rewards: {} \n    Rewards: {}"
        #              .format(move, acc_rewards, reward_dict))
        # for k, v in reward_dict.items(): 
        #     self.reward_log[k].append(v)
        # self.reward_log["sum"].append(acc_rewards)
        #
        # return acc_rewards
    

    
    @staticmethod
    def mirror_move(move: chess.Move): 
        def mirror_square(sq: chess.Square): 
            return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

        return chess.Move(
                mirror_square(move.from_square),
                mirror_square(move.to_square)
        )
