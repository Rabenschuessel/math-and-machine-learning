import chess

class ChessEnv: 
    def __init__(self, rewards={"win"}):
        self.board   = chess.Board()
        self.rewards = rewards



    def reset(self): 
        self.board.reset()



    def act(self, move): 
        '''
        Returns: 
        - state
        - reward
        - done
        '''
        self.board.push(move)
        return self.board, self.compute_rewards(), self.board.is_game_over()



    def compute_rewards(self):
        return 0
