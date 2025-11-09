from chess import Board, Move


class Environment: 
    def __init__(self, rewards=[]):
        '''
        Parameters: 
            rewards: set of activated reward functions
        '''
        self.board   = Board()
        self.rewards = rewards



    def reset(self) -> Board: 
        self.board.reset()
        return self.board



    def step(self, move: Move) -> tuple[Board, float]: 
        '''
        Perform move and evaluate rewards. 
        Mirrors board afterwards so white is always playing
        '''

        self.board.push(move)
        reward = self.evaluate_rewards()
        self.board.mirror()
        return self.board, reward
    


    def evaluate_rewards(self) -> float: 
        return sum([reward(self.board) for reward in self.rewards])
    
