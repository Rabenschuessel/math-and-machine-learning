from torch import nn
from .ChessNN import ChessNN


class ChessFeedForward(ChessNN): 
    def __init__(self, hidden:list[int]=[600]):
        '''Feed forward implementation of ChessNN.

        Parameters: 
            hidden: size of hidden layers
        '''
        super().__init__()
        input  = [ChessNN.input_size]
        output = [ChessNN.output_size]

        layers = [nn.Linear(*l) for l in zip(input + hidden, hidden + output)]

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *[v 
                for layer in layers[:-1] 
                for v in (layer, nn.ReLU())
            ],
            layers[-1]
        )



    def forward(self, x):
        f = self.flatten(x)
        return self.stack(f)
