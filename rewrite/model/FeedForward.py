from torch import nn
from rewrite.model.ChessNN import ChessNN


class ChessFeedForward(ChessNN): 
    def __init__(self):
        super().__init__()
        input_size  = ChessNN.input_size
        output_size = ChessNN.output_size
        hidden = 600

        self.flatten = nn.Flatten(start_dim=0)
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden    , output_size), 
        )


    def forward(self, x):
        f = self.flatten(x)
        return self.stack(f)
