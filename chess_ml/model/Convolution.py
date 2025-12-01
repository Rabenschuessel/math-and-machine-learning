from torch import nn
from .ChessNN import ChessNN
from typing import List


class ChessCNN(ChessNN): 
    def __init__(self, conv:List[tuple[int, int, int]]=[(12, 32, 3), (32, 64, 3), (64, 128, 3), (128, 256, 3)]):
        '''Feed forward implementation of ChessNN.

        Parameters: 
            hidden: size of hidden layers
        '''
        super().__init__()
        output = ChessNN.output_size

        self.conv = nn.Sequential(
            *[v 
                for c in conv
                for v in (nn.Conv2d(*c, padding=1), nn.ReLU())
            ],
        )

        fc_input = 8*8*conv[-1][1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, output)
        )




    def forward(self, x):
        channels = self.conv(x)
        return self.fc(channels)
