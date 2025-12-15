from torch import nn
from .ChessNN import ChessNN
from typing import List


class ChessCNN(ChessNN): 
    def __init__(self, conv:List[tuple[int, int, int]]=[(12, 32, 3), (32, 64, 3), (64, 128, 3), (128, 256, 3), (256, 32, 1)]):
        '''Feed forward implementation of ChessNN.

        Parameters: 
            hidden: size of hidden layers
        '''
        super().__init__()
        output = ChessNN.output_size

        self.conv = nn.Sequential(
            *[v 
                for c in conv[:-1]
                for v in (nn.Conv2d(*c, padding=1), nn.ReLU())
            ],
            nn.Conv2d(*conv[-1]), 
            nn.ReLU()
        )

        fc_input = 8*8*conv[-1][1]
        self.flatten = nn.Flatten()
        hidden = fc_input
        self.policy_fc = nn.Linear(fc_input, output)
        self.value_fc = nn.Linear(fc_input, 1)




    def forward(self, x):
        channels = self.conv(x)
        flat = self.flatten(channels)
        policy_logits = self.policy_fc(flat)
        values = self.value_fc(flat).squeeze(-1)
        return policy_logits, values
