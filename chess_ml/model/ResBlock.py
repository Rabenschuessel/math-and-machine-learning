from torch import nn
from .ChessNN import ChessNN
from typing import List



class ChessResBlock(ChessNN): 
    def __init__(self, num_blocks=20, conv_block=64):
        '''Residual block implementation of ChessNN.
        '''
        super().__init__()
        output = ChessNN.output_size

        self.conv = nn.Conv2d(12, conv_block, 3, padding=1)
        self.relu = nn.ReLU()
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(conv_block, conv_block, 3, padding=1),
                nn.BatchNorm2d(conv_block),
                nn.ReLU(),
                nn.Conv2d(conv_block, conv_block, 3, padding=1),
                nn.BatchNorm2d(conv_block))
            for i in range(num_blocks)
        ])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_block*64, output)
        )


    def forward(self, x):
        y = self.conv(x)

        for block in self.blocks: 
            y = block(y) + y
            y = self.relu(y)

        y = self.fc(y)
        return y
