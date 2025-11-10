import torch
import torch.nn as nn

class ChessModel(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        input  = 8*8*12
        hidden = 600
        output = 64*64*5

        self.flatten = nn.Flatten(start_dim=0)
        self.stack = nn.Sequential(
            nn.Linear(input, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, output), 
        )


    def forward(self, x):
        f = self.flatten(x)
        return self.stack(f)
