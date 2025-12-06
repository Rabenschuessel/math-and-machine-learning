import xarray as xr
import torch 
from chess_ml.env.Environment import Environment
from chess_ml.env import Rewards 

env = Environment(Rewards.ALL)



t  = torch.randn(8, 11, 3)
ds = xr.DataArray(
    t.cpu().numpy(), 
    dims=['game', 'time', 'reward'], 
    coords={ 'reward': [r.__name__ for r in env._rewards] }
)



ds.sum(dim=("game", "time"))


[0.5, 1]
gamma = 0.9

[1.4, 1]
