import numpy as np
import xarray as xr
import torch 
from chess_ml.env.Environment import Environment
from chess_ml.env import Rewards 
from pathlib import Path 



experiment = '2'
log_dir    = Path("logs/arena/experiment-2/games/batch-0000/")
ds         = xr.open_dataset(log_dir / 'rewards.nc')


ds['win'].sum(dim=("turn", 'game', 'color'))


np.absolute(ds['win']).sum(dim=("turn", 'game', 'color'))




