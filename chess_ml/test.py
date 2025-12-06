import xarray as xr
import torch 
from chess_ml.env.Environment import Environment
from chess_ml.env import Rewards 
from pathlib import Path 



experiment = 'trained_logs'
log_dir    = Path("logs/rl/experiment-{}".format(experiment))
ds         = xr.open_dataset(log_dir / 'rewards.nc')






