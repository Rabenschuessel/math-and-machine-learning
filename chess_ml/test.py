import xarray as xr
import matplotlib.pyplot as plt



ds = xr.open_dataset("logs/rl/experiment-0/001/games/batch-0000/rewards.nc")



(ds.win != 0).sum(dim='turn').sel(color=False)

ds.win.sel(color=False, game=0).values


ds['win'].sum(dim=("turn", 'game', 'color'))






