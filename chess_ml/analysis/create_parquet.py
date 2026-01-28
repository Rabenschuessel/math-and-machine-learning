import ast
import numpy
import pandas as pd
from pathlib import Path

# read results into data
files = [Path('./logs/tournament/500-games/cnn.out'),
         Path('./logs/tournament/500-games/resnet.out'), 
         Path('./logs/tournament/500-games/fc.out'), 
         ]
# files = [Path('./logs/tournament/cnn.out')]
data = []
for file in files: 
    with open(file) as f: 
        lines = f.readlines()

    for match, results in zip(lines[3::2], lines[4::2]): 
        models = match.rstrip().split("__vs__")
        r      = ast.literal_eval(results.rstrip().split('(')[1].split(")")[0])
        match  = dict(result=r)
        for color, model in zip(['white', 'black'], models): 
            p = model.split('-')
            m = dict(architecture=p[0], 
                     train=p[1],
                     rewards=p[2] if len(p)>2 else 'none')
            match[color] = m
        data.append(match)

# collect all possible hyperparameters
coords = dict()
for coord in ['architecture', 'train', 'rewards']:
    s = set()
    for color in ['black', 'white']: 
        s.update(d[color][coord] for d in data)
    coords[coord] = sorted(s)



# createa pd dataframe
idx = pd.MultiIndex.from_product(coords.values(), names=coords.keys())
df = pd.DataFrame(
    data = numpy.nan,
    index=idx, 
    columns=idx,
)
df.index.name = 'White'
df.columns.name = 'Black'



# fill dataframe
for match in data: 
    score = match['result'].get('1-0', 0) + (match['result'].get('1/2-1/2', 0) / 2)
    df.loc[
        (match['white']['architecture'], 
         match['white']['train'], 
         match['white']['rewards']), 

        (match['black']['architecture'], 
         match['black']['train'], 
         match['black']['rewards']), 
    ] = score 




df.to_parquet('results.parquet')
