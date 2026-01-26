import pandas as pd

pd.set_option('styler.format.precision', 3)
to_typst = True
def print_df(df): 
    if to_typst: 
        print(df.style.to_typst())
    else:
        print(df)


df        = pd.read_parquet('results.parquet')
values    = df.index.levels
variables = df.index.names

# row is for both black and white games
dfr = (df + (500 - df.transpose())) / 1000
df = dfr



# mean points won by each model config
s = df.mean(axis=1).unstack(level='architecture')
print_df(s)

# mean points by pretraining
# cnn grandmaster games best
# resnet puzzles best, but not by much
s = df.mean(axis=1).groupby(['architecture', "train"]).mean().unstack(level='architecture')
print_df(s)

# print(pd.DataFrame(s.unstack(level='architecture')).style.to_typst())


# mean points by rewards
# cnn is better with only winning
# resnet is better with no reinforcement learning
s = df.mean(axis=1).groupby(['architecture', "rewards"]).mean().unstack(level='architecture')
print_df(s)
