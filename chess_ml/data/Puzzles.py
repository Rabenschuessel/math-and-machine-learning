from torch.utils.data import Dataset
import pandas as pd


class PuzzleDataset(Dataset): 

    def __init__(self, path="./data/lichess_puzzle_labeled.csv"):
        self.data = pd.read_csv(path)
        

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        row = self.data.iloc[idx]
        features = row["FEN"]
        label    = row["Moves"]
        return features, label
