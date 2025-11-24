from torch.utils.data import Dataset
import pandas as pd

class PuzzleDataset(Dataset): 
    def __init__(self, path="./data/lichess_puzzle_labeled.csv"):
        '''PuzzleDataset implements a torch dataset for an underlying csv file with chess puzzles.

        Parameters: 
            path: path to csv file containing puzzles
                expects the csv file to contain columns 
                - "FEN": fen position
                - "Moves": optimal move in the position 

        '''
        self.data = pd.read_csv(path)
        

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        row = self.data.iloc[idx]
        features = row["FEN"]
        label    = row["Moves"]
        return features, label
