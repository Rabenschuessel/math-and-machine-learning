from torch.utils.data import Dataset
import pandas as pd

class PositionDataset(Dataset): 
    def __init__(self, path="./data/lichess_puzzle_labeled.csv", max_len=None):
        '''PuzzleDataset implements a torch dataset for 
        an underlying csv file with chess puzzles.

        Parameters: 
            path: path to csv file containing puzzles
                expects the csv file to contain columns 
                - "FEN": fen position
                - "Moves": optimal move in the position 

        '''
        self.data = pd.read_csv(path, nrows=max_len)
        

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        row = self.data.iloc[idx]
        features = row["FEN"]
        label    = row["Moves"]
        return features, label



class MergedDataset(Dataset): 
    def __init__(self, *datasets):
        '''Merges multiple datasets into one
        '''
        self.datasets = datasets
        

    def __len__(self): 
        return sum([len(ds) for ds in self.datasets])

    def __getitem__(self, idx): 
        # get the right dataset
        for ds in self.datasets: 
            if idx >= len(ds): 
                idx -= len(ds)
            else: 
                return ds[idx]
        raise Exception('idx larger than number of datapoints')

