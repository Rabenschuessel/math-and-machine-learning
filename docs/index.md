# Chess_ml


# Quickstart 

First get the puzzle dataset from kaggle with: 
```
curl -L -o ./data/lichess-chess-puzzle-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/tianmin/lichess-chess-puzzle-dataset
```


Then unzip the dataset: 

Then transform the dataset:
```
python -m chess_ml.data.transform
```
