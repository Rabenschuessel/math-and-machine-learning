'''
This module provides a training procedure for immitation learning

    python -m chess_ml.train.immitation
'''


import argparse
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from torch import device
from tqdm import tqdm 
from typing import Union

from chess_ml.data import PuzzleDataset
from chess_ml.model import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward

################################################################################
#### Dataset
################################################################################
def get_dataloader(path, validation=0.0, batch_size=512): 

    dataset         = PuzzleDataset(path=path)
    size            = int(len(dataset) * (1 - validation))
    train_size      = int(0.9 * size)
    test_size       = size - train_size
    validation_size = len(dataset) - train_size - test_size
    splits          = [train_size, test_size, validation_size]
    train_dataset, test_dataset, val_dataset = random_split(dataset, splits)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, test_loader, val_loader



################################################################################
#### Train
################################################################################
def train(dataloader, model, loss_fn, optimizer, device:Union[str,device]="cpu"):

    model.train()
    for batch, (fens, moves) in tqdm(enumerate(dataloader),
                              total=len(dataloader),
                              desc ="Training Routine",
                              unit ="Batch"):
        m = ChessNN.fen_to_mask(fens).to(device)
        x = ChessNN.fen_to_tensor(fens).to(device)
        y = ChessNN.move_to_labels(moves).to(device)
        pred   = model(x)
        logits = pred.masked_fill(~m, float('-inf'))
        loss   = loss_fn(logits, y)


        if (m[range(len(y)), y] == False).sum() != 0:
            idx = torch.argwhere(m[range(len(y)), y] == False)
            tqdm.write(fens[idx])
            tqdm.write(moves[idx])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss = loss.item()
            tqdm.write(f"batch: {batch} loss: {loss:>7f}")


################################################################################
#### Test
################################################################################
def test(dataloader, model, loss_fn, device:Union[str,device]="cpu"):
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(dataloader,
                                   total=len(dataloader),
                                   desc ="Testing Model",
                                   unit ="Batch"):
            m = ChessNN.fen_to_mask(x).to(device)
            x = ChessNN.fen_to_tensor(x).to(device)
            y = ChessNN.move_to_labels(y).to(device)
            pred   = model(x)
            logits = pred.masked_fill(~m, float('-inf'))
            # soft = torch.softmax(logits, dim=1)

            test_loss += loss_fn(logits, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct   /= size
    print(f"Test Error: \n Accuracy {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


################################################################################
#### Main
################################################################################
def main(experiment=1, 
         epochs=10,
         model_path=None, 
         path=None):
    log_dir    = Path("logs/im/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )
    models_dir = Path(log_dir/"models")
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    val_holdout = 0.1
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on {}".format(device))

    print("Load Dataset")
    train_dl, test_dl, val_dl = get_dataloader(path, val_holdout)

    print("Load Model")
    model             = ChessFeedForward([512, 512, 512])
    if model_path is not None: 
        model.load_state_dict(torch.load(model_path))
    model             = model.to(device)
    optimizer         = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn           = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Epochs", unit="Epoch"):
        tqdm.write("Train Model")
        train(train_dl, model, loss_fn, optimizer, device)

        tqdm.write("Test Model")
        test(test_dl, model, loss_fn, device)

        if epoch % 2 == 0: 
            tqdm.write("Save Checkpoint")
            torch.save(model.state_dict(), models_dir / f"checkpoint-{epoch}.pth")


    tqdm.write("Validation")
    test(val_dl, model, loss_fn, device)

    print("Save Model")
    torch.save(model.state_dict(), models_dir / f"final-model.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="immitation learning", 
        description="transform chess puzzle dataset")
    parser.add_argument('-e', '--epochs' , default=10, type=int)
    parser.add_argument('-n', '--experiment-name', default=1, type=int)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-d', '--data', default='./data/lichess_puzzle_labeled.csv')
    args = parser.parse_args()

    main(experiment=args.experiment_name, epochs=args.epochs, model_path=args.model, path=args.data)



