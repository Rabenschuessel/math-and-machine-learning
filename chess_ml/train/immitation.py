'''
This module provides a training procedure for immitation learning

    python -m chess_ml.train.immitation
'''


import torch
import logging
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from torch import device
from tqdm import tqdm 
from typing import Union

from chess_ml.data import PuzzleDataset
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward

################################################################################
#### Dataset
################################################################################
def get_dataloader(validation=0.0): 
    dataset         = PuzzleDataset()
    size            = int(len(dataset) * (1 - validation))
    train_size      = int(0.9 * size)
    test_size       = size - train_size
    validation_size = len(dataset) - train_size - test_size
    splits          = [train_size, test_size, validation_size]
    train_dataset, test_dataset, val_dataset = random_split(dataset, splits)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32)
    val_loader   = DataLoader(val_dataset, batch_size=32)
    return train_loader, test_loader, val_loader



################################################################################
#### Train
################################################################################
def train(dataloader, model, loss_fn, optimizer, device:Union[str,device]="cpu"):

    model.train()
    for batch, (x, y) in tqdm(enumerate(dataloader),
                              total=len(dataloader),
                              desc ="Training Routine",
                              unit ="Batch"):
        x = ChessNN.fen_to_tensor(x)
        y = ChessNN.move_to_tensor(y)
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        # soft = torch.softmax(pred, dim=1)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
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
            x = ChessNN.fen_to_tensor(x)
            y = ChessNN.move_to_tensor(y)
            x, y = x.to(device), y.to(device)
            pred = model(x)

            test_loss += loss_fn(pred, y).item()
            correct   += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct   /= size
    print(f"Test Error: \n Accuracy {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


################################################################################
#### Main
################################################################################
def main(experiment=1, 
         epochs=10):
    log_dir    = Path("logs/im/experiment-{}".format(experiment))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'log.log',
        level=logging.INFO,      
        format='%(message)s'  
    )

    val_holdout = 0.0
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Load Dataset")
    train_dl, test_dl, val_dl = get_dataloader(val_holdout)

    print("Load Model")
    model             = ChessFeedForward([512, 512, 512])
    optimizer         = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn           = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Epochs", unit="Epoch"):
        tqdm.write("Train Model")
        train(train_dl, model, loss_fn, optimizer, device)

        tqdm.write("Train Model")
        test(test_dl, model, loss_fn, device)

        if epoch % 5 == 0: 
            tqdm.write("Save Checkpoint")
            torch.save(model.state_dict(), log_dir / f"checkpoint-{epoch}-{val_holdout}-model.pth")


    tqdm.write("Validation")
    test(val_dl, model, loss_fn, device)

    print("Save Model")
    torch.save(model.state_dict(), log_dir / f"trained-{val_holdout}-final-model.pth")



if __name__ == "__main__":
    main()  

