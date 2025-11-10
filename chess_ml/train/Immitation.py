import torch
from torch.utils.data import random_split, DataLoader

from chess_ml.data.Puzzles import PuzzleDataset
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward

################################################################################
#### Dataset
################################################################################
def get_dataloader(): 
    dataset = PuzzleDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, test_loader



################################################################################
#### Model
################################################################################

def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
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
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}:{(current/size)*100:.2f}%]")


def test(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = ChessNN.fen_to_tensor(x)
            y = ChessNN.move_to_tensor(y)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    device = "cpu"
    train_dl, test_dl = get_dataloader()

    model             = ChessFeedForward([512, 512, 512])
    optimizer         = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn           = torch.nn.CrossEntropyLoss()
    train(train_dl, model, loss_fn, optimizer)

    test(test_dl, model, loss_fn)

if __name__ == "__main__":
    main()  

