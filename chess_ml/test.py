from chess_ml.model.FeedForward import ChessFeedForward
from chess_ml.model.Convolution import ChessCNN
from chess_ml.model.ResBlock import ChessResBlock

arc2class = {
    'linear': ChessFeedForward,
    'cnn': ChessCNN,
    'resnet': ChessResBlock
}



arc2class['linear']()
