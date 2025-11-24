import argparse
import torch
import chess
import chess.svg
import cairosvg
from tqdm import tqdm
from chess_ml.env.Environment import Environment
from chess_ml.env.Rewards import attack_center
from chess_ml.model.ChessNN import ChessNN
from chess_ml.model.FeedForward import ChessFeedForward


def pit(model1, model2, plot=False, game=0): 
    '''Pit two models against each other for a certain number of games'''
    env   = Environment([attack_center])
    board = env.reset()

    i    = 0
    over = False
    if plot: 
        flip = False
        svg  = chess.svg.board(board)
        cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                         write_to="games/game-{:02d}-move-{:03d}.png".format(game, i))

    while not over:
        if i % 2 == 0: 
            move, _        = model1.predict(board)
        else: 
            move, _        = model2.predict(board)

        board, _, over = env.step(move)

        i   += 1
        if plot: 
            flip = not flip
            b    = board.mirror() if flip else board
            svg  = chess.svg.board(b)
            cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                             write_to="games/game-{:02d}-move-{:03d}.png".format(game, i))

    return env._board.result()





def main(path1=None, path2=None):
    games = 1000
    m1 = ChessFeedForward([512, 512, 512])
    if path1 is not None: 
        m1.load_state_dict(torch.load(path1))

    m2 = ChessFeedForward([512, 512, 512])
    if path2 is not None: 
        m2.load_state_dict(torch.load(path2))

    results = {"win": 0, "draw": 0, "loss": 0}
    for i in tqdm(range(games)): 
        r = pit(m1, m2)
        if r == "1/2-1/2": 
            results["draw"] += 1
        elif r == "1-0": 
            results["win"] += 1
        elif r == "0-1": 
            results["loss"] += 1
        else: 
            print("parsing error")

    print(results)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="immitation learning", 
        description="transform chess puzzle dataset")
    parser.add_argument('-1', '--model1', default=None)
    parser.add_argument('-2', '--model2', default=None)
    args = parser.parse_args()
    main()



ChessNN()
