import tqdm
import argparse
import chess.pgn 
from io import StringIO
import pandas as pd


def transform_game(row: pd.Series): 
    def mirror_move(move: chess.Move): 
        def mirror_square(sq: chess.Square): 
            return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

        return chess.Move(
                mirror_square(move.from_square),
                mirror_square(move.to_square)
        )

    positions = []
    moves     = []
    game = chess.pgn.read_game(StringIO(row.pgn.splitlines()[-1]))
    board = game.board()
    # iterate moves in a game
    for move in game.mainline_moves(): 
        if move.promotion in [None, chess.QUEEN]:
            if board.turn == chess.BLACK: 
                positions.append(board.mirror().fen())
                moves.append(mirror_move(move).uci())
            else: 
                positions.append(board.fen())
                moves.append(move.uci())
        board.push(move)

    row['FEN']   = positions
    row['Moves'] = moves
    return row


def main():  
    max_positions = 6000000
    input  = "./data/GM_games_dataset.csv"
    output = "./data/gm_positions_labeled.csv"
    parser = argparse.ArgumentParser(
        prog="game-transform", 
        description="transform gm games dataset")
    parser.add_argument('-i', '--input' , default=input )
    parser.add_argument('-o', '--output', default=output)
    args = parser.parse_args()

    tqdm.tqdm.pandas()
    df1 = pd.read_csv(args.input, nrows=int(max_positions / 50))
    df1 = df1[['pgn']]
    df1 = df1.progress_apply(transform_game, axis=1)
    df1 = df1.explode(["FEN", "Moves"])
    df1 = df1[['FEN', 'Moves']]
    df1 = df1.head(max_positions)
    df1.to_csv(args.output)


if __name__ == "__main__":
    main()

