import argparse
import pandas as pd
import chess
import tqdm


def transform_position(row: pd.Series): 
    def mirror_move(move: chess.Move): 
        def mirror_square(sq: chess.Square): 
            return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

        return chess.Move(
                mirror_square(move.from_square),
                mirror_square(move.to_square)
        )

    # parsing row
    fen, moves = row[["FEN", "Moves"]]
    board = chess.Board(fen)
    moves = [chess.Move.from_uci(m) for m in moves.split()]

    # make sure it is white to play
    color = board.turn
    if board.turn == chess.BLACK: 
        board = board.mirror()


    # The first move and other moves by that player tend to be blunders
    # only train on the good moves
    skip  = True

    # create different positions from whole solution
    solutions = []
    positions = []
    for move in moves: 
        if color == chess.BLACK: 
            move = mirror_move(move)

        if (board.piece_at(move.from_square).piece_type == chess.PAWN 
                and chess.square_rank(move.to_square) in [0, 7]):
            move.promotion = chess.QUEEN

        if not skip: 
            solutions.append(move)
            positions.append(board.fen())

        # update board state
        board.push(move)
        board = board.mirror()
        color = not color
        skip  = not skip

    # update row
    row["Moves"] = solutions 
    row["FEN"]   = positions 
    return row


def main():  
    input  = "./data/lichess_puzzle_transformed.csv"
    output = "./data/lichess_transformed.csv"
    parser = argparse.ArgumentParser(
        prog="puzzle-transform", 
        description="transform chess puzzle dataset")
    parser.add_argument('-i', '--input' , default=input )
    parser.add_argument('-o', '--output', default=output)
    args = parser.parse_args()

    tqdm.tqdm.pandas()
    df1 = pd.read_csv(args.input)
    # df1 = df1.head(5000)
    df1 = df1[~df1["Themes"].str.contains("underPromotion")]
    df1 = df1.progress_apply(transform_position, axis=1)
    df1 = df1.explode(["FEN", "Moves"])
    df1.to_csv(args.output)


if __name__ == "__main__":
    main()

