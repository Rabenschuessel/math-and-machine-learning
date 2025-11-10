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

    # create different positions from whole solution
    solutions = []
    positions = []
    for move in moves: 
        if color == chess.BLACK: 
            move = mirror_move(move)
        solutions.append(move)
        positions.append(board.fen())

        # update board state
        board.push(move)
        board = board.mirror()
        color = not color

    # update row
    row["Moves"] = solutions 
    row["FEN"]   = positions 
    return row


def main():  
    tqdm.tqdm.pandas()
    df1 = pd.read_csv("./data/lichess_puzzle_transformed.csv")
    df2 = df1.head(5)
    df3 = df2.progress_apply(transform_position, axis=1)
    df4 = df3.explode(["FEN", "Moves"])
    df4.to_csv("./data/lichess_transformed.csv")


if __name__ == "__main__":
    main()

