import chess.pgn 
from io import StringIO
import pandas as pd





df = pd.read_csv('./data/GM_games_dataset.csv', nrows=100)

def transform_gm_games(df, max_positions): 
    def mirror_move(move: chess.Move): 
        def mirror_square(sq: chess.Square): 
            return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

        return chess.Move(
                mirror_square(move.from_square),
                mirror_square(move.to_square)
        )
    positions = []
    moves     = []
    # iterate games
    for i, row in df.iterrows(): 
        game = chess.pgn.read_game(StringIO(row.pgn.splitlines()[-1]))
        board = game.board()
        # iterate moves in a game
        for move in game.mainline_moves(): 
            if board.turn == chess.BLACK: 
                positions.append(board.mirror().fen())
                moves.append(mirror_move(move).uci())
            else: 
                positions.append(board.fen())
                moves.append(move.uci())
            board.push(move)

        # stop when max reached
        if len(positions) > max_positions: 
            break

    # create new dataframe
    out = pd.DataFrame()
    out['position'] = positions
    out['moves']    = moves
    return out

rewards = ['control_center', 'r_0']
name2reward = {r.__name__:r for r in Rewards.ALL}
rew = set()
for r in rewards: 
    if r in name2reward.keys(): 
        rew.add(name2reward[r])
    else: 
        rew.update(Rewards.reward_sets[r])
