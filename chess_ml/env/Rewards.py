from collections import deque
from chess import BLACK, WHITE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, Outcome
from chess import Board, Move
import chess

### REWARD HYPERPARAMETERS ###

# win(...)
WIN_VALUE = 3.0
DRAW_VALUE = -0.3

# give_check(...)
CHECK_REWARD = 0.05

# king_safety(...)
KING_SAFETY_SCALE = 50.0

# material(...)
MATERIAL_SCALE = 10.0
PIECE_VALUES = {
    PAWN:   1,
    KNIGHT: 3,
    BISHOP: 3,
    ROOK:   5,
    QUEEN:  9,
    KING:   0,
}

# blunder_prevention(...)
BLUNDER_PIECE_PENALTY = {
    QUEEN:  0.30,   # bigger than a pawn (0.1), smaller than a piece (0.3) in effect
    ROOK:   0.20,
    BISHOP: 0.12,
    KNIGHT: 0.12,
    # Pawns are excluded on purpose (too noisy; material() already covers it well)
}
BLUNDER_MATE_PENALTY   = 1.0

# promoting(...)
PROMOTION_REWARD = 1.0

# castling(...)
CASTLING_REWARD  = 0.1

# control_center(...)
MAX_PLY_INNER_CENTER = 30
INNER_CENTER_SCALE   = 100.0

# control_outer_center(...)
MAX_PLY_OUTER_CENTER = 20
OUTER_CENTER_SCALE  = 150.0

# step_penalty(...)
MIN_MOVE_STACK = 20
STEP_PENALTY = -0.001

### END ###

def win(state: Board, move: Move, result: Board):
    """
    Terminal reward based on game outcome.

    +WIN_VALUE  : current player eventually wins
    -WIN_VALUE  : current player eventually loses
     0.0        : draw or non-terminal position

    Reward is always given from the perspective of the player
    to move in `state`.
    """

    outcome = result.outcome()
    if outcome is None:
        return 0.0

    # Draw → neutral outcome
    if outcome.winner is None:
        return DRAW_VALUE

    # If the player who was to move in `state` wins
    if outcome.winner == state.turn:
        return WIN_VALUE

    # Otherwise, the player to move in `state` loses
    return -WIN_VALUE

def give_check(state: Board, move: Move, result: Board):
    """
    Shaping reward for giving check to the opponent king.

    - Rewards positions where the opponent is in check after the move.
    - Very small reward to encourage attacking play.
    - Does not apply to terminal positions.
    """

    # Do not reward checks in terminal positions (checkmate is handled by win(...))
    if result.outcome() is not None:
        return 0.0

    # If the opponent is in check after the move sequence
    if result.is_check():
        return CHECK_REWARD

    return 0.0


def king_safety(state: Board, move: Move, result: Board):
    """
    Shaping reward for king safety and king pressure.

    The idea:
        - Decrease in enemy attack pressure on our king zone is good.
        - Increase in our attack pressure on the enemy king zone is good.
    The reward is computed from the perspective of the player to move in `state`.

    King zone:
        - The king's square
        - All squares a king could move to from that square (adjacent squares)

    This is a weak shaping signal and should not dominate material or win rewards.
    """

    # Do not shape king safety in terminal positions
    if result.outcome() is not None:
        return 0.0

    def king_zone_pressure(board: Board, color: chess.Color) -> float:
        """
        Compute how many enemy attackers target the king's zone
        for the given color.
        """
        king_sq = board.king(color)
        if king_sq is None:
            # Should not happen in legal games, but let's be sure
            return 0.0

        # King zone: king square + all adjacent squares (king moves)
        king_zone = [king_sq]
        king_zone += list(chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]))

        enemy = not color
        # Sum up the number of enemy attackers on each square in the king zone
        pressure = sum(len(board.attackers(enemy, sq)) for sq in king_zone)
        return float(pressure)

    # Pressure before and after (lower is better for the side)
    white_pressure_before = king_zone_pressure(state, WHITE)
    black_pressure_before = king_zone_pressure(state, BLACK)

    white_pressure_after = king_zone_pressure(result, WHITE)
    black_pressure_after = king_zone_pressure(result, BLACK)

    # Convert "pressure" into a "safety score": lower pressure = higher score
    safety_state = {
        WHITE: -white_pressure_before,
        BLACK: -black_pressure_before,
    }
    safety_result = {
        WHITE: -white_pressure_after,
        BLACK: -black_pressure_after,
    }

    # Change in safety for each side
    diff_white = safety_result[WHITE] - safety_state[WHITE]
    diff_black = safety_result[BLACK] - safety_state[BLACK]

    # Perspective: player to move in `state`
    if state.turn == WHITE:
        diff_current = diff_white
        diff_opponent = diff_black
    else:
        diff_current = diff_black
        diff_opponent = diff_white

    raw_change = diff_current - diff_opponent

    # Scale: one "unit" change in relative safety ~ 0.02
    # (keeping this weaker than a pawn = 0.1 and far weaker than a win = 3.0)
    return raw_change / KING_SAFETY_SCALE

def material(state: Board, move: Move, result: Board):
    """
    Material-based shaping reward.

    Returns the change in material balance caused by the last move
    (including the opponent's reply), from the perspective of the
    player to move in `state`.

    Scaling:
        Pawn   ≈ 0.1
        Knight ≈ 0.3
        Bishop ≈ 0.3
        Rook   ≈ 0.5
        Queen ≈ 0.9
    """
    
    # --- Material before the move ---
    material_state = {WHITE: 0, BLACK: 0}
    for piece in state.piece_map().values():
        material_state[piece.color] += PIECE_VALUES[piece.piece_type]

    # --- Material after opponent's reply ---
    material_result = {WHITE: 0, BLACK: 0}
    for piece in result.piece_map().values():
        material_result[piece.color] += PIECE_VALUES[piece.piece_type]

    # Material difference for each color
    diff_white = material_result[WHITE] - material_state[WHITE]
    diff_black = material_result[BLACK] - material_state[BLACK]

    # Perspective: player to move in `state`
    if state.turn == WHITE:
        diff_current = diff_white
        diff_opponent = diff_black
    else:
        diff_current = diff_black
        diff_opponent = diff_white

    # Scale so that one pawn ≈ 0.1 reward
    return (diff_current - diff_opponent) / MATERIAL_SCALE

def blunder_prevention(state: Board, move: Move, result: Board):
    """
    Shaping penalty for obvious blunders.

    This reward is intentionally simple and cheap:
    - Penalize if after the move+reply sequence (`result`) the current player's
      queen/rooks/minor pieces are left en prise (attacked and not defended).
    - Penalize heavily if the opponent has an immediate checkmate available
      in `result` (i.e., we allow a mate-in-1).

    The reward is returned from the perspective of the player to move in `state`.
    It should be a small shaping signal and must not dominate material or win.
    """

    # No shaping in terminal positions (win() handles that)
    if result.outcome() is not None:
        return 0.0

    current = state.turn
    opponent = not current

    # -------------------------------------------------------------------------
    # Helper: count "hanging" pieces (attacked by opponent, not defended by us)
    # -------------------------------------------------------------------------
    def count_hanging(board: Board, color: chess.Color) -> float:
        penalty = 0.0
        for sq, piece in board.piece_map().items():
            if piece.color != color:
                continue
            if piece.piece_type not in BLUNDER_PIECE_PENALTY:
                continue

            attackers = board.attackers(not color, sq)
            if not attackers:
                continue

            defenders = board.attackers(color, sq)
            if not defenders:
                penalty += BLUNDER_PIECE_PENALTY[piece.piece_type]
        return penalty

    hanging_before = count_hanging(state, current)
    hanging_after  = count_hanging(result, current)

    # Only penalize if we increased the number/value of hanging pieces
    hanging_delta = max(0.0, hanging_after - hanging_before)

    # -------------------------------------------------------------------------
    # Mate-in-1 blunder: if opponent can checkmate immediately in `result`
    # -------------------------------------------------------------------------
    mate_in_one_penalty = 0.0
    # result.turn should be the side to move after the opponent reply; in our
    # setup that's typically the "current" player again. We want to check if
    # the opponent has a mate-in-1 available *against* the current player.
    #
    # So we temporarily set it to opponent to test their legal moves.
    tmp = result.copy()
    tmp.turn = opponent

    for m in tmp.legal_moves:
        tmp2 = tmp.copy()
        tmp2.push(m)
        if tmp2.is_checkmate():
            mate_in_one_penalty = BLUNDER_MATE_PENALTY  # strong shaping penalty, but still < WIN_VALUE (3.0)
            break

    # Total penalty (negative reward)
    return -(hanging_delta + mate_in_one_penalty)

def promoting(state: Board, move: Move, result: Board):
    """
    Reward for pawn promotion.

    Pawn promotion is a rare but decisive event, especially in endgames.
    This reward provides a dense signal for a key strategic objective
    that would otherwise be very sparse.

    The reward is intentionally smaller than the terminal win reward
    to ensure that winning the game remains the primary objective.
    """

    if move.promotion is not None:
        # A promotion occurred (to queen, rook, bishop, or knight)
        return PROMOTION_REWARD

    return 0.0

def castling(state: Board, move: Move, result: Board):
    """
    Small shaping reward for castling.

    Castling generally improves king safety and connects the rooks.
    This reward encourages timely castling during the opening without
    enforcing it in every position.

    The reward is deliberately small so that castling is not preferred
    over more urgent tactical or strategic considerations.
    """

    if state.is_castling(move):
        # The current player castles
        return CASTLING_REWARD

    return 0.0

def control_squares(
    state: Board,
    result: Board,
    squares: list,
    max_ply: int,
    scale: float
):
    """
    Generic shaping reward for control over a given set of squares.

    Parameters
    ----------
    state : Board
        Board position before the move.
    result : Board
        Board position after the move (including opponent reply).
    squares : list
        List of squares to evaluate (e.g. center or outer center).
    max_ply : int
        Maximum ply (half-move count) during which the reward is active.
    scale : float
        Scaling factor for the reward (larger = smaller reward).

    Returns
    -------
    float
        Shaping reward from the perspective of the player to move in `state`.
    """

    # Compute ply number (half-move count) since the start of the game
    fullmove = state.fullmove_number
    ply_from_start = (fullmove - 1) * 2 + (0 if state.turn == WHITE else 1)

    # Disable shaping reward after the specified ply
    if ply_from_start > max_ply:
        return 0.0

    def control(board: Board):
        white = sum(len(board.attackers(WHITE, sq)) for sq in squares)
        black = sum(len(board.attackers(BLACK, sq)) for sq in squares)
        return white, black

    white_before, black_before = control(state)
    white_after,  black_after  = control(result)

    diff_white = white_after - white_before
    diff_black = black_after - black_before

    # Perspective: player to move in `state`
    if state.turn == WHITE:
        diff_current  = diff_white
        diff_opponent = diff_black
    else:
        diff_current  = diff_black
        diff_opponent = diff_white

    raw_change = diff_current - diff_opponent

    return raw_change / scale

def control_center(state: Board, move: Move, result: Board):
    """
    Shaping reward for control of the inner center squares (D4, D5, E4, E5).

    The reward measures the change in control of the four central squares
    caused by the last move (including the opponent's reply), from the
    perspective of the player to move in `state`.

    This reward is only active during the opening and early middlegame
    (up to max_ply) and is scaled to be much smaller than
    material and terminal win rewards.

    Purpose:
        - Encourage principled opening play
        - Guide early piece development
        - Avoid influencing middlegame and endgame decisions
    """

    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]

    return control_squares(
        state=state,
        result=result,
        squares=center_squares,
        max_ply=MAX_PLY_INNER_CENTER, # ~15 moves per side
        scale=INNER_CENTER_SCALE # stronger than outer center
    )

def control_outer_center(state: Board, move: Move, result: Board):
    """
    Shaping reward for control of the extended (outer) center.

    The extended center is defined as the ring of squares surrounding
    the four main central squares. Control of these squares reflects
    early piece activity and development.

    This reward is only active in the early opening phase
    (up to max_ply) and is intentionally scaled weaker
    than the inner center reward.

    Purpose:
        - Encourage early piece activity
        - Discourage passive or edge-based development
        - Provide weak guidance without overriding strategic choices
    """

    outer_center_squares = [
        chess.C3, chess.C4, chess.C5, chess.C6,
        chess.D3, chess.D6,
        chess.E3, chess.E6,
        chess.F3, chess.F4, chess.F5, chess.F6,
    ]

    return control_squares(
        state=state,
        result=result,
        squares=outer_center_squares,
        max_ply=MAX_PLY_OUTER_CENTER, # ~10 moves per side
        scale=OUTER_CENTER_SCALE # weaker shaping
    )

def step_penalty(state, move, result):
    if result.outcome() is not None:
        return 0.0
    if len(state.move_stack) < MIN_MOVE_STACK:
        return 0.0
    return STEP_PENALTY

### REWARD LISTS ###
ALL = [control_outer_center, control_center, castling, promoting, blunder_prevention, material, king_safety, give_check, win]
BEGINNERS_HELP = [control_center, castling, promoting, blunder_prevention, material, king_safety, give_check, win]
WITH_CASTLING = [castling, promoting, blunder_prevention, material, king_safety, give_check, win]
NORMAL = [promoting, blunder_prevention, material, king_safety, give_check, win]
NO_PROMO = [blunder_prevention, material, king_safety, give_check, win]
NO_BLUNDER_PREVENTION = [material, king_safety, give_check, win]
NO_MATERIAL = [blunder_prevention, king_safety, give_check, win]
MATERIAL_GAME = [material, win]
JUST_WIN = [win]
### END ###
