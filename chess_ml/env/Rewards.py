from collections import deque
from chess import BLACK, WHITE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, Outcome
from chess import Board, Move
import chess


def win(state: Board, move: Move, result: Board): 
    '''returns Rewards.WIN_VALUE on white and -WIN_VALUE on loss
    '''
    if (outcome := result.outcome()) is not None: 
        if outcome.winner is None: 
            return 0.5 * WIN_VALUE
        if outcome.winner is state.turn: 
            return WIN_VALUE
        return -WIN_VALUE
    return 0.0
    


def control_center(state: Board, move: Move, result: Board): 
    '''returns the difference in control over center between state and board

    Example: state is white to play, puts pawn in middle. 
    Black also puts pawn in middle. central_control returns 0
    '''
    center = [chess.D4, chess.D5, chess.E4, chess.E5]

    sum_state = {}
    for c in [WHITE, BLACK]: 
        sum_state[c] = sum([len(state.attackers(c, s)) for s in center])

    sum_result = {}
    for c in [WHITE, BLACK]: 
        sum_result[c] = sum([len(result.attackers(c, s)) for s in center])

    diff        = {WHITE: 0, BLACK: 0}
    diff[WHITE] = sum_result[WHITE] - sum_state[WHITE]
    diff[BLACK] = sum_result[BLACK] - sum_state[BLACK]
    return (diff[state.turn] - diff[not state.turn]) / 100


def material(state: Board, move: Move, result: Board): 
    '''returns the difference in material imbalance between state and result.

    If one or more pieces were captured between state and result, 
    return the material difference.
    The sign represents whether the player won or lost material. 

    Example: state is white to play, white captures pawn with knight. 
    black follows up by capturing knight: white lost 2 points of material. 
    Material returns -2. 
    '''

    piece_value = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

    # Material Imbalance before the move
    sum_state = {WHITE: 0, BLACK: 0}
    for piece in state.piece_map().values(): 
        sum_state[piece.color] += piece_value[piece.piece_type]

    # Material Imbalance after opponent reacted
    sum_result = {WHITE: 0, BLACK: 0}
    for piece in result.piece_map().values(): 
        sum_result[piece.color] += piece_value[piece.piece_type]

    diff        = {WHITE: 0, BLACK: 0}
    diff[WHITE] = sum_result[WHITE] - sum_state[WHITE]
    diff[BLACK] = sum_result[BLACK] - sum_state[BLACK]
    return (diff[state.turn] - diff[not state.turn]) / 100




ALL       = [control_center, material, win]
WIN_VALUE = 1.0


################################################################################
#### Testing
################################################################################




def test_material(): 
    state    = Board('rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2')
    move     = Move.from_uci('c2c4')
    op       = Move.from_uci('d5c4')
    opponent = state.copy()
    opponent.push(move)
    opponent.push(op)

    print(material(state, move, opponent))

    state = opponent.copy()
    state.pop()
    move  = op
    op    = Move.from_uci("b1c3")
    opponent.push(op)

    print(material(state, move, opponent))





#
#
# test_material()
#
#
#
#
# def test_control_center(): 
#     state    = Board('rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2')
#     move     = Move.from_uci('c2c4')
#     op       = Move.from_uci('d5c4')
#     opponent = state.copy()
#     opponent.push(move)
#     opponent.push(op)
#
#     print(control_center(state, move, opponent))
#
#
#
#
# test_control_center()



# moves = ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "Be3", "Bg7", "Qd2", "c6", "f3", "b5",
#  "Nge2", "Nbd7", "Bh6", "Bxh6", "Qxh6", "Bb7", "a3", "e5", "O-O-O", "Qe7", 
#  "Kb1", "a6", "Nc1", "O-O-O", "Nb3", "exd4", "Rxd4", "c5", "Rd1", "Nb6", "g3",
#  "Kb8", "Na5", "Ba8", "Bh3", "d5", "Qf4+", "Ka7", "Rhe1", "d4", "Nd5", "Nbxd5",
#  "exd5", "Qd6", "Rxd4", "cxd4", "Re7+", "Kb6", "Qxd4+", "Kxa5", "b4+", "Ka4", 
#  "Qc3", "Qxd5", "Ra7", "Bb7", "Rxb7", "Qc4", "Qxf6", "Kxa3", "Qxa6+", "Kxb4",
#  "c3+", "Kxc3", "Qa1+", "Kd2", "Qb2+", "Kd1", "Bf1", "Rd2", "Rd7", "Rxd7",
#  "Bxc4", "bxc4", "Qxh8", "Rd3", "Qa8", "c3", "Qa4+", "Ke1", "f4", "f5", "Kc1",
#  "Rd2", "Qa7"]

def test_queue(): 
    moves = ["f3", 'e6', 'g4', 'Qh4#']

    from collections import deque
    board = Board()
    pos_q = deque([board.copy()])
    mov_q = deque()
    rewards = []

    for move in moves: 
        move = board.push_san(move)
        pos_q.append(board.copy())
        mov_q.append(move)

        if len(pos_q) >= 3: 
            state  = pos_q.popleft()
            result = pos_q[1]
            move   = mov_q.popleft()
            rewards.append((material(state, move, result),
                    control_center(state, move, result), 
                           win(state, move, result)))

        if board.outcome() is not None: 
            rewards.append((0,0,WIN_VALUE))

    print(rewards)


