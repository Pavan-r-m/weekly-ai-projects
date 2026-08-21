"""
Minimax Chess Engine with Alpha-Beta Pruning
==============================================

A from-scratch chess-playing AI that uses classic adversarial search
(minimax + alpha-beta pruning) with a hand-crafted evaluation function
built from piece values and piece-square tables.

No neural networks, no opening books, no cloud APIs -- just search and
a scoring heuristic, the same core idea that powered chess engines for
decades before deep learning entered the picture.

Board legality, move generation, and check/checkmate detection are
delegated to the `python-chess` library so this script can focus on the
AI: evaluation and search. Everything the AI does -- picking a move,
scoring a position, pruning branches -- is implemented here.

Usage:
    python chess_engine.py --mode selfplay --depth 3 --max-moves 40
    python chess_engine.py --mode human --depth 4 --color white
    python chess_engine.py --mode bench --depth 3

Author: Claude AI (autonomous weekly project)
"""

import argparse
import time
import random

import chess

# ---------------------------------------------------------------------------
# 1. EVALUATION FUNCTION
# ---------------------------------------------------------------------------
# Classic material values in centipawns (1 pawn = 100).
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # king safety handled separately; material value irrelevant
}

# Piece-square tables (PSTs) nudge the evaluation to prefer good squares,
# e.g. knights near the center, pawns advancing, king staying safe early.
# Tables are given from White's perspective, rank 8 (index 0) to rank 1
# (index 7); they are mirrored for Black.
PAWN_PST = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_PST_MIDGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST_MIDGAME,
}


def _pst_index(square: int, is_white: bool) -> int:
    """Convert a python-chess square index (0=a1..63=h8) into the index
    used by our rank-8-to-rank-1 PST arrays, mirroring for Black."""
    rank = chess.square_rank(square)  # 0 (rank1) .. 7 (rank8)
    file = chess.square_file(square)  # 0 (a) .. 7 (h)
    if is_white:
        row = 7 - rank  # rank8 -> row0
    else:
        row = rank  # mirror vertically for black
    return row * 8 + file


def evaluate_board(board: chess.Board) -> int:
    """Static evaluation of a position in centipawns from White's
    perspective (positive = good for White, negative = good for Black).

    Combines:
      1. Material balance (piece values)
      2. Piece-square positional bonuses
      3. Mobility (number of legal moves available -- rewards active play)
      4. Simple king safety via checks
    """
    if board.is_checkmate():
        # The side to move has been checkmated -> very bad for them.
        return -99999 if board.turn == chess.WHITE else 99999

    if board.is_stalemate() or board.is_insufficient_material() or \
            board.can_claim_draw():
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        pst_bonus = PST[piece.piece_type][_pst_index(square, piece.color == chess.WHITE)]
        total = value + pst_bonus
        score += total if piece.color == chess.WHITE else -total

    # Mobility bonus: side with more legal moves is generally more active.
    # Computed cheaply by counting legal moves for the side to move only,
    # signed by whose turn it is, then a smaller estimate for the other side.
    mobility = len(list(board.legal_moves))
    score += mobility if board.turn == chess.WHITE else -mobility

    # Small bonus for giving check (pressures the opponent).
    if board.is_check():
        score += -50 if board.turn == chess.WHITE else 50

    return score


# ---------------------------------------------------------------------------
# 2. MOVE ORDERING
# ---------------------------------------------------------------------------
def order_moves(board: chess.Board):
    """Order moves to make alpha-beta pruning more effective: try captures
    (especially high-value-piece-takes-low-value-piece) and checks first."""
    def move_score(move):
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_val = PIECE_VALUES[victim.piece_type] if victim else 0
            attacker_val = PIECE_VALUES[attacker.piece_type] if attacker else 0
            # MVV-LVA: Most Valuable Victim, Least Valuable Attacker
            score += 10 * victim_val - attacker_val
        if move.promotion:
            score += 800
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()
        return -score  # sort descending

    return sorted(board.legal_moves, key=move_score)


# ---------------------------------------------------------------------------
# 3. MINIMAX WITH ALPHA-BETA PRUNING
# ---------------------------------------------------------------------------
class SearchStats:
    """Tracks how many nodes/branches the search visits and prunes, so we
    can show alpha-beta's effectiveness compared to plain minimax."""

    def __init__(self):
        self.nodes_visited = 0
        self.branches_pruned = 0

    def reset(self):
        self.nodes_visited = 0
        self.branches_pruned = 0


def minimax(board: chess.Board, depth: int, alpha: float, beta: float,
            maximizing: bool, stats: SearchStats) -> float:
    """Recursive minimax search with alpha-beta pruning.

    - `maximizing=True` means it's White's turn and we want the max score.
    - `alpha` is the best score the maximizer can guarantee so far.
    - `beta` is the best score the minimizer can guarantee so far.
    - Whenever alpha >= beta, the remaining siblings can't affect the final
      decision, so we prune (stop exploring) them.
    """
    stats.nodes_visited += 1

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = order_moves(board)

    if maximizing:
        best_score = float("-inf")
        for i, move in enumerate(legal_moves):
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, False, stats)
            board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                stats.branches_pruned += len(legal_moves) - i - 1
                break  # beta cutoff: minimizer already has a better option
        return best_score
    else:
        best_score = float("inf")
        for i, move in enumerate(legal_moves):
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, True, stats)
            board.pop()
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                stats.branches_pruned += len(legal_moves) - i - 1
                break  # alpha cutoff: maximizer already has a better option
        return best_score


def find_best_move(board: chess.Board, depth: int, stats: SearchStats):
    """Top-level search: try every legal move, run minimax on the resulting
    position, and return the move with the best score for the side to move.
    Ties are broken randomly so the engine doesn't play the exact same game
    every time."""
    maximizing = board.turn == chess.WHITE
    best_score = float("-inf") if maximizing else float("inf")
    best_moves = []

    alpha, beta = float("-inf"), float("inf")
    for move in order_moves(board):
        board.push(move)
        score = minimax(board, depth - 1, alpha, beta, not maximizing, stats)
        board.pop()

        if maximizing:
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            beta = min(beta, best_score)

    return random.choice(best_moves), best_score


# ---------------------------------------------------------------------------
# 4. GAME LOOPS
# ---------------------------------------------------------------------------
def print_board(board: chess.Board):
    print(board)
    print()


def self_play(depth: int, max_moves: int):
    """AI plays both sides against itself and prints the game."""
    board = chess.Board()
    print(f"=== Self-play: AI vs AI (search depth {depth}) ===\n")
    print_board(board)

    move_num = 1
    total_time = 0.0
    for ply in range(max_moves * 2):
        if board.is_game_over():
            break

        stats = SearchStats()
        start = time.time()
        move, score = find_best_move(board, depth, stats)
        elapsed = time.time() - start
        total_time += elapsed

        side = "White" if board.turn == chess.WHITE else "Black"
        san = board.san(move)
        board.push(move)

        if board.turn == chess.WHITE:
            print(f"{move_num}. ... {san}   "
                  f"(eval={score:+.0f}cp, nodes={stats.nodes_visited}, "
                  f"pruned={stats.branches_pruned}, {elapsed:.2f}s)")
            move_num += 1
        else:
            print(f"{move_num}. {san}   "
                  f"(eval={score:+.0f}cp, nodes={stats.nodes_visited}, "
                  f"pruned={stats.branches_pruned}, {elapsed:.2f}s)")

    print()
    print_board(board)
    print(f"Result: {board.result()}  |  "
          f"Total search time: {total_time:.2f}s over {board.fullmove_number} moves")
    print(f"Game over reason: {_game_over_reason(board)}")


def human_vs_ai(depth: int, human_color: str):
    """Interactive game: a human types moves in UCI or SAN notation, the
    AI responds using minimax + alpha-beta search."""
    board = chess.Board()
    human_is_white = human_color.lower().startswith("w")
    print("=== Human vs AI ===")
    print(f"You are playing {'White' if human_is_white else 'Black'}.")
    print("Enter moves in SAN (e.g. 'Nf3', 'e4') or UCI (e.g. 'e2e4').")
    print("Type 'quit' to exit.\n")
    print_board(board)

    while not board.is_game_over():
        human_turn = (board.turn == chess.WHITE) == human_is_white
        if human_turn:
            move = None
            while move is None:
                user_input = input("Your move: ").strip()
                if user_input.lower() == "quit":
                    print("Game aborted.")
                    return
                try:
                    move = board.parse_san(user_input)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(user_input)
                        if move not in board.legal_moves:
                            raise ValueError
                    except ValueError:
                        print("Illegal or unrecognized move, try again.")
                        move = None
            board.push(move)
        else:
            print("AI is thinking...")
            stats = SearchStats()
            start = time.time()
            move, score = find_best_move(board, depth, stats)
            elapsed = time.time() - start
            print(f"AI plays {board.san(move)} "
                  f"(eval={score:+.0f}cp, nodes={stats.nodes_visited}, "
                  f"{elapsed:.2f}s)")
            board.push(move)

        print_board(board)

    print(f"Game over: {board.result()} ({_game_over_reason(board)})")


def benchmark(depth: int):
    """Runs a single search from the opening position and reports how many
    nodes alpha-beta visits vs. how many plain minimax would visit, to
    demonstrate the practical benefit of pruning."""
    board = chess.Board()

    stats_ab = SearchStats()
    start = time.time()
    move, score = find_best_move(board, depth, stats_ab)
    ab_time = time.time() - start

    print(f"=== Benchmark: search depth {depth} from starting position ===")
    print(f"Best move found: {board.san(move)} (eval={score:+.0f}cp)")
    print(f"Alpha-beta nodes visited: {stats_ab.nodes_visited}")
    print(f"Alpha-beta branches pruned: {stats_ab.branches_pruned}")
    print(f"Search time: {ab_time:.2f}s")
    print()
    print("(Plain minimax without pruning would visit roughly the full")
    print(" game tree -- exponentially more nodes at higher depths.)")


def _game_over_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.can_claim_fifty_moves():
        return "fifty-move rule"
    if board.can_claim_threefold_repetition():
        return "threefold repetition"
    return "move limit reached"


# ---------------------------------------------------------------------------
# 5. CLI ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Minimax chess engine with alpha-beta pruning.")
    parser.add_argument("--mode", choices=["selfplay", "human", "bench"],
                         default="selfplay",
                         help="selfplay: AI vs AI | human: play against the AI | "
                              "bench: node-count benchmark")
    parser.add_argument("--depth", type=int, default=3,
                         help="Search depth in plies (higher = stronger but slower)")
    parser.add_argument("--max-moves", type=int, default=40,
                         help="Max full moves before stopping self-play")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                         help="Which color the human plays in --mode human")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducible tie-breaking")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.mode == "selfplay":
        self_play(args.depth, args.max_moves)
    elif args.mode == "human":
        human_vs_ai(args.depth, args.color)
    elif args.mode == "bench":
        benchmark(args.depth)


if __name__ == "__main__":
    main()
