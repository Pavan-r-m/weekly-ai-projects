# Minimax Chess Engine with Alpha-Beta Pruning

A chess-playing AI built from classic adversarial search — no neural
networks, no cloud APIs, no opening book. Just a hand-crafted evaluation
function and a minimax search tree pruned with alpha-beta, the same core
technique that powered chess engines for decades (including early Deep
Blue-era ideas) before deep learning took over the field.

## Why it's interesting

Minimax + alpha-beta pruning is one of the foundational algorithms of
classical AI and still underlies parts of modern engines (combined with
neural evaluation in things like AlphaZero/Stockfish NNUE). This project
shows the algorithm in isolation: you can watch it evaluate positions,
prune huge chunks of the game tree, and still play sensible chess with
nothing more than material counting, positional tables, and depth-limited
lookahead.

It also makes the *value* of pruning concrete: the benchmark mode prints
how many branches alpha-beta cuts off compared to what plain minimax
would have explored, which grows exponentially with search depth.

## Tech stack & key concepts

- **Language:** Python 3
- **Board/move engine:** [`python-chess`](https://python-chess.readthedocs.io/)
  (`chess` package) — handles legal move generation, check/checkmate
  detection, and board representation, so the AI code can focus on search
  and evaluation.
- **Search algorithm:** Minimax with alpha-beta pruning (depth-limited,
  configurable)
- **Move ordering:** MVV-LVA (Most Valuable Victim, Least Valuable
  Attacker) heuristic for captures, plus promotion/check bonuses — this
  makes alpha-beta prune far more aggressively by trying strong moves
  first
- **Evaluation function:**
  - Material balance (standard piece values: P=100, N=320, B=330, R=500, Q=900)
  - Piece-square tables (PSTs) that reward good piece placement (e.g.
    knights toward the center, king tucked in the corner early game)
  - Mobility bonus (more legal moves = more active position)
  - Simple check bonus

## Installation

```bash
pip install -r requirements.txt
```

## How to run

**AI vs AI (self-play), depth 3, up to 40 full moves:**
```bash
python chess_engine.py --mode selfplay --depth 3 --max-moves 40
```

**Play against the AI as White (depth 4 is noticeably stronger but slower):**
```bash
python chess_engine.py --mode human --depth 4 --color white
```
Enter moves in SAN (`Nf3`, `e4`, `O-O`) or UCI (`g1f3`, `e2e4`) notation.

**Benchmark alpha-beta pruning efficiency from the starting position:**
```bash
python chess_engine.py --mode bench --depth 4
```

Optional flags: `--seed 42` for reproducible tie-breaking between equally
scored moves.

## Example output

```
=== Benchmark: search depth 3 from starting position ===
Best move found: Nf3 (eval=+28cp)
Alpha-beta nodes visited: 803
Alpha-beta branches pruned: 1017
Search time: 0.13s
```

```
=== Self-play: AI vs AI (search depth 2) ===
1. Nf3   (eval=+22cp, nodes=83, pruned=337, 0.01s)
1. ... Nf6   (eval=+28cp, nodes=109, pruned=351, 0.02s)
2. Nc3   (eval=+24cp, nodes=111, pruned=393, 0.02s)
2. ... Nc6   (eval=+16cp, nodes=119, pruned=429, 0.02s)
...
```

`eval` is the position score in centipawns (100 = one pawn's worth of
advantage) from White's perspective; `nodes`/`pruned` show how many tree
nodes alpha-beta visited vs. skipped for that move.

## How it works

1. **Move generation.** For any board position, `python-chess` gives us
   every legal move (respecting check, pins, castling rights, en passant,
   etc.).

2. **Evaluation (`evaluate_board`).** For a *leaf* position (search
   horizon reached, or game over), the engine scores it: sum each piece's
   material value plus a piece-square bonus based on where it sits on the
   board, add a small mobility bonus for the side to move, and check for
   checkmate/stalemate/draw as special cases returning extreme or neutral
   scores.

3. **Move ordering (`order_moves`).** Before recursing, moves are sorted
   so captures of valuable pieces by cheap pieces (MVV-LVA), promotions,
   and checks are tried first. Better moves explored earlier let
   alpha-beta prune more branches later.

4. **Minimax with alpha-beta (`minimax`).** The search alternates between
   a *maximizing* player (White, trying to increase the score) and a
   *minimizing* player (Black, trying to decrease it), recursing down to
   a fixed depth. Two bounds are tracked as the search progresses:
   `alpha` (the best score the maximizer can already guarantee) and
   `beta` (the best score the minimizer can already guarantee). As soon
   as `beta <= alpha` at any node, the remaining sibling moves are
   *pruned* — skipped entirely — because a rational opponent would never
   let the game reach that point in the first place. This produces the
   exact same result as plain minimax while exploring dramatically fewer
   nodes.

5. **Top-level move selection (`find_best_move`).** The root position
   tries every legal move, runs `minimax` on the resulting board one ply
   down, and keeps whichever move(s) produced the best score for the side
   to move (ties broken randomly so games vary).

6. **Game loops.** `selfplay` lets the AI play both colors and prints
   each move with its evaluation and search stats; `human` lets a person
   play against the AI via the terminal; `bench` runs a single search
   from the opening position and reports pruning efficiency.

## Notes & limitations

- No opening book or endgame tablebase — the engine reasons from
  first principles every move, so early play can look slightly
  unconventional compared to book theory.
- Search depth 3–4 plies runs in well under a second per move on a
  laptop; depth 5+ gets noticeably slower since the tree still grows
  exponentially (alpha-beta reduces the *constant factor*, not the
  underlying complexity class).
- No API keys or network access required — this project is fully
  self-contained and runs offline.
