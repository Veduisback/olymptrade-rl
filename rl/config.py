"""
Shared configuration for the ASIA_X multi-horizon RL project.

RECONSTRUCTED FILE
===================
This file was not present in the delivered project archive (only a stale
compiled .pyc from an earlier version existed, and it was never committed
to git). It has been rebuilt from scratch based purely on how its names
are *used* elsewhere in the codebase:

    rl/multi_horizon_agent_v3.py    -> DEVICE, RANDOM_SEED
    rl/multi_horizon_dataset_v3.py  -> ACTION_WAIT/BUY/SELL, WIN_REWARD,
                                        LOSS_REWARD, WAIT_REWARD,
                                        MAX_TIMESTAMP_GAP
    rl/train_multi_horizon_v3.py    -> DATA_DIR, MODEL_DIR, LOOKBACK_QUOTES
    rl/evaluate_multi_horizon_v3.py -> DATA_DIR, MODEL_DIR, WIN_REWARD,
                                        LOSS_REWARD

Two values are NOT recoverable from usage and are best-effort guesses that
you should sanity-check / tune against your own data and broker payout:

    WAIT_REWARD       -> assumed 0.0 (no P&L for not trading)
    MAX_TIMESTAMP_GAP -> assumed 2.0s (matches data/check_timestamp_gaps.py)

Everything else (action ids, WIN/LOSS reward, directory layout, lookback)
is pinned down unambiguously by the README and by how the other modules
consume it, so those should be safe.
"""

from __future__ import annotations

from pathlib import Path


# ============================================================
# PATHS
# ============================================================
# rl/config.py -> parents[1] is the project root (olymptrade-rl/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "rl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================
RANDOM_SEED = 42

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    # config.py is imported by pure-numpy modules too (data.py,
    # state.py, multi_horizon_dataset_v3.py); don't force a hard
    # torch dependency on code paths that never touch a tensor.
    DEVICE = "cpu"


# ============================================================
# STATE CONSTRUCTION
# ============================================================
# Number of raw quotes of history used to build one state vector.
# Confirmed by every call site in multi_horizon_dataset_v3.py
# (hardcoded literal 30 there); centralized here so it's a single
# source of truth instead of a magic number scattered across files.
LOOKBACK_QUOTES = 30

# Largest allowed gap (in seconds) between two consecutive quotes
# inside a lookback/horizon window before that window is treated as
# broken (e.g. platform reconnect, feed drop). Matches the threshold
# already used by data/check_timestamp_gaps.py for consistency.
# Tune this against your own feed's real jitter if it's too strict
# or too loose.
MAX_TIMESTAMP_GAP = 2.0


# ============================================================
# ACTIONS
# ============================================================
ACTION_WAIT = 0
ACTION_BUY = 1
ACTION_SELL = 2


# ============================================================
# REWARDS
# ============================================================
# 85% payout assumption, per README ("Reward System" section).
WIN_REWARD = 0.85
LOSS_REWARD = -1.0

# Reward for choosing WAIT. Assumed 0.0 (no stake placed, no P&L).
# If you want the model to be actively penalized for sitting out of
# genuinely profitable setups (an opportunity-cost signal) this could
# instead be set to a small negative number, e.g. -0.02.
WAIT_REWARD = 0.0
