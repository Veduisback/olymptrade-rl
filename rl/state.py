"""
State-vector construction for the ASIA_X multi-horizon RL project.

RECONSTRUCTED FILE
===================
Not present in the delivered archive (see rl/config.py header for the
full explanation).

IMPORTANT - this is the piece you should trust least
------------------------------------------------------
`state_size()` and the *shape* (22 floats) are pinned down exactly by
multi_horizon_agent_v3.py's hard assertion:

    STATE_DIM = state_size()
    if STATE_DIM != 22: raise RuntimeError(...)

But the actual *feature engineering* -- which 22 numbers, computed how
-- is NOT recoverable from any file in the archive. Nothing else
imports individual feature names, so there was no usage pattern to
reverse-engineer it from.

What this means concretely: the existing checkpoint
`models/rl/multi_horizon_dqn_v3.pt` was trained against whatever the
ORIGINAL state.py produced. Its weights encode a mapping from those
22 specific numbers to expected reward. This reconstructed
`build_state` almost certainly computes a DIFFERENT set of 22 numbers.
Loading the old checkpoint on top of this file will succeed (shapes
match) but its predictions will be meaningless noise, because index 7
in the old feature vector is not index 7 here.

=> Treat this as a fresh, defensible starting point. Retrain from
   scratch with this state.py (or replace the feature logic below with
   your own, better-remembered version, then retrain) rather than
   trying to reuse multi_horizon_dqn_v3.pt.

Feature design used here (22 features, built from the last
`lookback` raw quotes ending at and including `index`):

  0  ret_1            log return, 1 tick back
  1  ret_2            log return, 2 ticks back
  2  ret_5            log return, 5 ticks back
  3  ret_10           log return, 10 ticks back
  4  ret_20           log return, 20 ticks back
  5  ret_full         log return across the whole lookback window
  6  vol_full         std-dev of tick log returns, whole window
  7  vol_recent       std-dev of tick log returns, last 10 ticks
  8  vol_ratio        vol_recent / vol_full  (volatility regime shift)
  9  zscore           (price - window mean) / window std
 10  minmax_pos       price's position within the window's [min, max]
                       range, centered on 0 (-0.5 .. +0.5)
 11  ema_fast_dev     (EMA-5 - price) / price
 12  ema_cross_fs     (EMA-5 - EMA-15) / price
 13  ema_cross_sl     (EMA-15 - EMA-full) / price
 14  rsi_norm         RSI(window), rescaled from [0,100] to [-1, 1]
 15  mean_dt          mean seconds between ticks in the window
 16  std_dt           std-dev of seconds between ticks (feed jitter)
 17  last_dt          seconds since the previous tick (right now)
 18  up_frac          fraction of ticks in window that were upticks
 19  down_frac        fraction of ticks in window that were downticks
 20  max_drawdown     largest peak-to-trough decline within window
 21  momentum_accel   (recent 5-tick return) - (prior 5-tick return)

All ratio/return features are dimensionless, so the state generalizes
across the absolute price level of the underlying instrument.
"""

from __future__ import annotations

import numpy as np

STATE_SIZE = 22
_EPS = 1e-8


def state_size() -> int:
    return STATE_SIZE


def build_state(
    timestamps: np.ndarray,
    prices: np.ndarray,
    index: int,
    lookback: int,
) -> np.ndarray | None:
    """
    Build the 22-feature state vector for `index`, using the
    `lookback` quotes ending at (and including) `index`.

    Returns None if there isn't enough history yet.
    """

    if index < lookback - 1 or index >= len(prices):
        return None

    start = index - lookback + 1
    ts = timestamps[start : index + 1]
    px = prices[start : index + 1]

    if len(px) < lookback or len(px) < 22:
        # Need enough points for the longest lookback-based feature
        # (ret_20) to be well defined.
        return None

    current_price = float(px[-1])

    # ------------------------------------------------------------
    # Log returns of the raw tick series
    # ------------------------------------------------------------
    log_px = np.log(px)
    tick_returns = np.diff(log_px)  # length lookback - 1

    def ret_back(n: int) -> float:
        if n >= len(px):
            n = len(px) - 1
        return float(log_px[-1] - log_px[-1 - n])

    ret_1 = ret_back(1)
    ret_2 = ret_back(2)
    ret_5 = ret_back(5)
    ret_10 = ret_back(10)
    ret_20 = ret_back(20)
    ret_full = float(log_px[-1] - log_px[0])

    # ------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------
    vol_full = float(np.std(tick_returns))
    recent_window = tick_returns[-10:] if len(tick_returns) >= 10 else tick_returns
    vol_recent = float(np.std(recent_window))
    vol_ratio = vol_recent / (vol_full + _EPS)

    # ------------------------------------------------------------
    # Position within window
    # ------------------------------------------------------------
    price_mean = float(np.mean(px))
    price_std = float(np.std(px))
    zscore = (current_price - price_mean) / (price_std + _EPS)

    price_min = float(np.min(px))
    price_max = float(np.max(px))
    minmax_pos = (current_price - price_min) / (price_max - price_min + _EPS) - 0.5

    # ------------------------------------------------------------
    # EMA crossovers
    # ------------------------------------------------------------
    def ema(values: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        e = values[0]
        for v in values[1:]:
            e = alpha * v + (1.0 - alpha) * e
        return float(e)

    ema_fast = ema(px, span=5)
    ema_mid = ema(px, span=15)
    ema_slow = ema(px, span=min(30, len(px)))

    ema_fast_dev = (ema_fast - current_price) / (current_price + _EPS)
    ema_cross_fs = (ema_fast - ema_mid) / (current_price + _EPS)
    ema_cross_sl = (ema_mid - ema_slow) / (current_price + _EPS)

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------
    gains = np.clip(tick_returns, a_min=0.0, a_max=None)
    losses = -np.clip(tick_returns, a_min=None, a_max=0.0)

    avg_gain = float(np.mean(gains)) if len(gains) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0

    if avg_loss < _EPS:
        rsi = 100.0 if avg_gain > _EPS else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi_norm = (rsi - 50.0) / 50.0

    # ------------------------------------------------------------
    # Timing / feed regularity
    # ------------------------------------------------------------
    dt = np.diff(ts)
    mean_dt = float(np.mean(dt)) if len(dt) else 0.0
    std_dt = float(np.std(dt)) if len(dt) else 0.0
    last_dt = float(dt[-1]) if len(dt) else 0.0

    # ------------------------------------------------------------
    # Tick direction balance
    # ------------------------------------------------------------
    up_frac = float(np.mean(tick_returns > 0)) if len(tick_returns) else 0.0
    down_frac = float(np.mean(tick_returns < 0)) if len(tick_returns) else 0.0

    # ------------------------------------------------------------
    # Drawdown within window
    # ------------------------------------------------------------
    running_max = np.maximum.accumulate(px)
    drawdowns = (px - running_max) / (running_max + _EPS)
    max_drawdown = float(np.min(drawdowns))

    # ------------------------------------------------------------
    # Momentum acceleration
    # ------------------------------------------------------------
    recent_5 = ret_back(5)
    prior_5 = float(log_px[-6] - log_px[-11]) if len(px) >= 11 else 0.0
    momentum_accel = recent_5 - prior_5

    state = np.array(
        [
            ret_1,
            ret_2,
            ret_5,
            ret_10,
            ret_20,
            ret_full,
            vol_full,
            vol_recent,
            vol_ratio,
            zscore,
            minmax_pos,
            ema_fast_dev,
            ema_cross_fs,
            ema_cross_sl,
            rsi_norm,
            mean_dt,
            std_dt,
            last_dt,
            up_frac,
            down_frac,
            max_drawdown,
            momentum_accel,
        ],
        dtype=np.float32,
    )

    state = np.nan_to_num(
        state,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return state
