from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl.config import DATA_DIR, MODEL_DIR, WIN_REWARD, LOSS_REWARD
from rl.data import load_all_segments
from rl.multi_horizon_agent_v3 import (
    HORIZONS,
    NUM_HORIZONS,
    MultiHorizonDQNAgentV3,
)
from rl.multi_horizon_dataset_v3 import MultiHorizonOfflineDataset
from rl.state import state_size


MODEL_PATH = MODEL_DIR / "multi_horizon_dqn_v3.pt"

# IMPORTANT:
# Segments 0..20 were used for training.
# Segments 21..25 are completely unseen.
TRAIN_SEGMENT_END = 21

BATCH_SIZE = 1024

BREAK_EVEN_WR = 1.0 / (1.0 + WIN_REWARD)

ACTION_WAIT = 0
ACTION_BUY = 1
ACTION_SELL = 2


def action_name(horizon: float, action: int) -> str:
    if action == ACTION_WAIT:
        return "WAIT"
    if action == ACTION_BUY:
        return f"BUY_{int(horizon)}s"
    if action == ACTION_SELL:
        return f"SELL_{int(horizon)}s"
    raise ValueError(f"Invalid action: {action}")


def evaluate_policy(
    q_values: np.ndarray,
    rewards: np.ndarray,
    horizon_index: int,
) -> dict:
    """
    Evaluate one horizon independently.

    Policy:
        - choose BUY if predicted BUY reward is the best
          directional reward AND > 0
        - choose SELL if predicted SELL reward is the best
          directional reward AND > 0
        - otherwise WAIT

    The zero threshold is fixed by the reward system.
    It is NOT tuned on the test set.
    """

    q_buy = q_values[:, horizon_index, ACTION_BUY]
    q_sell = q_values[:, horizon_index, ACTION_SELL]

    buy_is_best = q_buy >= q_sell

    best_direction_q = np.maximum(q_buy, q_sell)

    actions = np.full(
        len(q_values),
        ACTION_WAIT,
        dtype=np.int64,
    )

    actions[
        buy_is_best & (best_direction_q > 0.0)
    ] = ACTION_BUY

    actions[
        (~buy_is_best) & (best_direction_q > 0.0)
    ] = ACTION_SELL

    rows = np.arange(len(actions))

    selected_rewards = rewards[
        rows,
        horizon_index,
        actions,
    ]

    trades = actions != ACTION_WAIT
    wins = selected_rewards == WIN_REWARD
    losses = selected_rewards == LOSS_REWARD

    trade_count = int(np.sum(trades))
    win_count = int(np.sum(wins & trades))
    loss_count = int(np.sum(losses & trades))
    wait_count = int(np.sum(~trades))

    total_reward = float(np.sum(selected_rewards))

    if trade_count > 0:
        win_rate = win_count / trade_count
        ev_per_trade = total_reward / trade_count
    else:
        win_rate = 0.0
        ev_per_trade = 0.0

    return {
        "actions": actions,
        "selected_rewards": selected_rewards,
        "best_direction_q": best_direction_q,
        "trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "waits": wait_count,
        "win_rate": win_rate,
        "total_reward": total_reward,
        "ev_per_trade": ev_per_trade,
    }


def calculate_drawdown(rewards: np.ndarray) -> float:
    if len(rewards) == 0:
        return 0.0

    equity = np.cumsum(rewards)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max

    return float(np.min(drawdown))


def print_policy_result(
    horizon: float,
    result: dict,
    total_states: int,
) -> None:
    print(
        f"{int(horizon):>3}s | "
        f"trades {result['trades']:>7,} | "
        f"W {result['wins']:>6,} | "
        f"L {result['losses']:>6,} | "
        f"wait {result['waits']:>6,} | "
        f"WR {result['win_rate'] * 100:>6.2f}% | "
        f"reward {result['total_reward']:>10.2f} | "
        f"EV/trade {result['ev_per_trade']:>8.4f} | "
        f"trade% {result['trades'] / total_states * 100:>6.2f}%"
    )


def evaluate_fixed_baseline(
    rewards: np.ndarray,
    horizon_index: int,
    action: int,
) -> dict:
    selected_rewards = rewards[
        :,
        horizon_index,
        action,
    ]

    wins = selected_rewards == WIN_REWARD
    losses = selected_rewards == LOSS_REWARD

    trade_count = len(selected_rewards)
    win_count = int(np.sum(wins))
    loss_count = int(np.sum(losses))

    total_reward = float(np.sum(selected_rewards))

    return {
        "trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": (
            win_count / trade_count
            if trade_count
            else 0.0
        ),
        "reward": total_reward,
        "ev": (
            total_reward / trade_count
            if trade_count
            else 0.0
        ),
    }


def collect_segment_samples(
    dataset: MultiHorizonOfflineDataset,
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    rewards = []

    for i in range(len(dataset)):
        sample = dataset.make_sample(i)

        states.append(sample.state)
        rewards.append(sample.rewards)

    if not states:
        return (
            np.empty(
                (0, state_size()),
                dtype=np.float32,
            ),
            np.empty(
                (0, NUM_HORIZONS, 3),
                dtype=np.float32,
            ),
        )

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
    )


@torch.no_grad()
def predict_in_batches(
    agent: MultiHorizonDQNAgentV3,
    states: np.ndarray,
) -> np.ndarray:
    outputs = []

    for start in range(
        0,
        len(states),
        BATCH_SIZE,
    ):
        batch = states[
            start : start + BATCH_SIZE
        ]

        q = agent.q_values(batch)

        outputs.append(q)

    if not outputs:
        return np.empty(
            (0, NUM_HORIZONS, 3),
            dtype=np.float32,
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


def print_confidence_report(
    q_values: np.ndarray,
    results: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("CONFIDENCE / SCORE ANALYSIS")
    print("=" * 100)

    for h, horizon in enumerate(HORIZONS):
        result = results[h]

        trade_mask = (
            result["actions"] != ACTION_WAIT
        )

        scores = result["best_direction_q"][
            trade_mask
        ]

        selected_rewards = result[
            "selected_rewards"
        ][trade_mask]

        if len(scores) == 0:
            print(
                f"{int(horizon):>3}s | "
                "NO TRADES"
            )
            continue

        bins = [
            (0.00, 0.05),
            (0.05, 0.10),
            (0.10, 0.20),
            (0.20, 0.40),
            (0.40, float("inf")),
        ]

        print()
        print(f"HORIZON {int(horizon)}s")

        for low, high in bins:
            mask = scores >= low

            if high != float("inf"):
                mask &= scores < high

            count = int(np.sum(mask))

            if count == 0:
                continue

            wr = (
                np.sum(
                    selected_rewards[mask]
                    == WIN_REWARD
                )
                / count
            )

            reward = float(
                np.sum(
                    selected_rewards[mask]
                )
            )

            ev = reward / count

            high_text = (
                "inf"
                if high == float("inf")
                else f"{high:.2f}"
            )

            print(
                f"  score [{low:.2f}, {high_text}) | "
                f"trades {count:>6,} | "
                f"WR {wr * 100:>6.2f}% | "
                f"EV {ev:>7.4f}"
            )


def print_segment_report(
    segment_results: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("UNSEEN SEGMENT PERFORMANCE")
    print("=" * 100)

    header = (
        "SEGMENT | QUOTES | "
        + " | ".join(
            f"{int(h):>5}s WR"
            for h in HORIZONS
        )
    )

    print(header)
    print("-" * len(header))

    for row in segment_results:
        values = []

        for h in range(NUM_HORIZONS):
            result = row["results"][h]

            if result["trades"] == 0:
                values.append("  N/A")
            else:
                values.append(
                    f"{result['win_rate'] * 100:6.2f}%"
                )

        print(
            f"{row['segment_id']:>7} | "
            f"{row['quotes']:>6,} | "
            + " | ".join(values)
        )


def main() -> None:
    print("=" * 100)
    print("MULTI-HORIZON V3 — TRUE UNSEEN WALK-FORWARD EVALUATION")
    print("=" * 100)

    print()
    print("MODEL:")
    print(f"  {MODEL_PATH}")

    print()
    print("RULES:")
    print("  Training segments : 0..20")
    print("  Unseen segments   : 21..25")
    print("  Test data enters  : evaluation only")
    print("  Retraining        : NONE")
    print("  Test tuning       : NONE")
    print(
        f"  Break-even WR     : "
        f"{BREAK_EVEN_WR * 100:.2f}%"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V3 model not found:\n{MODEL_PATH}"
        )

    all_segments = load_all_segments(
        DATA_DIR
    )

    print()
    print(
        f"Total segments available: "
        f"{len(all_segments)}"
    )

    if len(all_segments) <= TRAIN_SEGMENT_END:
        raise RuntimeError(
            "No unseen segments available."
        )

    unseen_segments = all_segments[
        TRAIN_SEGMENT_END:
    ]

    print(
        f"Unseen segments selected: "
        f"{TRAIN_SEGMENT_END}.."
        f"{len(all_segments) - 1}"
    )

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    agent = MultiHorizonDQNAgentV3(
        state_dim=state_size()
    )

    agent.load(MODEL_PATH)

    print()
    print(
        f"Model loaded successfully."
    )
    print(
        f"Training steps in checkpoint: "
        f"{agent.training_steps:,}"
    )

    # ------------------------------------------------------------------
    # BUILD UNSEEN DATA
    # ------------------------------------------------------------------

    all_states = []
    all_rewards = []
    segment_ranges = []

    total_states = 0

    print()
    print("=" * 100)
    print("BUILDING UNSEEN EVALUATION SET")
    print("=" * 100)

    for segment_index, segment in enumerate(
        unseen_segments,
        start=TRAIN_SEGMENT_END,
    ):
        dataset = MultiHorizonOfflineDataset(
            timestamps=segment.timestamps,
            prices=segment.prices,
            lookback=30,
        )

        states, rewards = collect_segment_samples(
            dataset
        )

        start = total_states
        end = start + len(states)

        all_states.append(states)
        all_rewards.append(rewards)

        segment_ranges.append(
            {
                "segment_id": segment_index,
                "source_file": segment.source_file,
                "quotes": segment.size,
                "start": start,
                "end": end,
            }
        )

        total_states = end

        print(
            f"Segment {segment_index:02d} | "
            f"{segment.source_file} | "
            f"quotes {segment.size:>7,} | "
            f"valid states {len(states):>7,}"
        )

    if total_states == 0:
        raise RuntimeError(
            "No valid unseen states were created."
        )

    states = np.concatenate(
        all_states,
        axis=0,
    )

    rewards = np.concatenate(
        all_rewards,
        axis=0,
    )

    print("-" * 100)
    print(
        f"TOTAL UNSEEN STATES: "
        f"{len(states):,}"
    )

    # ------------------------------------------------------------------
    # MODEL PREDICTIONS
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("RUNNING MODEL — NO TRAINING")
    print("=" * 100)

    q_values = predict_in_batches(
        agent,
        states,
    )

    print(
        f"Q-value shape: {q_values.shape}"
    )

    # ------------------------------------------------------------------
    # PER-HORIZON EVALUATION
    # ------------------------------------------------------------------

    results = []

    print()
    print("=" * 100)
    print("PER-HORIZON POLICY RESULTS")
    print("=" * 100)

    print(
        "Policy: directional Q > 0 => trade, "
        "otherwise WAIT"
    )

    print()

    for h, horizon in enumerate(HORIZONS):
        result = evaluate_policy(
            q_values=q_values,
            rewards=rewards,
            horizon_index=h,
        )

        results.append(result)

        print_policy_result(
            horizon=horizon,
            result=result,
            total_states=len(states),
        )

    # ------------------------------------------------------------------
    # BASELINES
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("FIXED DIRECTIONAL BASELINES — UNSEEN DATA")
    print("=" * 100)

    print(
        "These baselines trade every valid state."
    )

    for h, horizon in enumerate(HORIZONS):
        buy = evaluate_fixed_baseline(
            rewards,
            h,
            ACTION_BUY,
        )

        sell = evaluate_fixed_baseline(
            rewards,
            h,
            ACTION_SELL,
        )

        print()
        print(f"{int(horizon)}s:")

        print(
            f"  ALWAYS BUY  | "
            f"WR {buy['win_rate'] * 100:6.2f}% | "
            f"reward {buy['reward']:10.2f} | "
            f"EV {buy['ev']:8.4f}"
        )

        print(
            f"  ALWAYS SELL | "
            f"WR {sell['win_rate'] * 100:6.2f}% | "
            f"reward {sell['reward']:10.2f} | "
            f"EV {sell['ev']:8.4f}"
        )

    # ------------------------------------------------------------------
    # DRAW DOWN
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("MODEL DRAWDOWN")
    print("=" * 100)

    for h, horizon in enumerate(HORIZONS):
        result = results[h]

        dd = calculate_drawdown(
            result["selected_rewards"]
        )

        print(
            f"{int(horizon):>3}s | "
            f"max drawdown {dd:>10.2f}"
        )

    # ------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------

    print_confidence_report(
        q_values=q_values,
        results=results,
    )

    # ------------------------------------------------------------------
    # SEGMENT-BY-SEGMENT
    # ------------------------------------------------------------------

    segment_results = []

    for row in segment_ranges:
        start = row["start"]
        end = row["end"]

        segment_q = q_values[start:end]
        segment_rewards = rewards[start:end]

        segment_policy_results = []

        for h in range(NUM_HORIZONS):
            result = evaluate_policy(
                q_values=segment_q,
                rewards=segment_rewards,
                horizon_index=h,
            )

            segment_policy_results.append(
                result
            )

        segment_results.append(
            {
                **row,
                "results": segment_policy_results,
            }
        )

    print_segment_report(
        segment_results
    )

    # ------------------------------------------------------------------
    # GLOBAL BEST-HORIZON POLICY
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("GLOBAL MULTI-HORIZON POLICY")
    print("=" * 100)

    print(
        "For each state:"
    )
    print(
        "  1. Look at all 12 directional outputs."
    )
    print(
        "  2. Select the highest predicted directional reward."
    )
    print(
        "  3. Trade only if that prediction > 0."
    )
    print(
        "  4. Otherwise WAIT."
    )

    directional_q = q_values[
        :,
        :,
        1:3,
    ]

    flat_indices = np.argmax(
        directional_q.reshape(
            len(states),
            -1,
        ),
        axis=1,
    )

    best_q = np.max(
        directional_q.reshape(
            len(states),
            -1,
        ),
        axis=1,
    )

    global_actions = np.full(
        len(states),
        ACTION_WAIT,
        dtype=np.int64,
    )

    trade_mask = best_q > 0.0

    chosen_flat = flat_indices[
        trade_mask
    ]

    chosen_heads = (
        chosen_flat // 2
    )

    chosen_direction = (
        chosen_flat % 2
    ) + 1

    global_actions[
        trade_mask
    ] = chosen_direction

    trade_indices = np.where(
        trade_mask
    )[0]

    global_rewards = np.zeros(
        len(states),
        dtype=np.float32,
    )

    if len(trade_indices) > 0:
        global_rewards[
            trade_indices
        ] = rewards[
            trade_indices,
            chosen_heads,
            chosen_direction,
        ]

    global_trades = int(
        np.sum(trade_mask)
    )

    global_waits = (
        len(states) - global_trades
    )

    global_wins = int(
        np.sum(
            global_rewards
            == WIN_REWARD
        )
    )

    global_losses = int(
        np.sum(
            global_rewards
            == LOSS_REWARD
        )
    )

    global_total_reward = float(
        np.sum(global_rewards)
    )

    global_wr = (
        global_wins / global_trades
        if global_trades
        else 0.0
    )

    global_ev = (
        global_total_reward
        / global_trades
        if global_trades
        else 0.0
    )

    print()
    print(
        f"Trades       : {global_trades:,}"
    )
    print(
        f"Waits        : {global_waits:,}"
    )
    print(
        f"Wins         : {global_wins:,}"
    )
    print(
        f"Losses       : {global_losses:,}"
    )
    print(
        f"Win rate     : {global_wr * 100:.2f}%"
    )
    print(
        f"Total reward : {global_total_reward:.2f}"
    )
    print(
        f"EV / trade   : {global_ev:.4f}"
    )
    print(
        f"Trade rate   : "
        f"{global_trades / len(states) * 100:.2f}%"
    )
    print(
        f"Break-even   : "
        f"{BREAK_EVEN_WR * 100:.2f}%"
    )

    # Horizon selection distribution.
    if global_trades > 0:
        print()
        print("SELECTED HORIZON DISTRIBUTION:")

        for h, horizon in enumerate(HORIZONS):
            count = int(
                np.sum(
                    chosen_heads == h
                )
            )

            print(
                f"  {int(horizon):>3}s : "
                f"{count:>7,} "
                f"({count / global_trades * 100:6.2f}%)"
            )

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("V3 UNSEEN TEST VERDICT")
    print("=" * 100)

    positive_horizons = 0

    for h, horizon in enumerate(HORIZONS):
        result = results[h]

        if (
            result["trades"] > 0
            and result["win_rate"]
            >= BREAK_EVEN_WR
            and result["ev_per_trade"] > 0
        ):
            positive_horizons += 1

    print(
        f"Horizons above break-even: "
        f"{positive_horizons}/{NUM_HORIZONS}"
    )

    print()

    if global_trades == 0:
        print(
            "RESULT: MODEL CHOOSES WAIT ALMOST EVERYWHERE."
        )
        print(
            "The learned directional rewards did not exceed zero "
            "on the unseen data."
        )

    elif (
        global_wr >= BREAK_EVEN_WR
        and global_ev > 0
    ):
        print(
            "RESULT: POSITIVE UNSEEN PERFORMANCE."
        )
        print(
            "The global multi-horizon policy exceeded "
            "the mathematical break-even threshold."
        )

    else:
        print(
            "RESULT: NO PROFITABLE UNSEEN EDGE."
        )
        print(
            "The model did not beat the break-even requirement "
            "on its global policy."
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This evaluation used segments 21..25 only."
    )
    print(
        "No unseen labels were used to train or modify the model."
    )
    print(
        "A positive result here is evidence of out-of-sample "
        "generalization, not proof of live profitability."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()