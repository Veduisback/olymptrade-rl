from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rl.config import DATA_DIR, LOOKBACK_QUOTES, MODEL_DIR
from rl.multi_horizon_agent_v3 import MultiHorizonDQNAgentV3
from rl.multi_horizon_dataset_v3 import (
    HORIZONS,
    load_training_datasets,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_SEGMENTS_END = 21

BATCH_SIZE = 256
EPOCHS_PER_SEGMENT = 2

LEARNING_RATE = 3e-4

TARGET_UPDATE_EVERY = 250

MODEL_PATH = (
    MODEL_DIR / "multi_horizon_dqn_v3.pt"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 78)
    print(
        "MULTI-HORIZON DQN V3 — "
        "DIRECT ALL-HORIZON OFFLINE LEARNING"
    )
    print("=" * 78)
    print()

    print(
        "Each valid state trains all six horizon heads."
    )

    print(
        "Future prices are labels only; "
        "they never enter the state."
    )

    print()

    print(
        "Horizons: "
        + ", ".join(
            f"{h:g}s"
            for h in HORIZONS
        )
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs/segment: {EPOCHS_PER_SEGMENT}"
    )

    print(
        f"Training segments: "
        f"0..{TRAIN_SEGMENTS_END - 1}"
    )

    print(
        "UNSEEN segments: 21..25"
    )

    print()

    # --------------------------------------------------------
    # Load training datasets
    # --------------------------------------------------------

    datasets = load_training_datasets(
        data_dir=DATA_DIR,
        train_segment_end=TRAIN_SEGMENTS_END,
    )

    if not datasets:
        raise RuntimeError(
            "No training datasets were created."
        )

    # --------------------------------------------------------
    # Create agent
    # --------------------------------------------------------

    agent = MultiHorizonDQNAgentV3(
        state_dim=22,
        learning_rate=LEARNING_RATE,
    )

    if MODEL_PATH.exists():

        print(
            f"Existing V3 model found:"
        )

        print(
            f"  {MODEL_PATH}"
        )

        agent.load(
            MODEL_PATH
        )

        print(
            f"Existing training steps: "
            f"{agent.training_steps:,}"
        )

    else:

        print(
            "No V3 model found. "
            "Starting from scratch."
        )

    print("-" * 78)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_samples = 0
    total_updates = 0

    horizon_wins = np.zeros(
        len(HORIZONS),
        dtype=np.int64,
    )

    horizon_losses = np.zeros(
        len(HORIZONS),
        dtype=np.int64,
    )

    # ========================================================
    # SEGMENT LOOP
    # ========================================================

    for segment_number, (
        segment,
        dataset,
    ) in enumerate(
        datasets,
        start=1,
    ):

        print()
        print("=" * 78)

        print(
            f"SEGMENT {segment_number}/"
            f"{len(datasets)}"
        )

        print("=" * 78)

        print(
            f"File: {segment.source_file}"
        )

        print(
            f"Segment ID: {segment.segment_id}"
        )

        print(
            f"Quotes: {segment.size:,}"
        )

        print(
            f"Valid states: {len(dataset):,}"
        )

        if len(dataset) == 0:

            print(
                "No valid states. Skipping."
            )

            continue

        # ----------------------------------------------------
        # Build samples
        # ----------------------------------------------------

        print(
            "Building offline samples..."
        )

        states = []
        rewards = []
        next_states = []
        dones = []

        for position in range(
            len(dataset)
        ):

            sample = dataset.make_sample(
                position
            )

            states.append(
                sample.state
            )

            rewards.append(
                sample.rewards
            )

            next_states.append(
                sample.next_states
            )

            dones.append(
                sample.dones
            )

        states = np.asarray(
            states,
            dtype=np.float32,
        )

        rewards = np.asarray(
            rewards,
            dtype=np.float32,
        )

        next_states = np.asarray(
            next_states,
            dtype=np.float32,
        )

        dones = np.asarray(
            dones,
            dtype=np.float32,
        )

        print(
            f"States built: {len(states):,}"
        )

        # ----------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------

        for h in range(
            len(HORIZONS)
        ):

            buy_rewards = rewards[
                :, h, 1
            ]

            sell_rewards = rewards[
                :, h, 2
            ]

            horizon_wins[h] += int(
                np.sum(
                    buy_rewards == 0.85
                )
            )

            horizon_wins[h] += int(
                np.sum(
                    sell_rewards == 0.85
                )
            )

            horizon_losses[h] += int(
                np.sum(
                    buy_rewards == -1.0
                )
            )

            horizon_losses[h] += int(
                np.sum(
                    sell_rewards == -1.0
                )
            )

        # ----------------------------------------------------
        # PyTorch dataset
        # ----------------------------------------------------

        tensor_dataset = TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(dones),
        )

        loader = DataLoader(
            tensor_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        segment_updates = 0

        for epoch in range(
            EPOCHS_PER_SEGMENT
        ):

            epoch_loss = 0.0
            epoch_batches = 0

            for (
                batch_states,
                batch_rewards,
                batch_next_states,
                batch_dones,
            ) in loader:

                loss, td_errors = agent.train_batch(
                    states=batch_states,
                    rewards=batch_rewards,
                    next_states=batch_next_states,
                    dones=batch_dones,
                )

                epoch_loss += loss
                epoch_batches += 1

                segment_updates += 1
                total_updates += 1

                # ------------------------------------------------
                # Periodically synchronize target network.
                # ------------------------------------------------

                if (
                    total_updates
                    % TARGET_UPDATE_EVERY
                    == 0
                ):

                    agent.update_target()

            if epoch_batches > 0:

                mean_loss = (
                    epoch_loss
                    / epoch_batches
                )

            else:

                mean_loss = 0.0

            print(
                f"  Epoch "
                f"{epoch + 1}/"
                f"{EPOCHS_PER_SEGMENT} | "
                f"batches {epoch_batches:,} | "
                f"loss {mean_loss:.6f}"
            )

        total_samples += len(dataset)

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        agent.save(
            MODEL_PATH
        )

        print()

        print(
            f"Segment complete."
        )

        print(
            f"Training updates: "
            f"{segment_updates:,}"
        )

        print(
            f"Total updates: "
            f"{total_updates:,}"
        )

        print(
            f"Model saved:"
        )

        print(
            f"  {MODEL_PATH}"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "V3 TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        f"Total valid states: "
        f"{total_samples:,}"
    )

    print(
        f"Training updates: "
        f"{total_updates:,}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()
    print(
        "TRAINING LABEL STATISTICS"
    )

    print("-" * 78)

    for h, horizon in enumerate(
        HORIZONS
    ):

        wins = int(
            horizon_wins[h]
        )

        losses = int(
            horizon_losses[h]
        )

        trades = wins + losses

        if trades > 0:

            win_rate = (
                wins
                / trades
                * 100.0
            )

            reward = (
                wins * 0.85
                -
                losses
            )

        else:

            win_rate = 0.0
            reward = 0.0

        print(
            f"{horizon:>5g}s | "
            f"trades {trades:>9,} | "
            f"WR {win_rate:>6.2f}% | "
            f"reward {reward:>10.2f}"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are training-label statistics, "
        "NOT out-of-sample performance."
    )

    print(
        "Segments 21..25 were not used."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()