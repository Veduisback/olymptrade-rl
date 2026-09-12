from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.config import (
    ACTION_BUY,
    ACTION_SELL,
    ACTION_WAIT,
    LOSS_REWARD,
    MAX_TIMESTAMP_GAP,
    WAIT_REWARD,
    WIN_REWARD,
)
from rl.data import load_all_segments
from rl.state import build_state


HORIZONS = (5.0, 15.0, 30.0, 45.0, 60.0, 120.0)


@dataclass
class HorizonSample:
    """
    One offline learning sample.

    state:
        Current 22-feature local market state.

    rewards:
        Shape [6, 3].

        Horizon dimension:
            0 = 5s
            1 = 15s
            2 = 30s
            3 = 45s
            4 = 60s
            5 = 120s

        Action dimension:
            0 = WAIT
            1 = BUY
            2 = SELL

    next_states:
        State at the corresponding horizon expiry.

        Shape: [6, 22]

    dones:
        Shape [6].
        1.0 = terminal
        0.0 = bootstrap from next state
    """

    state: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


def _expiry_index(
    timestamps: np.ndarray,
    entry_index: int,
    horizon: float,
) -> int | None:
    """
    Find the first quote at or after entry_time + horizon.

    Returns None when:
        - the horizon extends beyond the segment
        - a timestamp gap > MAX_TIMESTAMP_GAP is crossed
    """

    target = timestamps[entry_index] + horizon

    # Binary search is much faster than scanning forward.
    expiry = int(
        np.searchsorted(
            timestamps,
            target,
            side="left",
        )
    )

    if expiry >= len(timestamps):
        return None

    # Make sure this horizon does not cross an interruption.
    if expiry > entry_index + 1:
        gaps = np.diff(
            timestamps[entry_index : expiry + 1]
        )

        if np.any(gaps > MAX_TIMESTAMP_GAP):
            return None

    return expiry


def _trade_reward(
    entry_price: float,
    exit_price: float,
    action: int,
) -> float:
    """
    Economic reward for one directional trade.
    """

    if action == ACTION_WAIT:
        return WAIT_REWARD

    if action == ACTION_BUY:
        if exit_price > entry_price:
            return WIN_REWARD

        return LOSS_REWARD

    if action == ACTION_SELL:
        if exit_price < entry_price:
            return WIN_REWARD

        return LOSS_REWARD

    raise ValueError(
        f"Invalid action: {action}"
    )


class MultiHorizonOfflineDataset:
    """
    Offline dataset for direct multi-horizon learning.

    Every valid state produces labels for ALL six horizons.

    Future prices are used only for calculating the training
    rewards. They are NEVER included in the state.

    This means the model can learn independently that a local
    pattern may behave differently at:

        5s
        15s
        30s
        45s
        60s
        120s
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        lookback: int = 30,
    ):
        self.timestamps = np.asarray(
            timestamps,
            dtype=np.float64,
        )

        self.prices = np.asarray(
            prices,
            dtype=np.float64,
        )

        self.lookback = int(lookback)

        if len(self.timestamps) != len(self.prices):
            raise ValueError(
                "timestamps and prices must have the same length"
            )

        if len(self.prices) == 0:
            self.valid_indices = []

            return

        self.valid_indices: list[int] = []

        longest_horizon = max(HORIZONS)

        # We need enough history for the state.
        #
        # We also require the complete 120s horizon to be
        # available so that every sample teaches all six heads.
        for i in range(
            self.lookback,
            len(self.prices),
        ):
            expiry = _expiry_index(
                self.timestamps,
                i,
                longest_horizon,
            )

            if expiry is not None:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def make_sample(
        self,
        position: int,
    ) -> HorizonSample:
        """
        Construct one training sample.
        """

        if position < 0 or position >= len(self.valid_indices):
            raise IndexError(
                f"Sample position {position} out of range "
                f"for dataset of size {len(self.valid_indices)}"
            )

        index = self.valid_indices[position]

        # ----------------------------------------------------
        # Current state
        # ----------------------------------------------------

        state = build_state(
            self.timestamps,
            self.prices,
            index,
            self.lookback,
        )

        if state is None:
            raise RuntimeError(
                f"State unavailable at index {index}"
            )

        state_dim = len(state)
        horizon_count = len(HORIZONS)

        # ----------------------------------------------------
        # Allocate outputs
        # ----------------------------------------------------

        rewards = np.zeros(
            (horizon_count, 3),
            dtype=np.float32,
        )

        next_states = np.zeros(
            (horizon_count, state_dim),
            dtype=np.float32,
        )

        dones = np.ones(
            horizon_count,
            dtype=np.float32,
        )

        entry_price = float(
            self.prices[index]
        )

        # ----------------------------------------------------
        # Build labels for every horizon
        # ----------------------------------------------------

        for h, horizon in enumerate(HORIZONS):

            expiry = _expiry_index(
                self.timestamps,
                index,
                horizon,
            )

            # This should normally never happen because valid_indices
            # requires the longest horizon, but keep the protection.
            if expiry is None:
                continue

            exit_price = float(
                self.prices[expiry]
            )

            # ------------------------------------------------
            # Action rewards
            #
            # [WAIT, BUY, SELL]
            # ------------------------------------------------

            rewards[h, ACTION_WAIT] = WAIT_REWARD

            rewards[h, ACTION_BUY] = _trade_reward(
                entry_price,
                exit_price,
                ACTION_BUY,
            )

            rewards[h, ACTION_SELL] = _trade_reward(
                entry_price,
                exit_price,
                ACTION_SELL,
            )

            # ------------------------------------------------
            # Next state
            # ------------------------------------------------

            next_state = build_state(
                self.timestamps,
                self.prices,
                expiry,
                self.lookback,
            )

            if next_state is None:
                continue

            # We can bootstrap if there is enough data after expiry.
            #
            # build_state() already guarantees sufficient history.
            # We additionally require the expiry not to be the final
            # quote.
            if expiry < len(self.prices) - 1:

                next_states[h] = next_state

                dones[h] = 0.0

        return HorizonSample(
            state=state,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )


def load_training_datasets(
    data_dir: str | Path,
    train_segment_end: int = 21,
) -> list[
    tuple[object, MultiHorizonOfflineDataset]
]:
    """
    Load training segments only.

    Segment numbering:

        0 .. train_segment_end-1

    With the current project configuration:

        0 .. 20 = training
        21 .. 25 = completely unseen

    The unseen segments are never loaded by this function.
    """

    data_dir = Path(data_dir)

    # IMPORTANT:
    #
    # load_csv_segments() expects ONE CSV FILE.
    #
    # load_all_segments() expects the DATA DIRECTORY.
    #
    # V3 receives the directory, so we must use load_all_segments().
    segments = load_all_segments(
        data_dir
    )

    if train_segment_end < 0:
        raise ValueError(
            "train_segment_end must be >= 0"
        )

    if train_segment_end > len(segments):
        raise ValueError(
            f"Requested training segments 0.."
            f"{train_segment_end - 1}, but only "
            f"{len(segments)} segments are available."
        )

    datasets: list[
        tuple[object, MultiHorizonOfflineDataset]
    ] = []

    for segment in segments[:train_segment_end]:

        dataset = MultiHorizonOfflineDataset(
            timestamps=segment.timestamps,
            prices=segment.prices,
            lookback=30,
        )

        datasets.append(
            (
                segment,
                dataset,
            )
        )

    return datasets


def dataset_report(
    datasets: list[
        tuple[object, MultiHorizonOfflineDataset]
    ],
) -> None:
    """
    Print a compact V3 dataset report.
    """

    print()
    print("=" * 75)
    print("V3 OFFLINE DATASET REPORT")
    print("=" * 75)

    total_samples = 0

    for segment, dataset in datasets:

        samples = len(dataset)

        total_samples += samples

        print(
            f"{segment.source_file} | "
            f"segment {segment.segment_id} | "
            f"quotes {segment.size:,} | "
            f"valid states {samples:,}"
        )

    print("-" * 75)
    print(
        f"Training segments : {len(datasets):,}"
    )
    print(
        f"Total valid states: {total_samples:,}"
    )
    print(
        f"Horizons          : "
        f"{', '.join(f'{h:g}s' for h in HORIZONS)}"
    )
    print("=" * 75)