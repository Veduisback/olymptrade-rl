"""
Quote-CSV loading for the ASIA_X multi-horizon RL project.

RECONSTRUCTED FILE
===================
Not present in the delivered archive (see rl/config.py header for the
full explanation). Rebuilt from usage:

    rl/multi_horizon_dataset_v3.py:
        from rl.data import load_all_segments
        segments = load_all_segments(data_dir)
        # each `segment` used as segment.timestamps, segment.prices,
        # segment.source_file, segment.segment_id, segment.size
        #
        # comment: "load_csv_segments() expects ONE CSV FILE.
        #           load_all_segments() expects the DATA DIRECTORY."

    rl/evaluate_multi_horizon_v3.py: same Segment attribute usage.

Design decision (best-effort, flagged clearly):
-------------------------------------------------
The README talks about "segments 0..25" while the shipped data/ folder
only contains 15 raw CSV files, and data/check_timestamp_gaps.py shows
several files DO contain internal timestamp gaps (>2s). The most
sensible reading of "load_csv_segments() expects ONE CSV FILE" (plural
"segments" out of one file) is that a single recording can be split
into multiple gap-free contiguous chunks, each becoming its own
training "segment" -- so no individual segment ever silently contains
a feed dropout in the middle of its price history.

That's what's implemented below: `load_csv_segments` reads one CSV and
splits it wherever the gap between consecutive quotes exceeds
`MAX_TIMESTAMP_GAP`. `load_all_segments` runs this over every CSV in
the data directory (sorted by filename, which sorts chronologically
here) and renumbers `segment_id` sequentially across the whole
dataset, which is what train_multi_horizon_v3.py / dataset_v3.py
assume when they slice `segments[:TRAIN_SEGMENTS_END]`.

If your original file did this differently (e.g. one segment per
file, gaps tolerated), this is an easy one-line change -- see the
comment in `load_all_segments`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rl.config import MAX_TIMESTAMP_GAP


@dataclass
class Segment:
    """
    One contiguous, gap-free run of quotes.
    """

    timestamps: np.ndarray
    prices: np.ndarray
    source_file: str
    segment_id: int

    @property
    def size(self) -> int:
        return len(self.timestamps)


def _load_raw_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one raw quote CSV and return (timestamps, prices), sorted by
    timestamp and de-duplicated.

    Expected columns (see data/paper_trader_v10.py):
        quote_index, timestamp, price, pair
    """

    df = pd.read_csv(path)

    required = {"timestamp", "price"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {sorted(missing)}"
        )

    df = df.dropna(subset=["timestamp", "price"])

    df = df.sort_values("timestamp", kind="mergesort")

    # Duplicate/stale timestamps (repeated ticks with identical time)
    # are collapsed to the last observation at that timestamp.
    df = df.drop_duplicates(subset="timestamp", keep="last")

    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    prices = df["price"].to_numpy(dtype=np.float64)

    return timestamps, prices


def load_csv_segments(
    path: str | Path,
    max_gap: float = MAX_TIMESTAMP_GAP,
) -> list[Segment]:
    """
    Load ONE CSV file and split it into gap-free contiguous segments.

    A new segment starts whenever the gap between two consecutive
    quotes exceeds `max_gap` seconds.

    `segment_id` here is local to this file (0, 1, 2, ...);
    `load_all_segments` renumbers these globally.
    """

    path = Path(path)

    timestamps, prices = _load_raw_csv(path)

    if len(timestamps) == 0:
        return []

    gaps = np.diff(timestamps)
    split_points = np.where(gaps > max_gap)[0] + 1

    chunks_t = np.split(timestamps, split_points)
    chunks_p = np.split(prices, split_points)

    segments: list[Segment] = []

    for local_id, (t_chunk, p_chunk) in enumerate(
        zip(chunks_t, chunks_p)
    ):

        if len(t_chunk) == 0:
            continue

        segments.append(
            Segment(
                timestamps=t_chunk,
                prices=p_chunk,
                source_file=path.name,
                segment_id=local_id,
            )
        )

    return segments


def load_all_segments(data_dir: str | Path) -> list[Segment]:
    """
    Load every *.csv file in `data_dir`, split each into gap-free
    segments, and return them all as one globally-numbered list.

    Files are processed in sorted filename order, which is
    chronological for the "ASIA_X_YYYYMMDD_HHMMSS.csv" naming scheme
    used by data/paper_trader_v10.py.

    NOTE: if you'd rather have exactly one segment per file (ignoring
    internal gaps, and instead relying purely on the per-sample gap
    check already done in multi_horizon_dataset_v3._expiry_index),
    replace the body of this function with a plain loop that calls
    `_load_raw_csv` directly and wraps the whole file in one Segment.
    """

    data_dir = Path(data_dir)

    csv_paths = sorted(data_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )

    all_segments: list[Segment] = []

    for csv_path in csv_paths:
        all_segments.extend(load_csv_segments(csv_path))

    # Renumber sequentially across the whole dataset so that
    # segments[:TRAIN_SEGMENTS_END] / segments[TRAIN_SEGMENTS_END:]
    # slicing in train_multi_horizon_v3.py behaves as documented.
    for global_id, segment in enumerate(all_segments):
        segment.segment_id = global_id

    return all_segments
