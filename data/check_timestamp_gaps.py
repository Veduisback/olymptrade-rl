from pathlib import Path
import csv

DATA_DIR = Path("data")
GAP_THRESHOLD = 2.0


def check_file(path: Path):
    gaps = []
    quote_count = 0

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        previous_timestamp = None
        previous_row = None

        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp"])
            except (KeyError, ValueError):
                continue

            if previous_timestamp is not None:
                gap = timestamp - previous_timestamp

                if gap > GAP_THRESHOLD:
                    gaps.append({
                        "row": row_number,
                        "previous_row": row_number - 1,
                        "previous_timestamp": previous_timestamp,
                        "timestamp": timestamp,
                        "gap": gap,
                        "previous_price": previous_row.get("price"),
                        "price": row.get("price"),
                    })

            quote_count += 1
            previous_timestamp = timestamp
            previous_row = row

    return gaps, quote_count


def main():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        print("No CSV files found in data/")
        return

    print("=" * 80)
    print("ASIA_X TIMESTAMP GAP CHECK")
    print(f"Threshold: > {GAP_THRESHOLD} seconds")
    print("=" * 80)

    total_gaps = 0
    total_quotes = 0

    for path in files:
        gaps, quote_count = check_file(path)
        total_quotes += quote_count

        print(f"\nFILE: {path.name}")
        print(f"QUOTES: {quote_count}")

        if gaps:
            print(f"GAPS FOUND: {len(gaps)}")

            for gap in gaps:
                print(
                    f"  Rows {gap['previous_row']} -> {gap['row']} | "
                    f"gap={gap['gap']:.3f}s | "
                    f"{gap['previous_timestamp']} -> {gap['timestamp']}"
                )

            total_gaps += len(gaps)
        else:
            print("  OK - no gaps greater than 2 seconds")

    print("\n" + "=" * 80)
    print(f"TOTAL FILES: {len(files)}")
    print(f"TOTAL QUOTES: {total_quotes}")
    print(f"TOTAL GAPS > {GAP_THRESHOLD}s: {total_gaps}")
    print("=" * 80)


if __name__ == "__main__":
    main()
