import csv
import json
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

PLATFORM_URL = "https://olymptrade.com/platform"
PROFILE_DIR = "./browser_profile"

PAIR = "ASIA_X"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SESSION_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_PATH = DATA_DIR / f"{PAIR}_{SESSION_TIME}.csv"


# ============================================================
# QUOTE PARSER
# ============================================================

def parse_quote(payload):
    """
    Parse OlympTrade WebSocket quote packets.

    Expected structure:
        e == 1
        d == list
        item.p == ASIA_X
        item.t == timestamp
        item.q == price
    """

    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")

        data = json.loads(payload)

        if not isinstance(data, list):
            data = [data]

        for event in data:

            if not isinstance(event, dict):
                continue

            if event.get("e") != 1:
                continue

            data_field = event.get("d")

            if not isinstance(data_field, list):
                continue

            for item in data_field:

                if not isinstance(item, dict):
                    continue

                if item.get("p") != PAIR:
                    continue

                timestamp = item.get("t")
                price = item.get("q")

                if timestamp is None or price is None:
                    continue

                return float(timestamp), float(price)

    except Exception:
        return None

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    quote_index = 0

    print()
    print("=" * 70)
    print("ASIA_X QUOTE LOGGER")
    print("=" * 70)
    print(f"Pair:        {PAIR}")
    print(f"Platform:    {PLATFORM_URL}")
    print(f"CSV:         {CSV_PATH}")
    print(f"Profile:     {PROFILE_DIR}")
    print()
    print("No trading logic.")
    print("No BUY/SELL logic.")
    print("No buttons are clicked.")
    print("Only raw ASIA_X quotes are recorded.")
    print("=" * 70)

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        # EXACTLY four columns.
        writer.writerow([
            "quote_index",
            "timestamp",
            "price",
            "pair",
        ])

        csv_file.flush()

        with sync_playwright() as p:

            # Opens its own Chromium/Chrome-compatible browser profile.
            # It does NOT connect to an already-running Chrome.
            context = p.chromium.launch_persistent_context(
                user_data_dir=PROFILE_DIR,
                headless=False,
            )

            try:

                page = (
                    context.pages[0]
                    if context.pages
                    else context.new_page()
                )

                def websocket_handler(ws):

                    def frame_handler(payload):

                        nonlocal quote_index

                        try:
                            result = parse_quote(payload)

                            if result is None:
                                return

                            timestamp, price = result

                            quote_index += 1

                            writer.writerow([
                                quote_index,
                                f"{timestamp:.3f}",
                                f"{price:.6f}",
                                PAIR,
                            ])

                            csv_file.flush()

                            #print(
                             #   f"[QUOTE] "
                              #  f"{quote_index:8d} | "
                               # f"{timestamp:.3f} | "
                                #f"{price:.6f} | "
                                #f"{PAIR}"
                            #)

                        except Exception as e:
                            print(
                                f"[QUOTE ERROR] "
                                f"{type(e).__name__}: {e}"
                            )

                    ws.on("framereceived", frame_handler)

                page.on("websocket", websocket_handler)

                print()
                print("Opening OlympTrade platform...")
                page.goto(
                    PLATFORM_URL,
                    wait_until="domcontentloaded",
                )

                print(f"TITLE: {page.title()}")
                print()
                print("Listening for ASIA_X quotes...")
                print("CSV is being written continuously.")
                print("Press Ctrl+C to stop.")
                print()

                while True:
                    page.wait_for_timeout(100)

            except KeyboardInterrupt:
                print()
                print("Stopping quote logger...")

            finally:
                context.close()

    print()
    print("=" * 70)
    print("QUOTE LOGGING COMPLETE")
    print("=" * 70)
    print(f"Quotes recorded: {quote_index}")
    print(f"CSV:             {CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
