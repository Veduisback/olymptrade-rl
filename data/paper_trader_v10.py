"""
paper_trader_v10.py

V10 LIVE-DATA PAPER TRADER — EVIDENCE-GATED EDITION
-----------------------------------------------------
PAPER TRADING ONLY. No broker/order API is used.

WHAT CHANGED FROM V9, AND WHY
------------------------------
I pulled the two files you gave me (paper_trades_v9_ASIA_X_...csv,
346 settled ORIGINAL trades) and actually checked whether any of the
8 "strategies" carried real predictive information.

Results on your v9 data:
  * Overall win rate: 173W / 173L = exactly 50.00%.
  * Split by agreement count (5, 7, 8-of-8): all ~47-54%, no trend.
  * Split by consistency, volatility, anti-chase ratio, every momentum
    window, acceleration, direction: every bucket lands in the 40-59%
    band you'd expect from coin-flip noise at n~85-90 per bucket.
  * A logistic regression trained on all 10 numeric features, tested
    out-of-sample with walk-forward splits: 44.9% average accuracy
    (i.e. *worse* than always guessing the majority class). That's
    what it looks like when a model is fitting noise, not signal.
  * Per exact strategy-combination (direction + which of the 8 voted):
    6 distinct combos, n=35-99 each. Wilson 95% lower-bound win rate
    for every single one sits well below the 54.05% break-even line
    that an 85% payout requires.

Conclusion: on this instrument, at this timeframe, these 8
momentum-derived indicators carry no detectable edge. That's not a
bug to patch — 15s prices on a synthetic/OTC index like ASIA_X behave
close to a random walk, and a payout of 85% on a coin flip is a
structurally losing game (you need >54.05% wins to break even, a
fair coin gives you 50%). No amount of extra indicators bolted onto
the same underlying momentum signals fixes that, and I'm not going
to pretend a rewrite conjures an edge that a walk-forward test says
isn't there.

What V10 actually does differently — real changes, not cosmetic ones:

1. EVIDENCE-GATED ENTRIES (the main change).
   Every trade is tagged with a "combo signature" (direction + exact
   set of strategies that voted). V10 tracks win/loss history per
   combo, warm-started from every past paper_trades_v*_ASIA_X_*.csv
   already sitting in data/ (including the v9 file you gave me).
   A combo may only trade live once it has a statistically
   significant (Wilson lower-bound, 95% one-sided) win rate above the
   54.05% break-even line. Until then it trades in a capped
   EXPLORE phase to gather evidence, and is BLOCKED the moment its
   evidence disproves an edge. Concretely: on your v9 history, this
   gate blocks all 6 previously-seen combos immediately, because none
   of them clear break-even. V10 will not repeat a losing pattern
   just because v9 already found it losing — that's the whole point.

2. CIRCUIT BREAKERS. A session stop-loss, a session stop-win, and a
   consecutive-loss pause. These don't create edge, they cap how much
   a losing streak (which will happen even with a real edge) can cost
   you before a human looks at it again.

3. HONEST REPORTING. The summary prints a per-combo evidence table
   (n, win rate, Wilson lower bound, pass/fail) so you can see exactly
   why a trade was allowed or refused, plus a running note that a
   flat or negative P/L on a fair/negative-EV instrument is the
   expected outcome, not a bug.

4. Two log files only, same as V9: a trade log and a raw-quote log.
   The trade log gets a few extra evidence-gating columns; nothing
   is removed from V9's schema.

If you get a longer stretch of live data and a combo actually clears
the evidence bar, V10 will start trading it — and will stop again the
moment new evidence pulls it back under break-even. That's the most
honest version of "trying to win" I can build here: refuse to bet
on patterns the data says don't work, rather than dress up the same
50/50 signals in more confident-looking code.
"""

import csv
import glob
import json
import math
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from statistics import median

from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

PLATFORM_URL = "https://olymptrade.com/platform"
PROFILE_DIR = "./browser_profile"

PAIR = "ASIA_X"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Main trade
# ------------------------------------------------------------

TRADE_DURATION = 60.0

# CHECK THIS: OlympTrade's payout % can differ by expiry duration.
# 0.85 was what your 15s runs showed. Before your first 60s session,
# confirm the actual payout offered for the 1-minute expiry on
# ASIA_X and update this if it's different — it directly sets
# BREAKEVEN_WIN_RATE below.
PAYOUT = 0.85
STAKE = 1.0

# Break-even win rate required at this payout: 1 / (1 + PAYOUT).
BREAKEVEN_WIN_RATE = 1.0 / (1.0 + PAYOUT)  # 0.540541

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

MIN_AGREEMENT = 4

# ------------------------------------------------------------
# Evidence gating (the core V10 change)
# ------------------------------------------------------------

# One-sided confidence z-score for the Wilson lower bound.
# 1.645 ~= 95% one-sided confidence that the true win rate is at
# least this high.
WILSON_Z = 1.645

# A combo may trade unconditionally (to gather data) until it has
# this many settled samples. After that, it needs its Wilson lower
# bound above BREAKEVEN_WIN_RATE to keep trading.
MIN_COMBO_SAMPLES = 20

# Hard cap on how many EXPLORE (unproven) trades a single combo can
# take in one session before it must have earned the right to
# continue, win or lose. Keeps an unlucky/unlucky-but-real combo from
# exploring forever.
MAX_EXPLORE_TRADES_PER_COMBO = 20

# Once a combo has >= MIN_COMBO_SAMPLES samples, if its Wilson lower
# bound is still below this floor it is blocked from normal trading.
COMBO_BLOCK_FLOOR = BREAKEVEN_WIN_RATE

# A blocked combo would otherwise be frozen forever, since a combo
# that never trades can never accumulate the evidence needed to
# prove (or definitively disprove) itself. Instead, at most once
# every PROBATION_SPACING_SECONDS of market time, let ONE probation
# trade through for that combo purely to keep its evidence current.
#
# This is spaced by elapsed time, not by how many quotes arrived —
# decide() is called on every incoming quote (roughly twice a
# second), so a call-count-based spacing (e.g. "every 40 calls")
# collapses to a probation trade every few seconds during a sustained
# signal, which defeats the gate entirely. Time-based spacing keeps
# probation trades genuinely rare regardless of quote rate.
PROBATION_SPACING_SECONDS = 20 * 60  # one probation trade per combo per ~20 min

# Don't spam the console: only print a given combo's "blocked" line
# again after this many seconds, even if the same signal keeps firing
# on every quote tick.
BLOCK_PRINT_THROTTLE_SECONDS = 15.0

# ------------------------------------------------------------
# Counter (loss-reduction hedge) — off by default, same as V9.
# If enabled, it is ALSO evidence-gated using its own combo tracker.
# ------------------------------------------------------------

COUNTER_ENABLED = False
COUNTER_TRIGGER_SECOND = 10.0
COUNTER_DURATION = 5.0

COUNTER_MIN_ADVERSE_MOVE = 0.00001
COUNTER_MAX_VOLATILITY = 0.30
COUNTER_MIN_CONSISTENCY = 0.70

# ------------------------------------------------------------
# Momentum windows
#
# These were originally 1s/3s/5s/10s, tuned for a 15s expiry
# (i.e. 1/15, 1/5, 1/3, 2/3 of the trade duration). Rather than
# leave them fixed at those absolute seconds while TRADE_DURATION
# changes, they're now derived as the same proportions of whatever
# TRADE_DURATION currently is — so a 60s expiry looks 4s/12s/20s/40s
# back, which is a more sensible lookback for predicting 60s ahead
# than reusing the old 1-10s windows would be. The names are kept
# for minimal code churn elsewhere; they no longer mean literal
# seconds.
# ------------------------------------------------------------

WINDOW_1S = TRADE_DURATION * (1.0 / 15.0)
WINDOW_3S = TRADE_DURATION * (3.0 / 15.0)
WINDOW_5S = TRADE_DURATION * (5.0 / 15.0)
WINDOW_10S = TRADE_DURATION * (10.0 / 15.0)

# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

VOL_PERCENTILE_LOOKBACK = 100

# ------------------------------------------------------------
# Anti-chasing
# ------------------------------------------------------------

ANTI_CHASE_3S_MULTIPLIER = 8.0
ANTI_CHASE_5S_MULTIPLIER = 8.0

# ------------------------------------------------------------
# Cooldown / risk controls
# ------------------------------------------------------------

COOLDOWN = 1.0

# After N consecutive losses (originals only), pause new entries for
# PAUSE_SECONDS to break tilt / a bad regime streak.
CONSECUTIVE_LOSS_PAUSE_THRESHOLD = 6
CONSECUTIVE_LOSS_PAUSE_SECONDS = 120.0

# Session-level circuit breakers (in units of STAKE).
SESSION_STOP_LOSS = -25.0
SESSION_STOP_WIN = 25.0

# Hard cap on total original trades in one session, regardless of P/L.
MAX_SESSION_TRADES = 2000

# ------------------------------------------------------------
# Live click execution (OFF by default)
#
# When enabled, the bot actually clicks Up/Down on the real
# OlympTrade page via Playwright, instead of only logging what it
# would have done. It ONLY ever operates on a demo account — see
# OlympTradeExecutor.preflight() for the hard check.
# ------------------------------------------------------------

CLICK_ENABLED = True

# Fixed test stake in platform currency units. Must sit inside the
# platform's own allowed range (seen as Đ1.00–Đ3,000.00 on this
# account; that range may differ on other accounts/currencies).
TEST_STAKE_AMOUNT = "100"

# Bug tripwire, not a trading limit: two real clicks firing less
# than this many seconds apart is not possible in correct operation
# (TRADE_DURATION + COOLDOWN keeps them naturally spaced far wider
# than this). If it ever happens, something is malfunctioning
# (e.g. a retry loop) and could hammer the broker's servers /trip
# anti-bot detection, so we halt immediately rather than continue.
MIN_CLICK_SPACING_SECONDS = 20.0

# Selectors, confirmed against real OlympTrade markup.
SEL_UP_BUTTON = '[data-test="deal-button-up"]'
SEL_DOWN_BUTTON = '[data-test="deal-button-down"]'
SEL_AMOUNT_INPUT = '[data-test="deal-amount-input"]'
SEL_DURATION_INPUT = '[data-test="deal-duration-input"]'
SEL_BALANCE_TITLE = '[data-test="account-balance-title"]'
SEL_PAYOUT_VALUE = '[data-test="asset-select-button-title-value"]'
SEL_TRADES_TAB_BADGE = '[data-test="sidebar-btn-trading-bar"] .sidebar-menu-vertical__counter'
# Amount and Duration blocks each contain a +/- pair with IDENTICAL
# data-test attributes ("deal-form-input-controls-minus/plus"), so
# every selector for them MUST be scoped to its own anchor or it can
# match the wrong control.
SEL_AMOUNT_ANCHOR = '[data-anchor="trading_deal_amount"]'
SEL_DURATION_ANCHOR = '[data-anchor="trading_deal_duration"]'


class ExecutorHaltError(Exception):
    """Raised when a safety check fails badly enough that the whole
    session should stop rather than continue in an unknown state."""
    pass


def duration_label(seconds):
    """Match the exact text format OlympTrade's duration field uses
    (confirmed from the live page: "1 min", "5 sec", etc.)."""

    if seconds < 60:
        return f"{int(seconds)} sec"

    return f"{int(seconds / 60)} min"


class OlympTradeExecutor:
    """
    Wraps the real button clicks on the live OlympTrade page. Every
    public method here either reads the DOM (safe, read-only) or
    performs exactly the click/typing action it's named for — no
    trading decisions live here, only execution of a decision that's
    already been made elsewhere.
    """

    def __init__(self, page):
        self.page = page
        self.last_click_time = None  # wall-clock time.time(), for
        # the tripwire; deliberately NOT market timestamp, since the
        # tripwire is about real elapsed wall-clock time between
        # actual browser actions.

    # --------------------------------------------------------
    # Preflight — run before every single click, not just once.
    # --------------------------------------------------------

    def preflight(self):
        """
        Returns nothing on success. Raises ExecutorHaltError on any
        failure that means we should not proceed — deliberately
        fails loud rather than trying to be clever about recovering,
        since silently continuing after a failed safety check is
        exactly the failure mode we're trying to avoid.
        """

        try:
            balance_title = self.page.locator(
                SEL_BALANCE_TITLE
            ).inner_text(timeout=5000).strip()
        except Exception as e:
            raise ExecutorHaltError(
                f"Could not read account type before clicking "
                f"(refusing to click blind): {type(e).__name__}: {e}"
            )

        if balance_title != "Demo Account":
            raise ExecutorHaltError(
                f"Account type is '{balance_title}', not "
                f"'Demo Account'. Refusing to click — this bot is "
                f"only authorized to click on a demo account."
            )

        try:
            payout_text = self.page.locator(
                SEL_PAYOUT_VALUE
            ).inner_text(timeout=5000).strip()
            live_payout = float(payout_text.replace("%", "")) / 100.0
        except Exception as e:
            raise ExecutorHaltError(
                f"Could not read live payout %: "
                f"{type(e).__name__}: {e}"
            )

        if abs(live_payout - PAYOUT) > 0.001:
            print(
                f"[EXECUTOR WARNING] Live payout is "
                f"{live_payout:.0%}, but PAYOUT constant is set to "
                f"{PAYOUT:.0%}. Break-even math in the evidence gate "
                f"is now stale — update PAYOUT at the top of the "
                f"file to match."
            )

    def enforce_click_spacing(self):
        now = time.time()

        if self.last_click_time is not None:
            elapsed = now - self.last_click_time

            if elapsed < MIN_CLICK_SPACING_SECONDS:
                raise ExecutorHaltError(
                    f"Click spacing tripwire: only {elapsed:.1f}s "
                    f"since the last click "
                    f"(minimum {MIN_CLICK_SPACING_SECONDS:.0f}s). "
                    f"This should be impossible in normal operation "
                    f"— halting rather than risk hammering the "
                    f"broker's servers from a runaway loop."
                )

    # --------------------------------------------------------
    # Form setup
    # --------------------------------------------------------

    def _read_amount(self):
        return self.page.locator(
            SEL_AMOUNT_INPUT
        ).input_value(timeout=5000).strip()

    def _read_duration(self):
        return self.page.locator(
            SEL_DURATION_INPUT
        ).input_value(timeout=5000).strip()

    def ensure_amount(self, target_amount):
        current = self._read_amount().replace(",", "")

        if current == target_amount:
            return

        field = self.page.locator(SEL_AMOUNT_INPUT)
        field.click()
        field.press("Control+A")
        field.press("Backspace")
        # Type character-by-character: this is a React-controlled
        # input with live validation (a Đ1–3,000 range tooltip
        # fires on invalid values). A single low-level .fill() can
        # update the visible DOM value without reliably firing
        # React's onChange for every keystroke, leaving React's
        # internal state stale even though the field looks right.
        # press_sequentially dispatches real per-character key
        # events, which React's controlled-input listeners pick up
        # correctly.
        field.press_sequentially(target_amount, delay=40)

        readback = self._read_amount().replace(",", "")

        if readback != target_amount:
            raise ExecutorHaltError(
                f"Amount field verification failed: wrote "
                f"'{target_amount}', field now reads '{readback}'. "
                f"Refusing to click with an unverified stake."
            )

    def ensure_duration(self, target_label):
        current = self._read_duration()

        if current == target_label:
            return

        field = self.page.locator(SEL_DURATION_INPUT)
        field.click()
        field.press("Control+A")
        field.press("Backspace")
        field.press_sequentially(target_label, delay=40)
        field.press("Enter")

        readback = self._read_duration()

        if readback != target_label:
            raise ExecutorHaltError(
                f"Duration field verification failed: wanted "
                f"'{target_label}', field now reads '{readback}'. "
                f"Refusing to click with an unverified duration."
            )

    # --------------------------------------------------------
    # The actual click
    # --------------------------------------------------------

    def _trades_badge_count(self):
        try:
            text = self.page.locator(
                SEL_TRADES_TAB_BADGE
            ).inner_text(timeout=1000).strip()
            return int(text) if text else 0
        except Exception:
            return 0

    def click_direction(self, direction, target_amount, target_duration):
        """
        Full sequence for one real trade: preflight, tripwire,
        set amount/duration, click, verify it registered.

        Returns a dict with what we could confirm, for logging
        alongside the paper trade record. Raises ExecutorHaltError
        on anything that means we shouldn't trust the result.
        """

        self.preflight()
        self.enforce_click_spacing()

        self.ensure_amount(target_amount)
        self.ensure_duration(target_duration)

        before_count = self._trades_badge_count()

        selector = SEL_UP_BUTTON if direction == "BUY" else SEL_DOWN_BUTTON
        self.page.locator(selector).click(timeout=5000)
        self.last_click_time = time.time()

        # Poll for the badge to increment rather than assuming.
        # Widened from ~4s to ~10s after seeing a session where
        # confirmation failures clustered hard in one ~30-minute
        # window (74% unconfirmed) versus ~12% elsewhere — that
        # pattern points at transient page/connection lag rather
        # than a genuinely failed click every time, so give slow
        # periods more room before giving up.
        confirmed = False
        last_seen_count = before_count

        for _ in range(50):  # up to ~10s
            self.page.wait_for_timeout(200)
            last_seen_count = self._trades_badge_count()
            if last_seen_count > before_count:
                confirmed = True
                break

        if not confirmed:
            # Grab real diagnostics instead of just guessing why.
            try:
                balance_now = self.page.locator(
                    SEL_BALANCE_TITLE
                ).inner_text(timeout=2000).strip()
            except Exception as e:
                balance_now = f"<unreadable: {e}>"

            try:
                page_visible = not self.page.is_hidden("body")
            except Exception:
                page_visible = "<unknown>"

            print(
                f"[EXECUTOR WARNING] Clicked {direction} but the "
                f"Trades badge never incremented within 10s "
                f"(before={before_count}, last_seen={last_seen_count}). "
                f"Diagnostics: account_label='{balance_now}', "
                f"page_visible={page_visible}. Treat this trade's "
                f"broker-side status as UNKNOWN — cross-check "
                f"OlympTrade's own Trades > History for this "
                f"timestamp to know whether it actually opened."
            )

        return {
            "click_confirmed": confirmed,
            "click_time": self.last_click_time,
        }




# ------------------------------------------------------------
# History
# ------------------------------------------------------------

HISTORY_SECONDS = 2.0 * TRADE_DURATION + WINDOW_10S


# ============================================================
# CSV
# ============================================================

SESSION_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_PATH = (
    DATA_DIR
    / f"paper_trades_v10_{PAIR}_{SESSION_TIME}.csv"
)

LOG_COLUMNS = [
    "trade_id",
    "trade_type",
    "parent_trade_id",
    "direction",
    "entry_timestamp",
    "entry_price",
    "expiry_timestamp",
    "expiry_price",
    "result",
    "pnl",

    "agreement",
    "strategies",

    "momentum_1s",
    "momentum_3s",
    "momentum_5s",
    "momentum_10s",
    "acceleration",

    "volatility",
    "volatility_percentile",
    "consistency",

    "anti_chase_ratio",

    "counter_reason",
    "original_entry_price",
    "original_current_price",
    "original_move",

    # ---- V10 evidence-gating diagnostics ----
    "trade_duration_s",
    "combo_signature",
    "combo_mode",
    "combo_samples_before",
    "combo_wins_before",
    "combo_winrate_before",
    "combo_wilson_lb_before",

    "click_status",

    "note",
]


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
            payload = payload.decode(
                "utf-8",
                errors="replace"
            )

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

                return (
                    float(timestamp),
                    float(price)
                )

    except Exception:
        return None

    return None


# ============================================================
# WILSON SCORE LOWER BOUND
# ============================================================

def wilson_lower_bound(wins, n, z=WILSON_Z):
    """
    One-sided Wilson score lower bound on the true win probability,
    given `wins` successes out of `n` trials. Returns 0.0 if n == 0.

    This is the statistically honest way to ask "is this combo's
    observed win rate believably above break-even, or could it just
    be noise?" A raw win rate of 60% on 5 trades means almost
    nothing; a raw win rate of 58% on 500 trades means a lot. Wilson
    LB accounts for that automatically.
    """

    if n <= 0:
        return 0.0

    phat = wins / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    adj = z * math.sqrt(
        (phat * (1 - phat) / n) + (z * z) / (4 * n * n)
    )

    return (center - adj) / denom


# ============================================================
# COMBO EVIDENCE TRACKER
# ============================================================

class ComboTracker:
    """
    Tracks win/loss evidence per exact strategy-combination signature
    ("BUY:strategyA|strategyB|..."), warm-started from every prior
    paper_trades_v*_{PAIR}_*.csv found in DATA_DIR (this is how your
    v9 log's 346 trades get folded in before a single live quote
    arrives).
    """

    def __init__(self, pair, data_dir, label="combo tracker",
                 trade_type="ORIGINAL"):

        self.label = label
        self.trade_type = trade_type
        self.stats = {}  # signature -> {"wins": int, "n": int}
        self.explore_count = {}  # signature -> int, this session only
        self.last_probation_time = {}  # signature -> market timestamp

        self._warm_start(pair, data_dir)

    def _warm_start(self, pair, data_dir):

        pattern = str(data_dir / f"paper_trades_v*_{pair}_*.csv")
        files = sorted(glob.glob(pattern))

        loaded_trades = 0
        loaded_files = 0

        for path in files:

            try:
                with open(path, "r", newline="", encoding="utf-8") as f:

                    reader = csv.DictReader(f)

                    if "trade_type" not in (reader.fieldnames or []):
                        continue

                    for row in reader:

                        if row.get("trade_type") != self.trade_type:
                            continue

                        result = row.get("result")

                        if result not in ("WIN", "LOSS"):
                            # skip ties / incomplete rows for signal
                            # purposes, they don't inform edge
                            continue

                        # COUNTER rows already carry their own literal
                        # combo_signature ("BUY:COUNTER"/"SELL:
                        # COUNTER"); ORIGINAL rows need the signature
                        # rebuilt from direction + strategies, sorted
                        # to match the live format exactly.
                        if (
                            self.trade_type == "COUNTER"
                            and row.get("combo_signature")
                        ):
                            signature = row["combo_signature"]
                        else:
                            direction = row.get("direction", "")
                            strategies_raw = row.get("strategies", "")

                            # Duration wasn't logged before this
                            # column existed. Every file written
                            # before this change used a 15s expiry,
                            # so absent/blank values default to 15s.
                            # This also means switching TRADE_DURATION
                            # naturally stops old-duration evidence
                            # from mixing into the new duration's
                            # combos, with no manual cleanup needed.
                            duration_raw = row.get("trade_duration_s")
                            try:
                                duration_s = (
                                    float(duration_raw)
                                    if duration_raw
                                    else 15.0
                                )
                            except ValueError:
                                duration_s = 15.0

                            # IMPORTANT: must match the exact
                            # signature format built at live decision
                            # time, which is direction + duration +
                            # alphabetically-sorted strategy names.
                            # The "strategies" column is stored in
                            # vote-insertion order (not sorted), in
                            # both V9 and V10 files, so it must be
                            # re-sorted here or warm start silently
                            # fails to connect to live trading.
                            strategy_list = sorted(
                                s for s in strategies_raw.split("|")
                                if s
                            )
                            signature = (
                                f"{direction}:{duration_s:.0f}s:"
                                + "|".join(strategy_list)
                            )

                        entry = self.stats.setdefault(
                            signature, {"wins": 0, "n": 0}
                        )

                        entry["n"] += 1

                        if result == "WIN":
                            entry["wins"] += 1

                        loaded_trades += 1

                loaded_files += 1

            except Exception as e:
                print(
                    f"[WARM START] Skipped {path}: "
                    f"{type(e).__name__}: {e}"
                )

        if loaded_files:
            print(
                f"[WARM START: {self.label}] Loaded {loaded_trades} "
                f"historical ORIGINAL trades from {loaded_files} "
                f"file(s) into {len(self.stats)} combo signature(s)."
            )
        else:
            print(
                f"[WARM START: {self.label}] No prior trade logs "
                f"found; all combos start with zero evidence."
            )

    def snapshot(self, signature):
        """Return (wins, n, winrate, wilson_lb) BEFORE this trade."""

        entry = self.stats.get(signature, {"wins": 0, "n": 0})
        wins = entry["wins"]
        n = entry["n"]
        winrate = (wins / n) if n else None
        lb = wilson_lower_bound(wins, n)

        return wins, n, winrate, lb

    def decide(self, signature, now):
        """
        `now` is the current market/quote timestamp (seconds), used
        to space out probation trades by real elapsed time rather
        than by how many times decide() happened to be called.

        Returns (allowed: bool, mode: str) where mode is one of:
          "EXPLORE"    - insufficient evidence yet, allowed to trade
                         to gather data (capped)
          "PROVEN"     - evidence clears the break-even bar, allowed
          "PROBATION"  - evidence doesn't clear the bar, but this is
                         the rare periodic trade that keeps its
                         evidence from going stale (see
                         PROBATION_SPACING_SECONDS)
          "BLOCKED"    - evidence says this combo does not clear
                         break-even; refused
          "EXPLORE_CAPPED" - out of explore budget with insufficient
                         evidence either way; refused until it earns
                         more real (non-exploration) evidence
        """

        wins, n, winrate, lb = self.snapshot(signature)

        if n < MIN_COMBO_SAMPLES:

            explored = self.explore_count.get(signature, 0)

            if explored >= MAX_EXPLORE_TRADES_PER_COMBO:
                return False, "EXPLORE_CAPPED"

            return True, "EXPLORE"

        if lb >= COMBO_BLOCK_FLOOR:
            return True, "PROVEN"

        # Below the bar. Rather than freezing this combo forever
        # (it can never earn new evidence if it never trades), allow
        # one probation trade per combo per PROBATION_SPACING_SECONDS
        # of real elapsed time so the estimate stays current and can
        # still recover if the true win rate is genuinely near the
        # line.
        last = self.last_probation_time.get(signature, -1e18)

        if now - last >= PROBATION_SPACING_SECONDS:
            self.last_probation_time[signature] = now
            return True, "PROBATION"

        return False, "BLOCKED"

    def register_explore(self, signature):
        self.explore_count[signature] = (
            self.explore_count.get(signature, 0) + 1
        )

    def record_result(self, signature, won):

        entry = self.stats.setdefault(
            signature, {"wins": 0, "n": 0}
        )
        entry["n"] += 1

        if won:
            entry["wins"] += 1

    def report_rows(self):
        """Sorted (by n desc) list of dicts for the end-of-session
        evidence table."""

        rows = []

        for signature, entry in self.stats.items():

            wins = entry["wins"]
            n = entry["n"]
            winrate = (wins / n) if n else 0.0
            lb = wilson_lower_bound(wins, n)

            rows.append({
                "signature": signature,
                "n": n,
                "wins": wins,
                "winrate": winrate,
                "wilson_lb": lb,
                "passes": lb >= COMBO_BLOCK_FLOOR,
            })

        rows.sort(key=lambda r: r["n"], reverse=True)
        return rows


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTraderV10:

    def __init__(self):

        self.quotes = deque()

        self.active_trade = None
        self.active_counter = None

        self.trade_id = 0

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.signal_count = 0
        self.blocked_signal_count = 0

        self.original_trades = 0
        self.original_wins = 0
        self.original_losses = 0
        self.original_ties = 0
        self.original_pnl = 0.0

        self.counter_trades = 0
        self.counter_wins = 0
        self.counter_losses = 0
        self.counter_ties = 0
        self.counter_pnl = 0.0

        self.total_pnl = 0.0

        self.quotes_seen = 0
        self.last_status_print = 0.0

        self.last_trade_end = -999999999.0

        self.consecutive_losses = 0
        self.paused_until = -1.0
        self.session_halted = False
        self.halt_reason = None

        # Real-click execution. The frame_handler callback (fired by
        # Playwright's WebSocket event dispatch) only ever sets this
        # to a pending intent — it never calls page methods directly,
        # since doing Playwright I/O from inside an event callback
        # that Playwright itself is mid-dispatching is a reentrancy
        # risk. The main loop, which is free to safely call page
        # methods, is the only thing that ever consumes it.
        self.pending_click = None  # dict or None
        self.executor = None  # set by main() if CLICK_ENABLED

        # Rolling volatility observations used for percentile.
        self.volatility_history = deque(
            maxlen=VOL_PERCENTILE_LOOKBACK
        )

        # Throttle repeated "[SIGNAL BLOCKED]" console spam: a
        # rejected signal can recur on every quote tick while the
        # underlying votes are unchanged; only re-print per combo
        # every BLOCK_PRINT_THROTTLE_SECONDS.
        self.last_block_print = {}

        # ----------------------------------------------------
        # Evidence gating
        # ----------------------------------------------------

        self.combo_tracker = ComboTracker(
            PAIR, DATA_DIR, label="original trades",
            trade_type="ORIGINAL"
        )
        self.counter_tracker = ComboTracker(
            PAIR, DATA_DIR, label="counter trades",
            trade_type="COUNTER"
        )

        # ----------------------------------------------------
        # CSV: trade log
        # ----------------------------------------------------

        self.file = open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=LOG_COLUMNS
        )

        self.writer.writeheader()
        self.file.flush()

        # ----------------------------------------------------
        # CSV: raw quote log (unchanged from V9)
        # ----------------------------------------------------

        self.quote_csv_path = (
            DATA_DIR
            / f"quotes_{PAIR}_{SESSION_TIME}.csv"
        )

        self.quote_file = open(
            self.quote_csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.quote_writer = csv.DictWriter(
            self.quote_file,
            fieldnames=[
                "quote_index",
                "timestamp",
                "price",
                "pair",
            ]
        )

        self.quote_writer.writeheader()
        self.quote_file.flush()
        self.quote_index = 0

    # ========================================================
    # BASIC PRICE HELPERS
    # ========================================================

    def price_before(self, target_timestamp):

        for q in reversed(self.quotes):

            if q["timestamp"] <= target_timestamp:
                return q["price"]

        return None

    def current_price(self):

        if not self.quotes:
            return None

        return self.quotes[-1]["price"]

    # ========================================================
    # MOMENTUM
    # ========================================================

    def momentum(self, now, window):

        current = self.current_price()

        old = self.price_before(
            now - window
        )

        if current is None or old is None:
            return None

        return current - old

    # ========================================================
    # DIRECTION CONSISTENCY
    # ========================================================

    def direction_consistency(
        self,
        direction,
        now,
        window=5.0
    ):

        recent = [
            q for q in self.quotes
            if now - window <= q["timestamp"] <= now
        ]

        if len(recent) < 3:
            return None

        positive = 0
        negative = 0

        previous = recent[0]["price"]

        for q in recent[1:]:

            change = q["price"] - previous

            if change > 0:
                positive += 1

            elif change < 0:
                negative += 1

            previous = q["price"]

        total = positive + negative

        if total == 0:
            return 0.0

        if direction == "BUY":
            return positive / total

        return negative / total

    # ========================================================
    # VOLATILITY
    # ========================================================

    def volatility(self):

        if len(self.quotes) < 5:
            return None

        recent = list(self.quotes)[-30:]

        changes = []

        for i in range(1, len(recent)):

            changes.append(
                abs(
                    recent[i]["price"]
                    - recent[i - 1]["price"]
                )
            )

        if not changes:
            return None

        return sum(changes) / len(changes)

    # ========================================================
    # VOLATILITY PERCENTILE
    # ========================================================

    def volatility_percentile(self, value):

        if value is None:
            return None

        if len(self.volatility_history) < 20:
            return None

        values = sorted(self.volatility_history)

        below_or_equal = sum(
            1
            for x in values
            if x <= value
        )

        return (
            100.0
            * below_or_equal
            / len(values)
        )

    def update_volatility_history(self, value):

        if value is None:
            return

        self.volatility_history.append(value)

    # ========================================================
    # SIGNAL STRATEGIES (unchanged logic from V9 — the walk-forward
    # test showed no single indicator carries edge on its own, so
    # rewriting the indicator math would just be theater; the real
    # fix lives in the evidence gate below)
    # ========================================================

    def strategy_mountain_stream(self, m1, m3, m5, acceleration):

        if m3 is None or m5 is None:
            return None

        if m3 > 0 and m5 > 0 and acceleration > 0:
            return "BUY"

        if m3 < 0 and m5 < 0 and acceleration < 0:
            return "SELL"

        return None

    def strategy_ema_stochastic(self, m1, m3, m5):

        if m1 is None or m3 is None or m5 is None:
            return None

        if (
            m3 > 0 and m5 > 0 and m1 > 0
            and m1 >= (m3 / 3.0)
        ):
            return "BUY"

        if (
            m3 < 0 and m5 < 0 and m1 < 0
            and m1 <= (m3 / 3.0)
        ):
            return "SELL"

        return None

    def strategy_bulls_bears_psar(self, m1, m3, acceleration):

        if (
            m1 is not None and m3 is not None
            and m1 > 0 and m3 > 0 and acceleration > 0
        ):
            return "BUY"

        if (
            m1 is not None and m3 is not None
            and m1 < 0 and m3 < 0 and acceleration < 0
        ):
            return "SELL"

        return None

    def strategy_supertrend(self, m5, m10):

        if m5 is None or m10 is None:
            return None

        if m5 > 0 and m10 > 0:
            return "BUY"

        if m5 < 0 and m10 < 0:
            return "SELL"

        return None

    def strategy_alligator_ao(self, m3, m5, m10):

        if m3 is None or m5 is None or m10 is None:
            return None

        if (
            m3 > 0 and m5 > 0 and m10 > 0
            and m3 > (m5 / 2.0)
        ):
            return "BUY"

        if (
            m3 < 0 and m5 < 0 and m10 < 0
            and m3 < (m5 / 2.0)
        ):
            return "SELL"

        return None

    def strategy_safari(self, m1, m3, acceleration):

        if m1 is None or m3 is None or acceleration is None:
            return None

        if m1 > 0 and m3 > 0 and acceleration > 0:
            return "BUY"

        if m1 < 0 and m3 < 0 and acceleration < 0:
            return "SELL"

        return None

    def strategy_heikin_stoch_macd(self, m1, m5, m10):

        if m1 is None or m5 is None or m10 is None:
            return None

        if m1 > 0 and m5 > 0 and m10 > 0:
            return "BUY"

        if m1 < 0 and m5 < 0 and m10 < 0:
            return "SELL"

        return None

    def strategy_microstructure(self, m1, m3, m5, acceleration):

        if (
            m1 is not None and m3 is not None
            and m5 is not None and acceleration is not None
        ):

            if m1 > 0 and m3 > 0 and m5 > 0 and acceleration >= 0:
                return "BUY"

            if m1 < 0 and m3 < 0 and m5 < 0 and acceleration <= 0:
                return "SELL"

        return None

    # ========================================================
    # STRATEGY VOTES
    # ========================================================

    def get_strategy_votes(self, m1, m3, m5, m10, acceleration):

        return {
            "mountain_stream":
                self.strategy_mountain_stream(m1, m3, m5, acceleration),
            "ema_stochastic":
                self.strategy_ema_stochastic(m1, m3, m5),
            "bulls_bears_psar":
                self.strategy_bulls_bears_psar(m1, m3, acceleration),
            "supertrend":
                self.strategy_supertrend(m5, m10),
            "alligator_ao":
                self.strategy_alligator_ao(m3, m5, m10),
            "safari":
                self.strategy_safari(m1, m3, acceleration),
            "heikin_stoch_macd":
                self.strategy_heikin_stoch_macd(m1, m5, m10),
            "microstructure":
                self.strategy_microstructure(m1, m3, m5, acceleration),
        }

    # ========================================================
    # ANTI-CHASE
    # ========================================================

    def _recent_abs_changes(self):

        return [
            abs(q["price"] - self.quotes[i - 1]["price"])
            for i, q in enumerate(self.quotes)
            if i > 0
        ][-30:]

    def anti_chase_ratio(self, m3, m5):

        if m3 is None or m5 is None:
            return None

        recent_abs = self._recent_abs_changes()

        if len(recent_abs) < 10:
            return None

        baseline = median(recent_abs)

        if baseline <= 0:
            return 0.0

        return max(abs(m3) / baseline, abs(m5) / baseline)

    def is_chasing(self, m3, m5):

        ratio = self.anti_chase_ratio(m3, m5)

        if ratio is None:
            return False, None

        recent_abs = self._recent_abs_changes()
        baseline = max(median(recent_abs), 1e-12) if recent_abs else 1e-12

        if abs(m3) > ANTI_CHASE_3S_MULTIPLIER * baseline:
            return True, ratio

        if abs(m5) > ANTI_CHASE_5S_MULTIPLIER * baseline:
            return True, ratio

        return False, ratio

    # ========================================================
    # GENERATE ENTRY SIGNAL
    # ========================================================

    def generate_signal(self):

        if len(self.quotes) < 20:
            return None

        now = self.quotes[-1]["timestamp"]
        price = self.current_price()

        m1 = self.momentum(now, WINDOW_1S)
        m3 = self.momentum(now, WINDOW_3S)
        m5 = self.momentum(now, WINDOW_5S)
        m10 = self.momentum(now, WINDOW_10S)

        if None in (m1, m3, m5, m10):
            return None

        acceleration = m1 - (m3 / 3.0)

        volatility = self.volatility()

        if volatility is None:
            return None

        self.update_volatility_history(volatility)

        vol_percentile = self.volatility_percentile(volatility)

        votes = self.get_strategy_votes(m1, m3, m5, m10, acceleration)

        buy_strategies = [
            name for name, vote in votes.items() if vote == "BUY"
        ]
        sell_strategies = [
            name for name, vote in votes.items() if vote == "SELL"
        ]

        buy_count = len(buy_strategies)
        sell_count = len(sell_strategies)

        if buy_count >= MIN_AGREEMENT and buy_count > sell_count:
            direction = "BUY"
            selected = buy_strategies

        elif sell_count >= MIN_AGREEMENT and sell_count > buy_count:
            direction = "SELL"
            selected = sell_strategies

        else:
            return None

        consistency = self.direction_consistency(direction, now)

        if consistency is None:
            return None

        chasing, chase_ratio = self.is_chasing(m3, m5)

        return {
            "direction": direction,
            "timestamp": now,
            "price": price,

            "m1": m1,
            "m3": m3,
            "m5": m5,
            "m10": m10,
            "acceleration": acceleration,

            "volatility": volatility,
            "volatility_percentile": vol_percentile,

            "consistency": consistency,

            "agreement": len(selected),
            "strategies": selected,

            "anti_chase_ratio": chase_ratio,
        }

    # ========================================================
    # RISK CONTROLS
    # ========================================================

    def risk_controls_allow_entry(self, now):
        """
        Returns (allowed: bool, reason: str-or-None). Checks circuit
        breakers unrelated to any individual signal's evidence.
        """

        if self.session_halted:
            return False, self.halt_reason

        if self.original_trades >= MAX_SESSION_TRADES:
            self.session_halted = True
            self.halt_reason = (
                f"reached MAX_SESSION_TRADES="
                f"{MAX_SESSION_TRADES}"
            )
            return False, self.halt_reason

        if self.total_pnl <= SESSION_STOP_LOSS:
            self.session_halted = True
            self.halt_reason = (
                f"session stop-loss hit "
                f"({self.total_pnl:+.2f} <= {SESSION_STOP_LOSS:+.2f})"
            )
            return False, self.halt_reason

        if self.total_pnl >= SESSION_STOP_WIN:
            self.session_halted = True
            self.halt_reason = (
                f"session stop-win hit "
                f"({self.total_pnl:+.2f} >= {SESSION_STOP_WIN:+.2f})"
            )
            return False, self.halt_reason

        if now < self.paused_until:
            return False, (
                f"cooling off after "
                f"{self.consecutive_losses} consecutive losses "
                f"(resumes at t={self.paused_until:.1f})"
            )

        return True, None

    # ========================================================
    # START ORIGINAL TRADE
    # ========================================================

    def start_trade(self, signal, combo_signature, combo_mode,
                     combo_snapshot):

        self.trade_id += 1
        self.signal_count += 1
        self.original_trades += 1

        expiry_timestamp = signal["timestamp"] + TRADE_DURATION

        self.active_trade = {
            "trade_id": self.trade_id,
            "direction": signal["direction"],
            "entry_timestamp": signal["timestamp"],
            "entry_price": signal["price"],
            "expiry_timestamp": expiry_timestamp,
            "m1": signal["m1"],
            "m3": signal["m3"],
            "m5": signal["m5"],
            "m10": signal["m10"],
            "acceleration": signal["acceleration"],
            "volatility": signal["volatility"],
            "volatility_percentile": signal["volatility_percentile"],
            "consistency": signal["consistency"],
            "agreement": signal["agreement"],
            "strategies": signal["strategies"],
            "anti_chase_ratio": signal["anti_chase_ratio"],
            "counter_checked": False,
            "combo_signature": combo_signature,
            "combo_mode": combo_mode,
            "combo_snapshot": combo_snapshot,
            "click_status": "not_attempted",  # updated by main loop
        }

        wins_b, n_b, winrate_b, lb_b = combo_snapshot

        print()
        print(f"[V10 TRADE #{self.trade_id}] {signal['direction']}"
              f"  [{combo_mode}]")
        print(f"Entry:       {signal['price']:.4f}")
        print(f"Agreement:   {signal['agreement']}/8")
        print(f"Strategies:  {', '.join(signal['strategies'])}")
        print(
            f"Combo prior: n={n_b} "
            f"winrate={'n/a' if winrate_b is None else f'{winrate_b:.1%}'} "
            f"wilson_lb={lb_b:.1%} (need >= {COMBO_BLOCK_FLOOR:.1%})"
        )
        print(f"Consistency: {signal['consistency']:.2%}")
        if signal["anti_chase_ratio"] is not None:
            print(f"Chase ratio: {signal['anti_chase_ratio']:.2f}")
        print(f"Expiry:      {expiry_timestamp:.3f}")

        if CLICK_ENABLED:
            # Queue the intent only — see the comment on
            # self.pending_click in __init__ for why this doesn't
            # call Playwright directly from here.
            self.pending_click = {
                "trade_id": self.trade_id,
                "direction": signal["direction"],
            }
            print("[LIVE] Click queued for next main-loop tick.")
        else:
            print("[PAPER ONLY] No button clicked.")

    # ========================================================
    # COUNTER SAFETY
    # ========================================================

    def counter_market_is_allowed(
        self, original, current_timestamp, current_price
    ):

        original_direction = original["direction"]
        entry_price = original["entry_price"]

        if original_direction == "BUY":
            original_move = current_price - entry_price
            losing = current_price < entry_price
            counter_direction = "SELL"
        else:
            original_move = entry_price - current_price
            losing = current_price > entry_price
            counter_direction = "BUY"

        if not losing:
            return False, "original trade is not losing", None

        adverse_fraction = (
            abs(current_price - entry_price)
            / max(abs(entry_price), 1e-12)
        )

        if adverse_fraction < COUNTER_MIN_ADVERSE_MOVE:
            return False, "adverse movement too small", None

        volatility = self.volatility()

        if volatility is None:
            return False, "counter volatility unavailable", None

        if volatility > COUNTER_MAX_VOLATILITY:
            return False, "market too unstable for counter", None

        consistency = self.direction_consistency(
            counter_direction, current_timestamp, window=5.0
        )

        if consistency is None:
            return False, "counter consistency unavailable", None

        if consistency < COUNTER_MIN_CONSISTENCY:
            return False, "counter direction not consistent", None

        # Evidence gate for the counter too — it needs its own
        # proven track record, it doesn't inherit the original's.
        counter_signature = f"{counter_direction}:COUNTER"
        allowed, mode = self.counter_tracker.decide(
            counter_signature, current_timestamp
        )

        if not allowed:
            return False, f"counter combo evidence gate: {mode}", None

        if mode == "EXPLORE":
            self.counter_tracker.register_explore(counter_signature)

        reason = (
            f"[{mode}] original losing; "
            f"adverse={adverse_fraction:.5f}; "
            f"counter_consistency={consistency:.2%}"
        )

        return True, reason, counter_signature

    # ========================================================
    # START COUNTER
    # ========================================================

    def start_counter(self, original, timestamp, price, reason,
                       counter_signature):

        self.trade_id += 1
        self.counter_trades += 1

        direction = (
            "SELL" if original["direction"] == "BUY" else "BUY"
        )

        expiry = timestamp + COUNTER_DURATION

        self.active_counter = {
            "trade_id": self.trade_id,
            "parent_trade_id": original["trade_id"],
            "direction": direction,
            "entry_timestamp": timestamp,
            "entry_price": price,
            "expiry_timestamp": expiry,
            "reason": reason,
            "combo_signature": counter_signature,
            "original_entry_price": original["entry_price"],
            "original_current_price": price,
            "original_move": (
                price - original["entry_price"]
                if original["direction"] == "BUY"
                else original["entry_price"] - price
            ),
        }

        print()
        print(f"[V10 COUNTER #{self.trade_id}] {direction}")
        print(f"Parent:      #{original['trade_id']}")
        print(f"Entry:       {price:.4f}")
        print(f"Expiry:      {expiry:.3f}")
        print(f"Reason:      {reason}")
        print("[PAPER ONLY] Counter is loss-reduction only.")

    # ========================================================
    # SETTLE ORIGINAL
    # ========================================================

    def settle_original(self, expiry_price, timestamp):

        trade = self.active_trade

        if trade is None:
            return

        entry = trade["entry_price"]
        direction = trade["direction"]

        if direction == "BUY":
            if expiry_price > entry:
                result = "WIN"
            elif expiry_price < entry:
                result = "LOSS"
            else:
                result = "TIE"
        else:
            if expiry_price < entry:
                result = "WIN"
            elif expiry_price > entry:
                result = "LOSS"
            else:
                result = "TIE"

        if result == "WIN":
            pnl = PAYOUT * STAKE
            self.original_wins += 1
            self.consecutive_losses = 0
        elif result == "LOSS":
            pnl = -1.0 * STAKE
            self.original_losses += 1
            self.consecutive_losses += 1
        else:
            pnl = 0.0
            self.original_ties += 1

        self.original_pnl += pnl
        self.total_pnl += pnl

        # Feed the outcome back into the evidence tracker BEFORE
        # clearing the trade, so the running record is always
        # current for the next decide() call.
        if result in ("WIN", "LOSS"):
            self.combo_tracker.record_result(
                trade["combo_signature"], result == "WIN"
            )

        if (
            self.consecutive_losses
            >= CONSECUTIVE_LOSS_PAUSE_THRESHOLD
        ):
            self.paused_until = timestamp + CONSECUTIVE_LOSS_PAUSE_SECONDS
            print(
                f"[RISK] {self.consecutive_losses} consecutive "
                f"losses — pausing new entries until "
                f"t={self.paused_until:.1f} "
                f"({CONSECUTIVE_LOSS_PAUSE_SECONDS:.0f}s)."
            )

        self.active_trade = None
        self.last_trade_end = timestamp

        wins_b, n_b, winrate_b, lb_b = trade["combo_snapshot"]

        self.write_row(
            trade_type="ORIGINAL",
            parent_trade_id="",
            trade=trade,
            expiry_price=expiry_price,
            result=result,
            pnl=pnl,
            counter_reason="",
            original_entry_price="",
            original_current_price="",
            original_move="",
            trade_duration_s=f"{TRADE_DURATION:.0f}",
            combo_signature=trade["combo_signature"],
            combo_mode=trade["combo_mode"],
            combo_samples_before=n_b,
            combo_wins_before=wins_b,
            combo_winrate_before=(
                "" if winrate_b is None else f"{winrate_b:.4f}"
            ),
            combo_wilson_lb_before=f"{lb_b:.4f}",
            click_status=trade.get("click_status", "not_attempted"),
            note=f"{TRADE_DURATION:.0f}s original trade",
        )

        print()
        print(f"[RESULT ORIGINAL #{trade['trade_id']}] {result} {pnl:+.2f}")
        print(f"Entry:       {entry:.4f}")
        print(f"Exit:        {expiry_price:.4f}")
        print(f"Original P/L: {self.original_pnl:+.2f}")
        print(f"Total P/L:    {self.total_pnl:+.2f}")
        print("-" * 70)

    # ========================================================
    # SETTLE COUNTER
    # ========================================================

    def settle_counter(self, expiry_price, timestamp):

        counter = self.active_counter

        if counter is None:
            return

        entry = counter["entry_price"]
        direction = counter["direction"]

        if direction == "BUY":
            if expiry_price > entry:
                result = "WIN"
            elif expiry_price < entry:
                result = "LOSS"
            else:
                result = "TIE"
        else:
            if expiry_price < entry:
                result = "WIN"
            elif expiry_price > entry:
                result = "LOSS"
            else:
                result = "TIE"

        if result == "WIN":
            pnl = PAYOUT * STAKE
            self.counter_wins += 1
        elif result == "LOSS":
            pnl = -1.0 * STAKE
            self.counter_losses += 1
        else:
            pnl = 0.0
            self.counter_ties += 1

        self.counter_pnl += pnl
        self.total_pnl += pnl

        if result in ("WIN", "LOSS"):
            self.counter_tracker.record_result(
                counter["combo_signature"], result == "WIN"
            )

        self.active_counter = None

        self.write_row(
            trade_type="COUNTER",
            parent_trade_id=counter["parent_trade_id"],
            trade=counter,
            expiry_price=expiry_price,
            result=result,
            pnl=pnl,
            counter_reason=counter["reason"],
            original_entry_price=counter["original_entry_price"],
            original_current_price=counter["original_current_price"],
            original_move=counter["original_move"],
            trade_duration_s=f"{COUNTER_DURATION:.0f}",
            combo_signature=counter["combo_signature"],
            combo_mode="",
            combo_samples_before="",
            combo_wins_before="",
            combo_winrate_before="",
            combo_wilson_lb_before="",
            click_status="not_attempted",
            note=f"{COUNTER_DURATION:.0f}s loss-reduction counter",
        )

        print()
        print(f"[RESULT COUNTER #{counter['trade_id']}] {result} {pnl:+.2f}")
        print(f"Entry:       {entry:.4f}")
        print(f"Exit:        {expiry_price:.4f}")
        print(f"Counter P/L: {self.counter_pnl:+.2f}")
        print(f"Total P/L:   {self.total_pnl:+.2f}")
        print("-" * 70)

    # ========================================================
    # CSV
    # ========================================================

    def write_row(
        self,
        trade_type,
        parent_trade_id,
        trade,
        expiry_price,
        result,
        pnl,
        counter_reason,
        original_entry_price,
        original_current_price,
        original_move,
        trade_duration_s,
        combo_signature,
        combo_mode,
        combo_samples_before,
        combo_wins_before,
        combo_winrate_before,
        combo_wilson_lb_before,
        click_status,
        note,
    ):

        def value(key, default=""):
            return trade.get(key, default)

        def num(v, default=0.0):
            try:
                if v is None or v == "":
                    return float(default)
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        row = {
            "trade_id": value("trade_id"),
            "trade_type": trade_type,
            "parent_trade_id": parent_trade_id,
            "direction": value("direction"),
            "entry_timestamp": f"{num(value('entry_timestamp')):.3f}",
            "entry_price": f"{num(value('entry_price')):.6f}",
            "expiry_timestamp": f"{num(value('expiry_timestamp')):.3f}",
            "expiry_price": f"{expiry_price:.6f}",
            "result": result,
            "pnl": f"{pnl:.2f}",
            "agreement": value("agreement"),
            "strategies":
                "|".join(value("strategies", []))
                if isinstance(value("strategies", []), list)
                else value("strategies"),
            "momentum_1s": f"{num(value('m1', 0.0)):.6f}",
            "momentum_3s": f"{num(value('m3', 0.0)):.6f}",
            "momentum_5s": f"{num(value('m5', 0.0)):.6f}",
            "momentum_10s": f"{num(value('m10', 0.0)):.6f}",
            "acceleration": f"{num(value('acceleration', 0.0)):.6f}",
            "volatility": f"{num(value('volatility', 0.0)):.6f}",
            "volatility_percentile":
                ""
                if value("volatility_percentile") is None
                else f"{num(value('volatility_percentile')):.2f}",
            "consistency": f"{num(value('consistency', 0.0)):.4f}",
            "anti_chase_ratio":
                ""
                if value("anti_chase_ratio") is None
                else f"{num(value('anti_chase_ratio')):.4f}",
            "counter_reason": counter_reason,
            "original_entry_price": original_entry_price,
            "original_current_price": original_current_price,
            "original_move": original_move,
            "trade_duration_s": trade_duration_s,
            "combo_signature": combo_signature,
            "combo_mode": combo_mode,
            "combo_samples_before": combo_samples_before,
            "combo_wins_before": combo_wins_before,
            "combo_winrate_before": combo_winrate_before,
            "combo_wilson_lb_before": combo_wilson_lb_before,
            "click_status": click_status,
            "note": note,
        }

        self.writer.writerow(row)
        self.file.flush()

    # ========================================================
    # PROCESS QUOTE
    # ========================================================

    def process_quote(self, timestamp, price):

        self.quotes_seen += 1

        self.quote_index += 1
        self.quote_writer.writerow({
            "quote_index": self.quote_index,
            "timestamp": f"{float(timestamp):.3f}",
            "price": f"{float(price):.6f}",
            "pair": PAIR,
        })
        self.quote_file.flush()

        self.quotes.append({
            "timestamp": float(timestamp),
            "price": float(price),
        })

        cutoff = timestamp - HISTORY_SECONDS

        while self.quotes and self.quotes[0]["timestamp"] < cutoff:
            self.quotes.popleft()

        self.status()

        # ----------------------------------------------------
        # Counter settlement has priority.
        # ----------------------------------------------------

        if self.active_counter:

            if timestamp >= self.active_counter["expiry_timestamp"]:
                self.settle_counter(price, timestamp)

            return

        # ----------------------------------------------------
        # Active original trade.
        # ----------------------------------------------------

        if self.active_trade:

            trigger = (
                self.active_trade["entry_timestamp"]
                + COUNTER_TRIGGER_SECOND
            )

            if (
                COUNTER_ENABLED
                and not self.active_trade["counter_checked"]
                and timestamp >= trigger
                and timestamp < self.active_trade["expiry_timestamp"]
            ):

                self.active_trade["counter_checked"] = True

                allowed, reason, counter_signature = (
                    self.counter_market_is_allowed(
                        self.active_trade, timestamp, price
                    )
                )

                if allowed:
                    self.start_counter(
                        self.active_trade, timestamp, price, reason,
                        counter_signature
                    )
                    return

                print(
                    f"[COUNTER SKIP] "
                    f"Trade #{self.active_trade['trade_id']}: {reason}"
                )

            if timestamp >= self.active_trade["expiry_timestamp"]:
                self.settle_original(price, timestamp)

            return

        # ----------------------------------------------------
        # No active trade: risk controls, then cooldown.
        # ----------------------------------------------------

        allowed, reason = self.risk_controls_allow_entry(timestamp)

        if not allowed:
            return

        if timestamp - self.last_trade_end < COOLDOWN:
            return

        # ----------------------------------------------------
        # New entry — subject to the evidence gate.
        # ----------------------------------------------------

        signal = self.generate_signal()

        if signal is None:
            return

        combo_signature = (
            f"{signal['direction']}:{TRADE_DURATION:.0f}s:"
            + "|".join(sorted(signal["strategies"]))
        )

        gate_allowed, mode = self.combo_tracker.decide(
            combo_signature, signal["timestamp"]
        )
        combo_snapshot = self.combo_tracker.snapshot(combo_signature)

        if not gate_allowed:
            self.blocked_signal_count += 1
            wins_b, n_b, winrate_b, lb_b = combo_snapshot

            last_print = self.last_block_print.get(
                combo_signature, -1e9
            )

            if timestamp - last_print >= BLOCK_PRINT_THROTTLE_SECONDS:
                self.last_block_print[combo_signature] = timestamp
                print(
                    f"[SIGNAL BLOCKED] {signal['direction']} "
                    f"{'/'.join(signal['strategies'])} -> {mode} "
                    f"(n={n_b}, wilson_lb={lb_b:.1%}, "
                    f"need >= {COMBO_BLOCK_FLOOR:.1%})"
                )

            return

        if mode == "EXPLORE":
            self.combo_tracker.register_explore(combo_signature)

        if mode == "PROBATION":
            print(
                f"[PROBATION TRADE] {signal['direction']} "
                f"{'/'.join(signal['strategies'])}: letting one "
                f"trade through to refresh stale evidence "
                f"(prior wilson_lb={combo_snapshot[3]:.1%})"
            )

        self.start_trade(signal, combo_signature, mode, combo_snapshot)

    # ========================================================
    # LIVE STATUS
    # ========================================================

    def status(self):
        if not self.quotes:
            return
        now = self.quotes[-1]["timestamp"]
        if now - self.last_status_print < 10.0:
            return
        self.last_status_print = now
        m1 = self.momentum(now, WINDOW_1S)
        m3 = self.momentum(now, WINDOW_3S)
        m5 = self.momentum(now, WINDOW_5S)
        m10 = self.momentum(now, WINDOW_10S)
        if None not in (m1, m3, m5, m10):
            votes = self.get_strategy_votes(m1, m3, m5, m10, m1 - m3 / 3.0)
            b = sum(v == "BUY" for v in votes.values())
            s = sum(v == "SELL" for v in votes.values())
            print(
                f"[STATUS] quotes={self.quotes_seen} "
                f"bars={len(self.quotes)} votes BUY={b} SELL={s} "
                f"blocked_signals={self.blocked_signal_count} "
                f"total_pnl={self.total_pnl:+.2f}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self):

        original_decided = self.original_wins + self.original_losses
        counter_decided = self.counter_wins + self.counter_losses

        original_win_rate = (
            self.original_wins / original_decided
            if original_decided else 0.0
        )
        counter_win_rate = (
            self.counter_wins / counter_decided
            if counter_decided else 0.0
        )

        print()
        print("=" * 70)
        print("V10 PAPER TRADER SUMMARY")
        print("=" * 70)

        print(f"Signals seen:        {self.signal_count + self.blocked_signal_count}")
        print(f"Signals blocked:     {self.blocked_signal_count} (evidence gate)")
        print(f"Signals traded:      {self.signal_count}")
        print()

        print(f"Original trades:     {self.original_trades}")
        print(f"Original wins:       {self.original_wins}")
        print(f"Original losses:     {self.original_losses}")
        print(f"Original ties:       {self.original_ties}")
        print(f"Original win rate:   {original_win_rate:.2%}")
        print(f"Original P/L:        {self.original_pnl:+.2f}")
        print()

        print(f"Counter trades:      {self.counter_trades}")
        print(f"Counter wins:        {self.counter_wins}")
        print(f"Counter losses:      {self.counter_losses}")
        print(f"Counter win rate:    {counter_win_rate:.2%}")
        print(f"Counter P/L:         {self.counter_pnl:+.2f}")
        print()

        print(f"TOTAL P/L:           {self.total_pnl:+.2f}")
        print(f"Break-even win rate: {BREAKEVEN_WIN_RATE:.2%}")

        if original_win_rate >= BREAKEVEN_WIN_RATE:
            print("Status:              ABOVE break-even")
        else:
            print("Status:              BELOW break-even (expected on a")
            print("                     fair/negative-EV instrument)")

        if self.session_halted:
            print(f"Session halted:      {self.halt_reason}")

        print()
        print("Combo evidence table (direction:strategies | n, win rate,")
        print("Wilson 95% lower bound, pass >= break-even):")
        print("-" * 70)

        for row in self.combo_tracker.report_rows():
            status = "PASS" if row["passes"] else "fail"
            print(
                f"  [{status}] n={row['n']:4d}  "
                f"win_rate={row['winrate']:.1%}  "
                f"wilson_lb={row['wilson_lb']:.1%}  "
                f"{row['signature'][:70]}"
            )

        print()
        print(f"Trade CSV:            {CSV_PATH}")
        print(f"Quote CSV:            {self.quote_csv_path}")
        print("=" * 70)

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass

        try:
            self.quote_file.flush()
            self.quote_file.close()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    trader = PaperTraderV10()

    print()
    print("=" * 70)
    print("15-SECOND V10 LIVE-DATA PAPER TRADER — EVIDENCE-GATED")
    print("=" * 70)

    print()
    print(f"Pair:                 {PAIR}")
    print(f"Main expiry:          {TRADE_DURATION:.1f}s")
    print(f"Payout:               {PAYOUT:.0%}")
    print(f"Break-even win rate:  {BREAKEVEN_WIN_RATE:.2%}")

    print()
    print(f"Entry agreement:      {MIN_AGREEMENT}/8")
    print(
        f"Evidence gate:        Wilson {WILSON_Z} lower bound >= "
        f"{COMBO_BLOCK_FLOOR:.2%}, min {MIN_COMBO_SAMPLES} samples"
    )
    print(
        f"Explore cap:          "
        f"{MAX_EXPLORE_TRADES_PER_COMBO} trades/combo before proof required"
    )
    print(
        f"Probation:            1 trade per combo per "
        f"{PROBATION_SPACING_SECONDS/60:.0f} min of market time "
        f"(keeps evidence fresh)"
    )

    print()
    print("RISK CONTROLS:")
    print(f"  Session stop-loss:  {SESSION_STOP_LOSS:+.2f}")
    print(f"  Session stop-win:   {SESSION_STOP_WIN:+.2f}")
    print(
        f"  Loss-streak pause:  {CONSECUTIVE_LOSS_PAUSE_THRESHOLD} losses "
        f"-> {CONSECUTIVE_LOSS_PAUSE_SECONDS:.0f}s pause"
    )
    print(f"  Max session trades: {MAX_SESSION_TRADES}")

    print()
    print(f"Counter enabled:      {COUNTER_ENABLED}")
    if COUNTER_ENABLED:
        print(f"  Trigger:            t+{COUNTER_TRIGGER_SECOND:.1f}s")
        print(f"  Duration:           {COUNTER_DURATION:.1f}s")
        print("  Also evidence-gated (its own combo tracker).")

    print()
    print("LIVE EXECUTION:       " + ("ENABLED — REAL CLICKS" if CLICK_ENABLED else "DISABLED"))
    print("PAPER TRADING:        ENABLED")
    if CLICK_ENABLED:
        print(f"  Test stake:         {TEST_STAKE_AMOUNT}")
        print(f"  Duration label:     {duration_label(TRADE_DURATION)}")
        print(f"  Min click spacing:  {MIN_CLICK_SPACING_SECONDS:.0f}s (tripwire, not a limit)")

    print()
    print(f"Trade CSV:            {CSV_PATH}")
    print(f"Quote CSV:            {trader.quote_csv_path}")
    print("=" * 70)

    with sync_playwright() as p:

        context = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=False,
        )

        page = context.pages[0] if context.pages else context.new_page()

        def websocket_handler(ws):

            def frame_handler(payload):

                try:
                    result = parse_quote(payload)

                    if result is None:
                        return

                    timestamp, price = result

                    trader.process_quote(timestamp, price)

                except Exception as e:
                    print(f"[QUOTE ERROR] {type(e).__name__}: {e}")

            ws.on("framereceived", frame_handler)

        page.on("websocket", websocket_handler)

        print()
        print("Opening OlympTrade...")

        page.goto(PLATFORM_URL, wait_until="domcontentloaded")

        print(f"TITLE: {page.title()}")

        if CLICK_ENABLED:
            print()
            print(
                "[LIVE] Waiting for the trading UI to finish "
                "rendering (this is a heavy SPA — domcontentloaded "
                "fires well before the account panel actually "
                "exists)..."
            )
            try:
                page.wait_for_selector(
                    SEL_BALANCE_TITLE, timeout=30000
                )
                print("[LIVE] Trading UI is ready.")
            except Exception as e:
                print(
                    f"[LIVE] UI never became ready within 30s: "
                    f"{type(e).__name__}: {e}"
                )
                print(
                    "[LIVE] Disabling click execution for this "
                    "session; continuing in paper-only mode. If "
                    "this keeps happening, check whether you're "
                    "logged in / on the right page in that browser "
                    "profile."
                )

            trader.executor = OlympTradeExecutor(page)
            print()
            print("[LIVE] Executor ready. Running one preflight "
                  "check now before listening...")
            try:
                trader.executor.preflight()
                print("[LIVE] Preflight OK — Demo Account confirmed.")
            except ExecutorHaltError as e:
                print(f"[LIVE] PREFLIGHT FAILED: {e}")
                print("[LIVE] Disabling click execution for this "
                      "session; continuing in paper-only mode.")
                trader.executor = None

        print()
        print("Listening for live ASIA_X quotes...")
        print("Build sufficient history first.")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                page.wait_for_timeout(100)

                # Consume any queued click intent here, in the main
                # loop, never inside the WebSocket frame_handler —
                # see the comment on self.pending_click in
                # PaperTraderV10.__init__.
                if trader.pending_click is not None:

                    intent = trader.pending_click
                    trader.pending_click = None

                    if trader.executor is None:
                        print(
                            f"[LIVE] Trade #{intent['trade_id']} "
                            f"wanted a click but no executor is "
                            f"active (preflight failed earlier); "
                            f"skipping — this trade stays paper-only."
                        )
                        continue

                    try:
                        result = trader.executor.click_direction(
                            intent["direction"],
                            TEST_STAKE_AMOUNT,
                            duration_label(TRADE_DURATION),
                        )

                        if trader.active_trade and (
                            trader.active_trade["trade_id"]
                            == intent["trade_id"]
                        ):
                            trader.active_trade["click_status"] = (
                                "confirmed"
                                if result["click_confirmed"]
                                else "unconfirmed"
                            )

                        print(
                            f"[LIVE] Trade #{intent['trade_id']} "
                            f"click "
                            f"{'confirmed' if result['click_confirmed'] else 'UNCONFIRMED'}."
                        )

                    except ExecutorHaltError as e:
                        print(f"[LIVE] HALT: {e}")
                        print(
                            "[LIVE] Disabling click execution for "
                            "the rest of this session; the paper "
                            "trader keeps running normally."
                        )
                        trader.executor = None

        except KeyboardInterrupt:
            print()
            print("Stopping V10 paper trader...")

        finally:
            trader.summary()
            trader.close()
            context.close()

    print()
    print("=" * 70)
    print("V10 PAPER TRADING COMPLETE")
    print("=" * 70)
    print(f"Trade CSV: {CSV_PATH}")
    print(f"Quote CSV: {trader.quote_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()