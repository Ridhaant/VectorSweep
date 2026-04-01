"""
# Author: Ridhaant Ajoy Thackur
sweep_core.py — Shared Vectorised Sweep Engine  v9.0
=====================================================
Single source of truth for ALL sweep logic used by scanner1, scanner2, scanner3.

Implements EVERY behaviour from Algofinal.py:
  ✓ Level formula:           buy_above = prev_close + x, step = x (×0.6 for special)
  ✓ T1..T5 / ST1..ST5       buy_above + step×1..5 / sell_below - step×1..5
  ✓ SL:                      buy_sl = buy_above − x, sell_sl = sell_below + x (x = deviation)
  ✓ Dynamic qty:             int(100000 // price)
  ✓ Premarket adjust:        09:15–09:34 level ladder shifts (mirrors adjust_levels_premarket)
  ✓ 09:35 open-anchor:       recalc levels around actual 09:35 price
  ✓ Entry:                   price >= buy_above / price <= sell_below; one open trade/variation
  ✓ Target exits T1..T5:     gross = (Tn - entry_px) × qty   BUY
                              gross = (entry_px - STn) × qty  SELL
  ✓ SL exits:                gross = (sl_price - entry_px) × qty (negative for BUY loss)
  ✓ L re-anchor after T/ST/SL: BA=L+step, SB=L−step (retreat uses separate re-anchor)
  ✓ RETREAT 65/45/25:        peak_reached guard → 65% warn → 45% activate → 25% EXIT
                              gross = (price - entry_px) × qty  BUY
                              gross = (entry_px - price) × qty  SELL
                              After retreat: re-anchor levels, allow immediate re-entry
  ✓ EOD square-off:          equity 15:11, commodity 23:30, crypto 2h re-anchor
  ✓ Brokerage:               flat ₹20 per round-trip (₹10 entry + ₹10 exit)
  ✓ Trade log:               every event appended to trade_logs[variation_index]
  ✓ EOD save:                CSV per X-value + summary XLSX via ProcessPool
  ✓ ZMQ subscriber:          receives live prices from Algofinal ZMQ PUB
  ✓ JSON fallback:           polls live_prices.json if ZMQ unavailable

Usage in scanner files:
    from sweep_core import (
        StockSweep, SweepConfig, SweepRunner,
        PriceStore, PriceSubscriber,
        spill_trade_logs_to_disk, save_results,
        read_prev_closes_from_algofinal, fetch_930_price,
        _dynamic_quantity, now_ist, in_session, in_premarket, after_930,
        in_trading_for, IST, BROKERAGE_PER_SIDE,
    )
"""

from __future__ import annotations

import csv
import json
import logging
import math
import multiprocessing
import os
import re
import socket
import threading
import time
from copy import deepcopy
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

log = logging.getLogger("sweep_core")

# ── Exact X variant generation helpers ─────────────────────────────────────
# Many scanner ranges depend on an exact decimal step (e.g. 0.000001). Using
# float arithmetic can introduce subtle endpoint/step drift, so we generate
# ranges using Decimal and convert to float64 at the end.
from decimal import Decimal, getcontext

_DEC_CTX_PREC = 28
getcontext().prec = _DEC_CTX_PREC

def gen_x_values(x_min: float, x_max: float, step: float) -> np.ndarray:
    """
    Generate inclusive X values with an exact decimal step.

    Example:
      gen_x_values(0.008, 0.009, 1e-6) -> 1001 values (0.008000..0.009000).
    """
    dmin = Decimal(str(x_min))
    dmax = Decimal(str(x_max))
    dstep = Decimal(str(step))
    if dstep <= 0:
        raise ValueError("step must be > 0")
    if dmax < dmin:
        raise ValueError("x_max must be >= x_min")
    span = dmax - dmin

    # Number of steps (may be non-integer due to input formatting); we round.
    k = span / dstep
    k_int = int(k.to_integral_value(rounding="ROUND_HALF_EVEN"))
    last = dmin + dstep * k_int

    # If endpoints don't align on the step grid, prefer the rounded count but
    # warn since it contradicts "exact stepping".
    if abs(last - dmax) > Decimal("1e-15"):
        log.warning(
            "gen_x_values: endpoints not on exact step grid (x_min=%s x_max=%s step=%s) last=%s",
            x_min, x_max, step, float(last),
        )
    n_values = k_int + 1  # inclusive
    arr = [float(dmin + dstep * i) for i in range(n_values)]
    return np.asarray(arr, dtype=np.float64)

# ── GPU detection ─────────────────────────────────────────────────────────────
CUDA_AVAILABLE = False
GPU_NAME = "CPU (NumPy)"
cp = None
try:
    import warnings as _w; _w.filterwarnings("ignore")
    import cupy as _cp
    _cp.cuda.Device(0).use()
    _t = _cp.zeros(1000, dtype=_cp.float64); _t += 1.0
    assert float(_t.sum()) == 1000.0
    _cp.cuda.Stream.null.synchronize()
    del _t
    try:
        _props = _cp.cuda.runtime.getDeviceProperties(0)
        GPU_NAME = _props["name"].decode() if isinstance(_props["name"], bytes) else str(_props["name"])
    except Exception:
        GPU_NAME = "NVIDIA GPU"
    cp = _cp
    CUDA_AVAILABLE = True
except Exception:
    pass

xp = cp if CUDA_AVAILABLE else np
# CPU workers: leave 2 cores for OS + dashboard, use remaining for parallel work
# On i5-12450H (12 logical): 4 workers for sweeps, rest for other processes
# i5-12450H: 12 logical cores. Leave 4 for OS/Dashboard/Engine.
# Each scanner gets its own core via process_affinity.py
# Use 4 workers for parallel EOD writing (I/O bound)
CPU_WORKERS = max(2, min(multiprocessing.cpu_count() - 4, 4))

# ── Constants ─────────────────────────────────────────────────────────────────
IST              = pytz.timezone("Asia/Kolkata")
BROKERAGE_PER_SIDE = 10.0       # ₹10 per side → ₹20 per round-trip
USDT_TO_INR = float(os.getenv("USDT_TO_INR", "0") or 84.0)
LIVE_JSON_PATH   = os.path.join("levels", "live_prices.json")
ZMQ_PUB_ADDR        = os.getenv("ZMQ_PUB_ADDR", "tcp://127.0.0.1:28081")
ZMQ_CRYPTO_SUB_ADDR = os.getenv("ZMQ_CRYPTO_SUB", "tcp://127.0.0.1:28082")
ZMQ_TOPIC        = b"prices"

SPECIAL_SYMBOLS  = {"RELIANCE", "SBIN", "KOTAKBANK", "ICICIBANK", "HUL", "HDFC"}
INDEX_SYMBOLS    = {"NIFTY", "BANKNIFTY"}

STOCKS: List[str] = [
    "NIFTY", "BANKNIFTY",
    "HDFCBANK", "KOTAKBANK", "SBIN", "ICICIBANK", "INDUSINDBK",
    "ADANIPORTS", "ADANIENT", "ASIANPAINT", "BAJFINANCE", "DRREDDY",
    "SUNPHARMA", "INFY", "TCS", "TECHM",
    "TITAN", "TATAMOTORS", "RELIANCE", "INDIGO", "JUBLFOOD",
    "BATAINDIA", "PIDILITIND", "ZEEL", "BALKRISIND", "VOLTAS",
    "ITC", "BPCL", "BRITANNIA", "HEROMOTOCO",
    "HINDUNILVR", "UPL", "SRF", "TATACONSUM", "BALRAMCHIN",
    "ABFRL", "VEDL", "COFORGE",
]
COMMODITY_SYMBOLS: List[str] = ["GOLD", "SILVER", "NATURALGAS", "CRUDE", "COPPER"]
CRYPTO_SYMBOLS:    List[str] = ["BTC", "ETH", "BNB", "SOL", "ADA"]
_COMMODITY_SET = frozenset(COMMODITY_SYMBOLS)
_CRYPTO_SET    = frozenset(CRYPTO_SYMBOLS)


# ════════════════════════════════════════════════════════════════════════════
# TIME HELPERS
# ════════════════════════════════════════════════════════════════════════════

def now_ist() -> datetime:
    return datetime.now(IST)

def in_session(dt: datetime) -> bool:
    """True if ANY market (equity, commodity, crypto) is active. Crypto=always True."""
    t = dt.hour * 60 + dt.minute
    return 9 * 60 + 15 <= t <= 23 * 60 + 30

def in_premarket(dt: datetime) -> bool:
    t = dt.hour * 60 + dt.minute
    return 9 * 60 + 15 <= t < 9 * 60 + 35

def after_930(dt: datetime) -> bool:
    return dt.hour > 9 or (dt.hour == 9 and dt.minute >= 35)


# Entry blackout — mirrors Algofinal.in_entry_blackout
_ENTRY_OPEN_MIN = 36

def in_entry_blackout(dt: datetime, *, is_commodity: bool = False) -> bool:
    t = dt.hour * 60 + dt.minute
    start = 9 * 60 + _ENTRY_OPEN_MIN
    if is_commodity:
        return 9 * 60 + 0 <= t < start
    return 9 * 60 + 15 <= t < start

def in_commodity_session(dt: datetime) -> bool:
    """MCX session: 09:00–23:30 IST, Mon–Fri."""
    t = dt.hour * 60 + dt.minute
    return 9 * 60 + 0 <= t < 23 * 60 + 30

def is_commodity_eod(dt: datetime) -> bool:
    """True at or after 23:30 IST (MCX square-off time)."""
    return dt.hour == 23 and dt.minute >= 30

def in_930_blackout(dt: datetime) -> bool:
    """Legacy name — equity entry blackout (09:15–09:35)."""
    return in_entry_blackout(dt, is_commodity=False)

def in_trading_for(symbol: str, dt: datetime) -> bool:
    """True if symbol is in its trading window at dt.
    Equity:    09:36 – 15:11 IST  (Mon–Fri)
    Commodity: 09:36 – 23:30 IST  (Mon–Fri)
    Crypto:    24/7 always True
    """
    sym_upper = symbol.upper()
    if sym_upper in _CRYPTO_SET:
        return True  # crypto never closes
    t = dt.hour * 60 + dt.minute
    if sym_upper in _COMMODITY_SET:
        return 9 * 60 + _ENTRY_OPEN_MIN <= t < 23 * 60 + 30
    return 9 * 60 + _ENTRY_OPEN_MIN <= t <= 15 * 60 + 11

# LRU-cached quantity
_qty_cache_inr: dict = {}
_qty_cache_usdt: dict = {}
_qty_cache_crypto: dict = {}
def _dynamic_quantity(
    price: float,
    *,
    is_commodity: bool = False,
    is_crypto: bool = False,
) -> int:
    """
    Equity qty (INR price):  int(100_000 // price)
    Commodity qty (USDT price): int((100_000/USDT_TO_INR) // price)
    Crypto qty (USDT price):  int(100_000 // price)
    """
    if price <= 0:
        return 0
    key = int(price)  # cache at whole-unit granularity for speed

    if is_crypto:
        # Crypto prices are in USDT; prefer dedicated USDT budget from config.
        cache = _qty_cache_crypto
        try:
            from config import cfg as _cfg
            capital_usdt = float(getattr(_cfg, "CRYPTO_BUDGET_USDT", 0) or 0)
            if capital_usdt <= 0:
                capital_usdt = float(_cfg.CRYPTO_BUDGET_INR) / float(_cfg.USDT_TO_INR or 84.0)
        except Exception:
            capital_usdt = 1065.0
        if key not in cache:
            cache[key] = int(capital_usdt // price)
            if len(cache) > 5000:
                cache.clear()
        return cache[key]

    if is_commodity and USDT_TO_INR > 0:
        cache = _qty_cache_usdt
        capital_usdt = 100_000 / USDT_TO_INR
        if key not in cache:
            cache[key] = int(capital_usdt // price)
            if len(cache) > 5000:
                cache.clear()
        return cache[key]

    cache = _qty_cache_inr
    if key not in cache:
        cache[key] = int(100_000 // price)
        if len(cache) > 5000:
            cache.clear()
    return cache[key]


# ════════════════════════════════════════════════════════════════════════════
# IPC — PRICE STORE + SUBSCRIBER
# ════════════════════════════════════════════════════════════════════════════

class PriceStore:
    """Thread-safe live price dict. Shared between subscriber and sweep engine."""
    __slots__ = ("_lock", "_prices", "_ticks")

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._prices: Dict[str, float] = {}
        self._ticks  = 0

    def update(self, prices: Dict[str, float]) -> None:
        with self._lock:
            for k, v in prices.items():
                self._prices[k.upper()] = float(v)
            self._ticks += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)

    @property
    def ticks(self) -> int:
        with self._lock:
            return self._ticks


class PriceSubscriber:
    """
    Background thread: ZMQ SUB → PriceStore. JSON file fallback when ZMQ unavailable.
    Writes every tick to a CSV file for EOD analysis.
    """

    def __init__(self, store: PriceStore, csv_path: str, topic: bytes = None) -> None:
        self._store    = store
        self._csv_path = csv_path
        self._stop     = threading.Event()
        self._zmq_ok   = False
        self._topic    = topic or ZMQ_TOPIC
        try:
            import zmq as _zmq  # noqa
            self._zmq_ok = True
        except ImportError:
            pass
        mode = "ZMQ" if self._zmq_ok else "JSON-file"
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"PriceSub-{os.getpid()}-{mode}"
        )
        self._mode = mode
        self._last_json_sig: Optional[tuple] = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)
        topic_s = self._topic.decode() if isinstance(self._topic, bytes) else str(self._topic)
        force_json = (os.getenv("FORCE_JSON_IPC", "0") == "1") or (
            "commodity" in topic_s and os.getenv("FORCE_ZMQ_IPC", "0") != "1"
        )
        self._mode = "JSON-file" if force_json or not self._zmq_ok else "ZMQ"
        self._thread.start()
        log.info("PriceSubscriber started (mode=%s)", self._mode)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)

    def _run(self) -> None:
        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts", "symbol", "price"])
            topic_s = self._topic.decode() if isinstance(self._topic, bytes) else str(self._topic)
            force_json = (os.getenv("FORCE_JSON_IPC", "0") == "1") or (
                "commodity" in topic_s and os.getenv("FORCE_ZMQ_IPC", "0") != "1"
            )
            if self._zmq_ok and not force_json:
                self._run_zmq(writer, fh)
            else:
                self._run_json(writer, fh)

    def _run_zmq(self, writer, fh) -> None:
        import zmq
        fallback_s = float(os.getenv("IPC_ZMQ_FALLBACK_S", "12.0") or 12.0)
        ctx  = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 20)
        sock.setsockopt(zmq.LINGER, 0)
        # v8.0: 20ms timeout (was 100ms) — 5× faster loop exit on stop
        sock.setsockopt(zmq.RCVTIMEO, 20)
        # v8.0: CONFLATE=1 — keep only the latest message, drop stale prices
        # This prevents price backlog buildup on slow CPUs
        try:
            sock.setsockopt(zmq.CONFLATE, 1)
        except AttributeError:
            pass  # older zmq versions may not have CONFLATE
        connect_addr = ZMQ_CRYPTO_SUB_ADDR if self._topic == b"crypto" else ZMQ_PUB_ADDR
        sock.connect(connect_addr)
        sock.setsockopt(zmq.SUBSCRIBE, self._topic)
        sock.setsockopt(zmq.SUBSCRIBE, b"hb")
        log.info("ZMQ SUB connected → %s  (RCVTIMEO=20ms, CONFLATE=1)", connect_addr)
        last_data = time.monotonic()
        while not self._stop.is_set():
            try:
                _topic, raw = sock.recv_multipart()
                payload = json.loads(raw)
                if payload.get("heartbeat"):
                    if fallback_s > 0 and (time.monotonic() - last_data) > fallback_s:
                        log.warning("ZMQ heartbeat-only for %.1fs; switching to JSON IPC", time.monotonic() - last_data)
                        sock.close()
                        self._run_json(writer, fh)
                        return
                    continue
                if _topic != self._topic:
                    continue
                ts     = payload.get("ts", "")
                prices = payload.get("prices", {})
                if not prices:
                    continue
                last_data = time.monotonic()
                self._store.update(prices)
                for sym, px in prices.items():
                    writer.writerow([ts, sym.upper(), px])
                fh.flush()
            except zmq.Again:
                if fallback_s > 0 and (time.monotonic() - last_data) > fallback_s:
                    log.warning("ZMQ idle for %.1fs; switching to JSON IPC", time.monotonic() - last_data)
                    sock.close()
                    self._run_json(writer, fh)
                    return
                continue
            except zmq.ZMQError:
                if self._stop.is_set():
                    break
                time.sleep(0.1)
            except Exception:
                time.sleep(0.1)
        sock.close()

    def _run_json(self, writer, fh) -> None:
        """JSON-file fallback. v10.5 FIX: reads topic-correct price section."""
        while not self._stop.is_set():
            try:
                if not os.path.exists(LIVE_JSON_PATH):
                    self._stop.wait(0.5)
                    continue
                with open(LIVE_JSON_PATH, "r", encoding="utf-8") as pf:
                    payload = json.load(pf)
                ts = payload.get("ts", "")
                # v10.5: read the correct section per subscription topic
                _topic = self._topic.decode() if isinstance(self._topic, bytes) else str(self._topic)
                if "crypto" in _topic:
                    prices = payload.get("crypto_prices") or payload.get("prices", {})
                    _CRYPTO = {"BTC","ETH","BNB","SOL","ADA"}
                    prices = {k:v for k,v in prices.items() if k.upper() in _CRYPTO}
                    tick = payload.get("crypto_ts") or ts
                elif "commodity" in _topic:
                    prices = payload.get("commodity_prices") or payload.get("prices", {})
                    _COMM = {"GOLD","SILVER","NATURALGAS","CRUDE","COPPER"}
                    prices = {k:v for k,v in prices.items() if k.upper() in _COMM}
                    tick = payload.get("commodity_ts") or ts
                else:
                    prices = payload.get("equity_prices") or payload.get("prices", {})
                    _COMM = {"GOLD","SILVER","NATURALGAS","CRUDE","COPPER"}
                    _CRYP = {"BTC","ETH","BNB","SOL","ADA"}
                    prices = {k:v for k,v in prices.items()
                              if k.upper() not in _COMM and k.upper() not in _CRYP}
                    tick = ts
                row: List[Tuple[str, float]] = []
                for k, v in prices.items():
                    try:
                        row.append((str(k).upper(), float(v)))
                    except (TypeError, ValueError):
                        continue
                sig = (tick, tuple(sorted(row)))
                if sig == self._last_json_sig or not prices:
                    self._stop.wait(0.02)
                    continue
                self._last_json_sig = sig
                self._store.update(prices)
                for sym, px in prices.items():
                    writer.writerow([tick, sym.upper(), px])
                fh.flush()
            except json.JSONDecodeError:
                self._stop.wait(0.05)
            except Exception:
                self._stop.wait(0.25)


# ════════════════════════════════════════════════════════════════════════════
# PREV-CLOSE LOADER
# ════════════════════════════════════════════════════════════════════════════

def read_prev_closes_from_algofinal(date_str: str) -> Dict[str, float]:
    """
    Read prev_close values from Algofinal's persistent cache.
    Order of priority:
      1. levels/prev_closes_persistent_YYYYMMDD.json  (fastest)
      2. levels/prev_closes_cache_YYYYMMDD.json
      3. levels/initial_levels_*.xlsx  (XLSX parse, slowest)
    Returns {SYMBOL_UPPER: prev_close_float}.
    """
    result: Dict[str, float] = {}

    for fname in (
        f"prev_closes_persistent_{date_str}.json",
        f"prev_closes_cache_{date_str}.json",
    ):
        path = os.path.join("levels", fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Handle both {SYM: price} and {SYM: {prev_close: price}}
                for sym, val in data.items():
                    if isinstance(val, (int, float)):
                        result[sym.upper()] = float(val)
                    elif isinstance(val, dict) and "prev_close" in val:
                        result[sym.upper()] = float(val["prev_close"])
            except Exception as exc:
                log.debug("prev_close JSON load failed (%s): %s", fname, exc)

    # XLSX fallback
    candidates = []
    if os.path.isdir("levels"):
        for f in os.listdir("levels"):
            if f.startswith("initial_levels_") and f.endswith(".xlsx") and date_str in f:
                candidates.append(os.path.join("levels", f))
    if candidates:
        xlsx_path = max(candidates, key=os.path.getmtime)
        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl")
            for _, row in df.iterrows():
                sym = str(row.get("Symbol", "")).strip().upper()
                for col in ("Price", "Prev Close", "PrevClose", "prev_close"):
                    if col in row and pd.notna(row[col]):
                        try:
                            result[sym] = float(row[col])
                            break
                        except (ValueError, TypeError):
                            pass
            log.info("Loaded prev_close for %d stocks from XLSX", len(result))
        except Exception as exc:
            log.debug("XLSX prev_close load failed: %s", exc)

    if result:
        log.info("Loaded prev_close values for %d stocks (merged JSON + XLSX)", len(result))
    return result


@functools.lru_cache(maxsize=1)
def load_equity_universe_from_levels_xlsx() -> List[str]:
    """
    Load equity symbol universe from the latest equity levels XLSX.

    Mirrors the dashboard's block-parsing approach (7-row blocks), and filters out
    known commodity/crypto symbols.
    """
    import openpyxl

    target_count = int(os.getenv("EQUITY_TARGET_COUNT", "40") or 40)
    extra_raw = os.getenv("EQUITY_EXTRA_SYMBOLS", "").strip()
    extra_symbols = [s.strip().upper() for s in extra_raw.split(",") if s.strip()]

    cands: List[str] = []
    for dirp in (
        os.path.join("levels", "adjusted_levels"),
        "levels",
        os.path.join("levels", "initial_levels_930"),
        os.path.join("levels", "initial_levels_prevday"),
        os.path.join("levels", "initial_levels_eod"),
    ):
        if not os.path.isdir(dirp):
            continue
        for f in os.listdir(dirp):
            if not f.endswith(".xlsx"):
                continue
            fn = f.lower()
            if "initial_levels" not in fn:
                continue
            if "commodity" in fn or "crypto" in fn:
                continue
            cands.append(os.path.join(dirp, f))

    if not cands:
        return []

    cands = list(set(cands))
    cands.sort(key=os.path.getmtime, reverse=True)

    for best in cands[:12]:
        try:
            wb = openpyxl.load_workbook(best, read_only=True, data_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            wb.close()

            result: List[str] = []
            i = 0
            while i < len(rows):
                r = rows[i]
                if r and r[0] is not None and isinstance(r[0], str) and len(r[0]) >= 2:
                    sym = str(r[0]).upper().strip()
                    if sym in _COMMODITY_SET or sym in _CRYPTO_SET:
                        i += 7
                        continue
                    result.append(sym)
                    i += 7
                else:
                    i += 1

            if len(result) >= 10:
                log.info(
                    "load_equity_universe_from_levels_xlsx: loaded %d equity symbols from %s",
                    len(result),
                    os.path.basename(best),
                )
                # Optional: extend via env for deployments where levels XLSX lags reality.
                if extra_symbols:
                    for s in extra_symbols:
                        if s not in result:
                            result.append(s)
                if len(result) < target_count:
                    log.warning(
                        "Equity universe short: got %d symbols, target=%d. (Set EQUITY_EXTRA_SYMBOLS to add missing tickers.)",
                        len(result),
                        target_count,
                    )
                return result
        except Exception:
            continue

    return []


def fetch_prev_close(symbol: str) -> Optional[float]:
    """yfinance fallback for prev_close (only called on cache miss)."""
    try:
        import yfinance as yf
        t = yf.Ticker(f"{symbol}.NS")
        df = t.history(period="5d", interval="1d")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_930_price(symbol: str, prev_close: float,
                    target_date: Optional[datetime] = None) -> Optional[float]:
    """Fetch the 09:30 IST price for re-anchoring. Mirrors Algofinal.get_price_at_930."""
    # First check shared market data written by Algofinal
    try:
        date_str = (target_date or now_ist()).strftime("%Y%m%d")
        smd_path = os.path.join("levels", f"shared_market_data_{date_str}.json")
        if os.path.exists(smd_path):
            with open(smd_path, "r", encoding="utf-8") as fh:
                smd = json.load(fh)
            prices = smd.get("935_prices") or smd.get("open_anchor_prices") or smd.get("930_prices") or {}
            p = prices.get(symbol.upper())
            if p and abs(float(p) - prev_close) > 0.001:
                return float(p)
    except Exception:
        pass
    # yfinance fallback
    try:
        import yfinance as yf
        t    = yf.Ticker(f"{symbol}.NS")
        df   = t.history(period="1d", interval="1m")
        if df is None or df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC).tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
        ref  = (target_date or now_ist()).replace(hour=9, minute=35, second=0, microsecond=0)
        mask = (df.index >= ref) & (df.index < ref + timedelta(minutes=1))
        sl   = df.loc[mask]
        if sl.empty:
            before = df.loc[df.index < ref]
            return float(before["Close"].iloc[-1]) if not before.empty else None
        return float(sl["Close"].iloc[-1])
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# STOCK SWEEP — vectorised simulation of ONE stock across ALL X variations
# ════════════════════════════════════════════════════════════════════════════

class StockSweep:
    """
    Simulates all N X-factor variations for a single stock simultaneously.

    Array shapes: (N,) where N = number of X variations for this symbol.

    Faithfully mirrors every Algofinal behaviour including:
      - 65/45/25% retreat monitoring with peak-guard
      - L re-anchor after T/ST/SL exits (not retreat)
      - Level re-anchoring after retreat exit
    """

    def __init__(
        self,
        symbol:       str,
        prev_close:   float,
        x_values_np:  np.ndarray,        # caller supplies the X multiplier array
        *,
        is_commodity: bool = False,
        is_crypto: bool = False,
    ) -> None:
        self.symbol       = symbol
        self.prev_close   = prev_close
        self.is_commodity = is_commodity
        self.is_crypto    = is_crypto
        self.is_special   = symbol in SPECIAL_SYMBOLS
        self.is_index     = symbol in INDEX_SYMBOLS

        n  = len(x_values_np)
        self._n    = n
        self._xv_np = x_values_np.copy()   # multipliers (not rupees)

        # Rupee X values
        xr = (prev_close * x_values_np).astype(np.float64)
        self.x_arr    = xr.copy()
        # step = x×0.6 for special symbols, else x
        self.step_arr = (xr * 0.6) if self.is_special else xr.copy()

        self._levels_locked = False
        self.tick_count     = 0
        self.last_price     = 0.0

        self._init_levels()
        self._init_position_state(n)

    # ── Level initialisation ──────────────────────────────────────────────────

    def _init_levels(self) -> None:
        pc  = self.prev_close
        xa  = self.x_arr
        s   = self.step_arr
        n   = self._n
        self.buy_above  = pc + xa              # (N,)
        self.sell_below = pc - xa              # (N,)
        self.buy_sl     = self.buy_above - xa
        self.sell_sl    = self.sell_below + xa
        self.t  = np.stack([self.buy_above  + s * i for i in range(1, 6)])  # (5,N)
        self.st = np.stack([self.sell_below - s * i for i in range(1, 6)])  # (5,N)

    def _init_position_state(self, n: int) -> None:
        # ── Position ──────────────────────────────────────────────────────────
        self.in_position  = np.zeros(n, dtype=bool)
        self.is_buy       = np.zeros(n, dtype=bool)
        self.entry_price  = np.zeros(n, dtype=np.float64)
        self.entry_qty    = np.zeros(n, dtype=np.int32)
        self.exited_today = np.zeros(n, dtype=bool)

        self.custom_sl     = np.full(n, np.nan, dtype=np.float64)
        self.custom_target = np.full(n, np.nan, dtype=np.float64)

        # ── Retreat state (NEW — mirrors handle_retreat_monitoring) ───────────
        # entry_level: buy_above at entry for BUY, sell_below for SELL
        self.retreat_entry_level  = np.zeros(n, dtype=np.float64)
        # peak_reached: True once price ≥65% of one step toward profit
        #   → prevents instant-fire at entry giving gross≈0
        self.retreat_peak_reached = np.zeros(n, dtype=bool)
        self.retreat_65_alerted   = np.zeros(n, dtype=bool)
        self.retreat_45_alerted   = np.zeros(n, dtype=bool)

        # ── P&L tracking ──────────────────────────────────────────────────────
        self.total_pnl   = np.zeros(n, dtype=np.float64)
        self.trade_count = np.zeros(n, dtype=np.int32)
        self.win_count   = np.zeros(n, dtype=np.int32)
        self.loss_count  = np.zeros(n, dtype=np.int32)

        # ── Drawdown tracking (peak-to-trough on cumulative P&L) ─────────────
        # equity_peak holds the best cumulative P&L seen so far.
        # max_drawdown is the maximum peak-to-current drawdown observed.
        self.equity_peak   = np.zeros(n, dtype=np.float64)
        self.max_drawdown  = np.zeros(n, dtype=np.float64)

        # ── Tick-level ladder breach counters ───────────────────────────────
        # Counts "every incoming tick where price is beyond the level",
        # but only while the variation is in an active BUY/SELL position.
        #
        # Shapes:
        #   breach_ticks_T  : (5, N) where T[0] corresponds to T1
        #   breach_ticks_ST : (5, N) where ST[0] corresponds to ST1
        self.breach_ticks_T  = np.zeros((5, n), dtype=np.uint32)
        self.breach_ticks_ST = np.zeros((5, n), dtype=np.uint32)

        # First tick index (within this StockSweep) when the variation first
        # breached a given level. 0 = never breached.
        self.first_breach_tick_T  = np.zeros((5, n), dtype=np.uint32)
        self.first_breach_tick_ST = np.zeros((5, n), dtype=np.uint32)

        # ── Per-variation trade log (CPU — list of dicts) ─────────────────────
        self.trade_logs: List[List[dict]] = [[] for _ in range(n)]

    def _reanchor_variation(self, vi: int, L: float) -> None:
        """After T/ST/SL exit: new BA=L+band, new SB=L−band (band=target_step). Not for retreat."""
        band = float(self.step_arr[vi])
        if band <= 0 or L <= 0:
            return
        self.x_arr[vi] = band
        self.step_arr[vi] = (band * 0.6) if self.is_special else band
        s = float(self.step_arr[vi])
        self.buy_above[vi] = L + band
        self.sell_below[vi] = L - band
        for j in range(5):
            self.t[j][vi] = self.buy_above[vi] + s * (j + 1)
            self.st[j][vi] = self.sell_below[vi] - s * (j + 1)
        self.buy_sl[vi] = self.buy_above[vi] - self.x_arr[vi]
        self.sell_sl[vi] = self.sell_below[vi] + self.x_arr[vi]

    # ── Premarket level adjustment ────────────────────────────────────────────

    def premarket_adjust(self, price: float) -> None:
        """
        Vectorised mirror of adjust_levels_premarket() in Algofinal.py.
        Shifts the level ladder one step per tick when price crosses a level.
        """
        if self._levels_locked or self.is_commodity or self.is_crypto:
            return
        n  = self._n
        ba = self.buy_above.copy()
        sb = self.sell_below.copy()
        s  = self.step_arr

        # BUY side: find highest level crossed
        buy_levels = np.vstack([ba[np.newaxis, :], self.t])  # (6,N)
        crossed_b  = price >= buy_levels
        highest_b  = np.full(n, -1, dtype=np.int8)
        for i in range(5, -1, -1):
            m = crossed_b[i] & (highest_b == -1)
            if m.any():
                highest_b[m] = i

        any_b = highest_b >= 0
        if any_b.any():
            for i in range(6):
                m = (highest_b == i) & any_b
                if not m.any():
                    continue
                # Shift buy_above to t[i] (or t[0]+step if already at t[0])
                if i == 0:
                    self.buy_above[m]  = ba[m] + s[m]
                else:
                    self.buy_above[m]  = self.t[i-1][m] + s[m]
                self.buy_above[m]  = np.maximum(self.buy_above[m], ba[m])
                for j in range(5):
                    self.t[j][m]   = self.buy_above[m] + s[m] * (j + 1)
                shift = self.buy_above[m] - ba[m]
                self.sell_below[m] = sb[m] + shift
                for j in range(5):
                    self.st[j][m]  = self.sell_below[m] - s[m] * (j + 1)
                self.buy_sl[m]     = self.buy_above[m] - self.x_arr[m]
                self.sell_sl[m]    = self.sell_below[m] + self.x_arr[m]
            return

        # SELL side: find lowest level crossed
        sell_levels = np.vstack([sb[np.newaxis, :], self.st])  # (6,N)
        crossed_s   = price <= sell_levels
        lowest_s    = np.full(n, -1, dtype=np.int8)
        for i in range(5, -1, -1):
            m = crossed_s[i] & (lowest_s == -1)
            if m.any():
                lowest_s[m] = i

        any_s = lowest_s >= 0
        if any_s.any():
            for i in range(6):
                m = (lowest_s == i) & any_s
                if not m.any():
                    continue
                if i == 0:
                    self.sell_below[m] = sb[m] - s[m]
                else:
                    self.sell_below[m] = self.st[i-1][m] - s[m]
                self.sell_below[m] = np.minimum(self.sell_below[m], sb[m])
                for j in range(5):
                    self.st[j][m]  = self.sell_below[m] - s[m] * (j + 1)
                shift = sb[m] - self.sell_below[m]
                self.buy_above[m]  = ba[m] - shift
                for j in range(5):
                    self.t[j][m]   = self.buy_above[m] + s[m] * (j + 1)
                self.buy_sl[m]     = self.buy_above[m] - self.x_arr[m]
                self.sell_sl[m]    = self.sell_below[m] + self.x_arr[m]

    def reanchor_at_930(self, price_930: float) -> None:
        """
        One-time open-anchor re-anchor (09:35 IST). Mirrors Algofinal.
        Rebuilds all levels centred on anchor price.
        """
        if abs(price_930 - self.prev_close) < 0.01:
            self._levels_locked = True
            return
        x   = self.x_arr
        s   = self.step_arr
        self.buy_above  = price_930 + x
        self.sell_below = price_930 - x
        for i in range(5):
            self.t[i]  = self.buy_above  + s * (i + 1)
            self.st[i] = self.sell_below - s * (i + 1)
        self.buy_sl  = self.buy_above - x
        self.sell_sl = self.sell_below + x
        self._levels_locked = True

    # ── Main price tick ───────────────────────────────────────────────────────

    def on_price(self, price: float, ts: datetime) -> None:
        """
        Process one live price tick across ALL N variations simultaneously.
        Order mirrors Algofinal main loop:
          1. Entries
          2. Target exits T1..T5
          3. RETREAT 65/45/25
          4. SL exits
        Fast-exit: if price unchanged AND no open positions,
        skip all vectorised work (~3µs vs ~200µs per full tick).
        """
        # Fast-exit: nothing to do if price unchanged and no activity
        if (price == self.last_price and
                not self.in_position.any()):
            self.tick_count += 1
            return
        self.last_price  = price
        self.tick_count += 1

        qty     = _dynamic_quantity(price, is_commodity=self.is_commodity, is_crypto=self.is_crypto)
        ts_str  = ts.strftime("%Y-%m-%d %H:%M:%S IST")
        brok    = BROKERAGE_PER_SIDE * 2.0   # ₹20 per round-trip (equity/computations)
        if (self.is_commodity or self.is_crypto) and USDT_TO_INR > 0:
            brok = brok / USDT_TO_INR       # convert brokerage to USDT (price is USDT)

        if qty <= 0:
            return

        ep      = self.entry_price      # (N,) float64
        qty_arr = self.entry_qty.astype(np.float64)   # (N,) float64

        not_in = ~self.in_position
        ep     = self.entry_price
        qty_arr = self.entry_qty.astype(np.float64)

        # ── 1. ENTRIES ────────────────────────────────────────────────────────
        _in_blackout = (not self.is_crypto) and in_entry_blackout(ts, is_commodity=self.is_commodity)
        buy_entry  = not_in & (price >= self.buy_above) & (not _in_blackout)
        sell_entry = not_in & (price <= self.sell_below) & (not _in_blackout)

        buy_idx = np.where(buy_entry)[0]
        if buy_idx.size > 0:
            self.in_position[buy_idx]         = True
            self.is_buy[buy_idx]              = True
            self.entry_price[buy_idx]         = price
            self.entry_qty[buy_idx]           = qty
            self.exited_today[buy_idx]        = False
            self.custom_sl[buy_idx]           = np.nan
            self.custom_target[buy_idx]       = np.nan
            self.retreat_entry_level[buy_idx] = self.buy_above[buy_idx]
            self.retreat_peak_reached[buy_idx]= False
            self.retreat_65_alerted[buy_idx]  = False
            self.retreat_45_alerted[buy_idx]  = False
            for vi in buy_idx:
                self.trade_logs[vi].append({
                    "ts": ts_str, "sym": self.symbol, "event": "BUY_ENTRY",
                    "level_breached": "BUY_ABOVE",
                    "level_px": round(float(self.buy_above[vi]), 2),
                    "price": price, "qty": qty, "entry_px": price,
                    "gross": 0.0, "net": 0.0,
                    "x": round(float(self._xv_np[vi]), 8),
                })

        sell_idx = np.where(sell_entry)[0]
        if sell_idx.size > 0:
            self.in_position[sell_idx]          = True
            self.is_buy[sell_idx]               = False
            self.entry_price[sell_idx]          = price
            self.entry_qty[sell_idx]            = qty
            self.exited_today[sell_idx]         = False
            self.custom_sl[sell_idx]            = np.nan
            self.custom_target[sell_idx]        = np.nan
            self.retreat_entry_level[sell_idx]  = self.sell_below[sell_idx]
            self.retreat_peak_reached[sell_idx] = False
            self.retreat_65_alerted[sell_idx]   = False
            self.retreat_45_alerted[sell_idx]   = False
            for vi in sell_idx:
                self.trade_logs[vi].append({
                    "ts": ts_str, "sym": self.symbol, "event": "SELL_ENTRY",
                    "level_breached": "SELL_BELOW",
                    "level_px": round(float(self.sell_below[vi]), 2),
                    "price": price, "qty": qty, "entry_px": price,
                    "gross": 0.0, "net": 0.0,
                    "x": round(float(self._xv_np[vi]), 8),
                })

        # Refresh ep / qty_arr after entries
        ep      = self.entry_price
        qty_arr = self.entry_qty.astype(np.float64)

        # ── 3. TARGET EXITS ───────────────────────────────────────────────────
        buy_open  = self.in_position & self.is_buy  & (ep > 0)
        sell_open = self.in_position & ~self.is_buy & (ep > 0)

        # Tick-level ladder breach counting:
        # count every tick that is beyond Tn / STn while this variation is actively open.
        # This is updated before we process target exits so the "breaching tick" is included.
        breached_T = buy_open[None, :] & (price >= self.t)      # (5, N)
        breached_ST = sell_open[None, :] & (price <= self.st)  # (5, N)
        self.breach_ticks_T += breached_T.astype(np.uint32)
        self.breach_ticks_ST += breached_ST.astype(np.uint32)

        # Record the first tick index where each level was breached
        # for each variation (disk logger can reconstruct "full events"
        # without storing a line per tick).
        tick_u32 = np.uint32(self.tick_count)
        new_T = breached_T & (self.first_breach_tick_T == 0)
        if new_T.any():
            self.first_breach_tick_T[new_T] = tick_u32
        new_ST = breached_ST & (self.first_breach_tick_ST == 0)
        if new_ST.any():
            self.first_breach_tick_ST[new_ST] = tick_u32

        # BUY — custom target
        if buy_open.any():
            ct     = self.custom_target
            has_ct = buy_open & np.isfinite(ct)
            ct_idx = np.where(has_ct & (price >= ct) & (ct > ep))[0]
            if ct_idx.size > 0:
                gross = (ct[ct_idx] - ep[ct_idx]) * qty_arr[ct_idx]
                net   = gross - brok
                self.total_pnl[ct_idx]   += net
                self.trade_count[ct_idx] += 1
                self.win_count[ct_idx]   += (net > 0).astype(np.int32)
                self.loss_count[ct_idx]  += (net <= 0).astype(np.int32)
                self.in_position[ct_idx]  = False
                self.is_buy[ct_idx]       = False
                self.exited_today[ct_idx] = True
                self.retreat_peak_reached[ct_idx] = False
                self.retreat_65_alerted[ct_idx]   = False
                self.retreat_45_alerted[ct_idx]   = False
                for vi in ct_idx:
                    self._reanchor_variation(int(vi), float(ct[vi]))
                self.custom_sl[ct_idx]    = np.nan
                self.custom_target[ct_idx]= np.nan
                for j, vi in enumerate(ct_idx):
                    g = float(gross[j]); n_ = g - brok
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": "REENTRY_TARGET_HIT",
                        "level_breached": "CUSTOM_TARGET",
                        "level_px": round(float(ct[vi]), 2),
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(ep[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })
                buy_open = buy_open & ~np.isin(np.arange(self._n), ct_idx)

            # BUY — T1..T5 (first hit only — mirrors Algofinal first-target-exit)
            hit_m   = buy_open[None, :] & (price >= self.t) & (self.t > ep[None, :])
            hit_any = np.any(hit_m, axis=0)
            first_i = np.argmax(hit_m, axis=0)
            for i in range(5):
                m   = hit_any & (first_i == i) & ~np.isin(np.arange(self._n),
                                                            np.where(~buy_open)[0])
                idx = np.where(m)[0]
                if idx.size == 0:
                    continue
                gross = (self.t[i][idx] - ep[idx]) * qty_arr[idx]
                net   = gross - brok
                self.total_pnl[idx]   += net
                self.trade_count[idx] += 1
                self.win_count[idx]   += (net > 0).astype(np.int32)
                self.loss_count[idx]  += (net <= 0).astype(np.int32)
                self.in_position[idx]  = False
                self.is_buy[idx]       = False
                self.exited_today[idx] = True
                self.retreat_peak_reached[idx] = False
                self.retreat_65_alerted[idx]   = False
                self.retreat_45_alerted[idx]   = False
                for vi in idx:
                    self._reanchor_variation(int(vi), float(self.t[i][vi]))
                for j, vi in enumerate(idx):
                    g = float(gross[j]); n_ = g - brok
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": f"T{i+1}_HIT",
                        "level_breached": f"T{i+1}",
                        "level_px": round(float(self.t[i][vi]), 2),
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(ep[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })

        # SELL — custom target
        if sell_open.any():
            ct     = self.custom_target
            has_ct = sell_open & np.isfinite(ct)
            ct_idx = np.where(has_ct & (price <= ct) & (ct < ep))[0]
            if ct_idx.size > 0:
                gross = (ep[ct_idx] - ct[ct_idx]) * qty_arr[ct_idx]
                net   = gross - brok
                self.total_pnl[ct_idx]   += net
                self.trade_count[ct_idx] += 1
                self.win_count[ct_idx]   += (net > 0).astype(np.int32)
                self.loss_count[ct_idx]  += (net <= 0).astype(np.int32)
                self.in_position[ct_idx]  = False
                self.is_buy[ct_idx]       = False
                self.exited_today[ct_idx] = True
                self.retreat_peak_reached[ct_idx] = False
                self.retreat_65_alerted[ct_idx]   = False
                self.retreat_45_alerted[ct_idx]   = False
                for vi in ct_idx:
                    self._reanchor_variation(int(vi), float(ct[vi]))
                self.custom_sl[ct_idx]    = np.nan
                self.custom_target[ct_idx]= np.nan
                for j, vi in enumerate(ct_idx):
                    g = float(gross[j]); n_ = g - brok
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": "REENTRY_TARGET_HIT",
                        "level_breached": "CUSTOM_TARGET",
                        "level_px": round(float(ct[vi]), 2),
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(ep[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })
                sell_open = sell_open & ~np.isin(np.arange(self._n), ct_idx)

            # SELL — ST1..ST5
            hit_m   = sell_open[None, :] & (price <= self.st) & (self.st < ep[None, :])
            hit_any = np.any(hit_m, axis=0)
            first_i = np.argmax(hit_m, axis=0)
            for i in range(5):
                m   = hit_any & (first_i == i) & ~np.isin(np.arange(self._n),
                                                            np.where(~sell_open)[0])
                idx = np.where(m)[0]
                if idx.size == 0:
                    continue
                gross = (ep[idx] - self.st[i][idx]) * qty_arr[idx]
                net   = gross - brok
                self.total_pnl[idx]   += net
                self.trade_count[idx] += 1
                self.win_count[idx]   += (net > 0).astype(np.int32)
                self.loss_count[idx]  += (net <= 0).astype(np.int32)
                self.in_position[idx]  = False
                self.is_buy[idx]       = False
                self.exited_today[idx] = True
                self.retreat_peak_reached[idx] = False
                self.retreat_65_alerted[idx]   = False
                self.retreat_45_alerted[idx]   = False
                for vi in idx:
                    self._reanchor_variation(int(vi), float(self.st[i][vi]))
                for j, vi in enumerate(idx):
                    g = float(gross[j]); n_ = g - brok
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": f"ST{i+1}_HIT",
                        "level_breached": f"ST{i+1}",
                        "level_px": round(float(self.st[i][vi]), 2),
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(ep[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })

        # Refresh after T exits — retreat and SL must only act on STILL-open positions
        ep      = self.entry_price
        qty_arr = self.entry_qty.astype(np.float64)
        buy_open  = self.in_position & self.is_buy  & (ep > 0)
        sell_open = self.in_position & ~self.is_buy & (ep > 0)

        # ── 4. RETREAT 65/45/25  (NEW — mirrors handle_retreat_monitoring) ────
        # Priority: T-exits first (done above), then retreat, then SL.
        # This is the CORRECT Algofinal order.
        self._process_retreat(price, ts_str, brok, qty_arr, ep, buy_open, sell_open)

        # Refresh again after retreat may have exited some positions
        ep      = self.entry_price
        qty_arr = self.entry_qty.astype(np.float64)
        buy_open  = self.in_position & self.is_buy  & (ep > 0)
        sell_open = self.in_position & ~self.is_buy & (ep > 0)

        # ── 5. SL EXITS ───────────────────────────────────────────────────────
        # BUY SL
        if buy_open.any():
            buy_sl = np.where(np.isfinite(self.custom_sl), self.custom_sl, self.buy_sl)
            sl_m   = buy_open & (price <= buy_sl)
            sl_idx = np.where(sl_m)[0]
            if sl_idx.size > 0:
                # gross = (sl_price - entry_price) × qty  ← negative = loss
                gross = (buy_sl[sl_idx] - ep[sl_idx]) * qty_arr[sl_idx]
                net   = gross - brok
                self.total_pnl[sl_idx]   += net
                self.trade_count[sl_idx] += 1
                self.loss_count[sl_idx]  += 1
                self.in_position[sl_idx]  = False
                self.is_buy[sl_idx]       = False
                self.exited_today[sl_idx] = True
                self.retreat_peak_reached[sl_idx] = False
                self.retreat_65_alerted[sl_idx]   = False
                self.retreat_45_alerted[sl_idx]   = False
                for j, vi in enumerate(sl_idx):
                    g = float(gross[j]); n_ = g - brok
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": "BUY_SL",
                        "level_breached": "BUY_SL",
                        "level_px": round(float(buy_sl[vi]), 2),
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(ep[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })
                for vi in sl_idx:
                    self._reanchor_variation(int(vi), float(buy_sl[vi]))

        # SELL SL
        if sell_open.any():
            sell_sl = np.where(np.isfinite(self.custom_sl), self.custom_sl, self.sell_sl)
            sl_m    = sell_open & (price >= sell_sl)
            sl_idx  = np.where(sl_m)[0]
            if sl_idx.size > 0:
                gross = (ep[sl_idx] - sell_sl[sl_idx]) * qty_arr[sl_idx]
                net   = gross - brok
                self.total_pnl[sl_idx]   += net
                self.trade_count[sl_idx] += 1
                self.loss_count[sl_idx]  += 1
                self.in_position[sl_idx]  = False
                self.is_buy[sl_idx]       = False
                self.exited_today[sl_idx] = True
                self.retreat_peak_reached[sl_idx] = False
                self.retreat_65_alerted[sl_idx]   = False
                self.retreat_45_alerted[sl_idx]   = False
                for j, vi in enumerate(sl_idx):
                    g = float(gross[j]); n_ = g - brok
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": "SELL_SL",
                        "level_breached": "SELL_SL",
                        "level_px": round(float(sell_sl[vi]), 2),
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(ep[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })
                for vi in sl_idx:
                    self._reanchor_variation(int(vi), float(sell_sl[vi]))

    # ── Retreat monitoring (extracted for clarity) ────────────────────────────

    def _process_retreat(
        self,
        price:    float,
        ts_str:   str,
        brok:     float,
        qty_arr:  np.ndarray,
        ep:       np.ndarray,
        buy_open: np.ndarray,
        sell_open:np.ndarray,
    ) -> None:
        """
        65/45/25% retreat monitoring — exact mirror of handle_retreat_monitoring().

        BUY retreat:
          lvl_65 = retreat_entry_level + 0.65 × step   (65% of step toward T1)
          lvl_45 = retreat_entry_level + 0.45 × step
          lvl_25 = retreat_entry_level + 0.25 × step   ← EXIT HERE
          Guard: retreat_peak_reached must be True before any alert fires.
          Set peak_reached when price >= lvl_65 (price has been in profit zone).

        SELL retreat: mirror with directions flipped.

        After a 25% exit:
          Re-anchor: T1 → new buy_above, old buy_above → new sell_below.
          Reset retreat state. Set exited_today=False (allow immediate re-entry).
          gross = (exit_price - entry_price) × qty   BUY
          gross = (entry_price - exit_price) × qty   SELL
        """
        s = self.step_arr

        # ── BUY ───────────────────────────────────────────────────────────────
        if buy_open.any():
            el     = self.retreat_entry_level
            lvl_65 = el + 0.65 * s
            lvl_45 = el + 0.45 * s
            lvl_25 = el + 0.25 * s

            # Price reached T1 — full profit, clear retreat state
            at_t1 = buy_open & (price >= self.t[0])
            if at_t1.any():
                self.retreat_peak_reached[at_t1] = False
                self.retreat_65_alerted[at_t1]   = False
                self.retreat_45_alerted[at_t1]   = False

            # Price in ≥65% profit zone — mark peak reached, clear stale alerts
            peak_zone = buy_open & (price >= lvl_65) & (price < self.t[0])
            if peak_zone.any():
                self.retreat_peak_reached[peak_zone] = True
                # Clear stale retreat alerts when price returns to good zone
                recovering = peak_zone & (self.retreat_65_alerted | self.retreat_45_alerted)
                if recovering.any():
                    self.retreat_65_alerted[recovering] = False
                    self.retreat_45_alerted[recovering] = False

            # Only variations that have been in the profit zone AND have retreated
            peaked  = buy_open & self.retreat_peak_reached & (price < lvl_65)

            # 45% alert flag (no exit yet — mirrors Algofinal's two-phase approach)
            alert_45 = peaked & ~self.retreat_45_alerted & (price <= lvl_45)
            if alert_45.any():
                self.retreat_45_alerted[alert_45] = True

            # 65% alert flag
            alert_65 = peaked & ~self.retreat_65_alerted
            if alert_65.any():
                self.retreat_65_alerted[alert_65] = True

            # 25% EXIT — requires 45% to have been alerted (exact Algofinal condition)
            retreat_exit = peaked & self.retreat_45_alerted & (price <= lvl_25)
            rx_idx = np.where(retreat_exit)[0]
            if rx_idx.size > 0:
                # FORMULA (per spec): gross = x * 0.25 * qty  (always positive)
                # x = self.x_arr (Rs deviation for this symbol/variation)
                # (price - entry_level) can be negative if price drops below entry_level
                # so we use the canonical formula directly.
                gross = self.x_arr[rx_idx] * 0.25 * qty_arr[rx_idx]
                net   = gross - brok
                self.total_pnl[rx_idx]   += net
                self.trade_count[rx_idx] += 1
                wins = (net > 0)
                self.win_count[rx_idx]   += wins.astype(np.int32)
                self.loss_count[rx_idx]  += (~wins).astype(np.int32)

                # ── Re-anchor levels (mirrors _reanchor_after_retreat_exit BUY) ──
                # T1 → new buy_above, old buy_above → new sell_below
                new_ba = self.t[0][rx_idx].copy()
                new_sb = self.buy_above[rx_idx].copy()
                self.buy_above[rx_idx]  = new_ba
                self.sell_below[rx_idx] = new_sb
                for j in range(5):
                    self.t[j][rx_idx]  = new_ba + s[rx_idx] * (j + 1)
                    self.st[j][rx_idx] = new_sb - s[rx_idx] * (j + 1)
                self.buy_sl[rx_idx]  = new_ba - self.x_arr[rx_idx]
                self.sell_sl[rx_idx] = new_sb + self.x_arr[rx_idx]

                # Reset position + retreat state
                self.in_position[rx_idx]          = False
                self.is_buy[rx_idx]               = False
                self.exited_today[rx_idx]         = False  # allow immediate re-entry
                self.retreat_peak_reached[rx_idx] = False
                self.retreat_65_alerted[rx_idx]   = False
                self.retreat_45_alerted[rx_idx]   = False
                self.retreat_entry_level[rx_idx]  = 0.0
                self.custom_sl[rx_idx]    = np.nan
                self.custom_target[rx_idx]= np.nan

                for j, vi in enumerate(rx_idx):
                    g  = float(gross[j]); n_ = g - brok
                    lp = round(float(lvl_25[vi]), 2) if lvl_25.ndim > 0 else round(float(lvl_25), 2)
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": "BUY_RETREAT_25PCT",
                        "level_breached": "RETREAT_25PCT",
                        "level_px": lp,
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(self.retreat_entry_level[vi]), 2),
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })

        # ── SELL ──────────────────────────────────────────────────────────────
        if sell_open.any():
            el     = self.retreat_entry_level
            lvl_65 = el - 0.65 * s   # toward ST1 (prices decrease)
            lvl_45 = el - 0.45 * s
            lvl_25 = el - 0.25 * s   # EXIT

            at_st1 = sell_open & (price <= self.st[0])
            if at_st1.any():
                self.retreat_peak_reached[at_st1] = False
                self.retreat_65_alerted[at_st1]   = False
                self.retreat_45_alerted[at_st1]   = False

            peak_zone = sell_open & (price <= lvl_65) & (price > self.st[0])
            if peak_zone.any():
                self.retreat_peak_reached[peak_zone] = True
                recovering = peak_zone & (self.retreat_65_alerted | self.retreat_45_alerted)
                if recovering.any():
                    self.retreat_65_alerted[recovering] = False
                    self.retreat_45_alerted[recovering] = False

            peaked = sell_open & self.retreat_peak_reached & (price > lvl_65)

            alert_45 = peaked & ~self.retreat_45_alerted & (price >= lvl_45)
            if alert_45.any():
                self.retreat_45_alerted[alert_45] = True

            alert_65 = peaked & ~self.retreat_65_alerted
            if alert_65.any():
                self.retreat_65_alerted[alert_65] = True

            retreat_exit = peaked & self.retreat_45_alerted & (price >= lvl_25)
            rx_idx = np.where(retreat_exit)[0]
            if rx_idx.size > 0:
                # FORMULA (per spec): gross = x * 0.25 * qty  (always positive)
                gross = self.x_arr[rx_idx] * 0.25 * qty_arr[rx_idx]
                net   = gross - brok
                self.total_pnl[rx_idx]   += net
                self.trade_count[rx_idx] += 1
                wins = (net > 0)
                self.win_count[rx_idx]   += wins.astype(np.int32)
                self.loss_count[rx_idx]  += (~wins).astype(np.int32)

                # Re-anchor: ST1 → new sell_below, old sell_below → new buy_above
                new_sb = self.st[0][rx_idx].copy()
                new_ba = self.sell_below[rx_idx].copy()
                self.buy_above[rx_idx]  = new_ba
                self.sell_below[rx_idx] = new_sb
                for j in range(5):
                    self.t[j][rx_idx]  = new_ba + s[rx_idx] * (j + 1)
                    self.st[j][rx_idx] = new_sb - s[rx_idx] * (j + 1)
                self.buy_sl[rx_idx]  = new_ba - self.x_arr[rx_idx]
                self.sell_sl[rx_idx] = new_sb + self.x_arr[rx_idx]

                self.in_position[rx_idx]          = False
                self.is_buy[rx_idx]               = False
                self.exited_today[rx_idx]         = False
                self.retreat_peak_reached[rx_idx] = False
                self.retreat_65_alerted[rx_idx]   = False
                self.retreat_45_alerted[rx_idx]   = False
                self.retreat_entry_level[rx_idx]  = 0.0
                self.custom_sl[rx_idx]    = np.nan
                self.custom_target[rx_idx]= np.nan

                for j, vi in enumerate(rx_idx):
                    g  = float(gross[j]); n_ = g - brok
                    lp = round(float(lvl_25[vi]), 2) if lvl_25.ndim > 0 else round(float(lvl_25), 2)
                    self.trade_logs[vi].append({
                        "ts": ts_str, "sym": self.symbol, "event": "SELL_RETREAT_25PCT",
                        "level_breached": "RETREAT_25PCT",
                        "level_px": lp,
                        "price": price, "qty": int(qty_arr[vi]),
                        "entry_px": round(float(el[j]), 2),   # entry_level, not fill price
                        "gross": round(g, 2), "net": round(n_, 2),
                        "x": round(float(self._xv_np[vi]), 8),
                    })

        # Drawdown update at end of tick (after all exits for this tick)
        self.equity_peak  = np.maximum(self.equity_peak, self.total_pnl)
        dd                = self.equity_peak - self.total_pnl
        self.max_drawdown = np.maximum(self.max_drawdown, dd)

    # ── EOD square-off ────────────────────────────────────────────────────────

    def eod_square_off(self, price: float, ts: datetime) -> None:
        """Close all open positions at EOD price. Mirrors Algofinal eod_square_off."""
        ts_str  = ts.strftime("%Y-%m-%d %H:%M:%S IST")
        brok    = BROKERAGE_PER_SIDE * 2.0
        if (self.is_commodity or self.is_crypto) and USDT_TO_INR > 0:
            brok = brok / USDT_TO_INR       # convert brokerage to USDT (price is USDT)
        open_m  = self.in_position & (self.entry_qty > 0)
        if not open_m.any():
            return

        ep      = self.entry_price[open_m]
        qty_arr = self.entry_qty[open_m].astype(np.float64)
        is_buy  = self.is_buy[open_m]

        # BUY: gross = (price - entry) × qty  |  SELL: gross = (entry - price) × qty
        gross = np.where(is_buy, (price - ep) * qty_arr, (ep - price) * qty_arr)
        net   = gross - brok

        self.total_pnl[open_m]   += net
        self.trade_count[open_m] += 1
        self.win_count[open_m]   += (net > 0).astype(np.int32)
        self.loss_count[open_m]  += (net <= 0).astype(np.int32)
        self.in_position[open_m]  = False
        self.exited_today[open_m] = True
        # Reset retreat state
        self.retreat_peak_reached[open_m] = False
        self.retreat_65_alerted[open_m]   = False
        self.retreat_45_alerted[open_m]   = False

        for j, vi in enumerate(np.where(open_m)[0]):
            g = float(gross[j]); n_ = g - brok
            if self.is_commodity:
                evt = "EOD_SQUAREOFF_2330"
            elif self.symbol.upper() in _CRYPTO_SET:
                evt = "EOD_SQUAREOFF_REANCHOR"
            else:
                evt = "EOD_SQUAREOFF_1511"
            self.trade_logs[vi].append({
                "ts": ts_str, "sym": self.symbol, "event": evt,
                "level_breached": "EOD",
                "level_px": price,
                "price": price, "qty": int(qty_arr[j]),
                "entry_px": round(float(ep[j]), 2),
                "gross": round(g, 2), "net": round(n_, 2),
                "x": round(float(self._xv_np[vi]), 8),
            })

    # ── Daily reset ───────────────────────────────────────────────────────────

    def reset_for_new_day(self, new_prev_close: float) -> None:
        self.prev_close = new_prev_close
        self.x_arr      = new_prev_close * self._xv_np
        self.step_arr   = (self.x_arr * 0.6) if self.is_special else self.x_arr.copy()
        self._init_levels()
        self._init_position_state(self._n)
        self._levels_locked = False
        self.tick_count     = 0
        self.last_price     = 0.0

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        w_pnl: float = 0.45,
        w_winrate: float = 0.25,
        w_trade_count: float = 0.15,
        w_drawdown: float = 0.15,
    ) -> np.ndarray:
        """Multi-metric score per variation. Higher = better.

        Uses:
          - total P&L (higher better)
          - win-rate (higher better)
          - trade count (higher better, scaled)
          - max drawdown (lower better; inverted for scoring)
        """
        tc  = self.trade_count.astype(float)
        wc  = self.win_count.astype(float)
        pnl = self.total_pnl.astype(float)
        dd  = self.max_drawdown.astype(float)

        has_trades = tc > 0
        score = np.full(self._n, -1e18, dtype=np.float64)
        if not has_trades.any():
            return score

        # Valid slices for normalization (ignore no-trade variations).
        pnl_v = pnl[has_trades]
        tc_v  = tc[has_trades]
        wc_v  = wc[has_trades]
        dd_v  = dd[has_trades]

        pnl_min = pnl_v.min()
        pnl_max = pnl_v.max()
        pnl_den = pnl_max - pnl_min
        pnl_norm = (pnl - pnl_min) / pnl_den if pnl_den > 1e-9 else np.zeros(self._n)

        wr = np.where(tc > 0, wc / tc, 0.0)
        wr_v = wr[has_trades]
        wr_max = wr_v.max()
        wr_norm = wr / wr_max if wr_max > 1e-9 else np.zeros(self._n)

        tc_max = tc_v.max()
        tc_norm = tc / tc_max if tc_max > 0 else np.zeros(self._n)

        dd_max = dd_v.max()
        dd_norm = dd / dd_max if dd_max > 1e-9 else np.zeros(self._n)
        dd_score = 1.0 - dd_norm  # lower drawdown => higher score

        score_valid = (
            w_pnl * pnl_norm
            + w_winrate * wr_norm
            + w_trade_count * tc_norm
            + w_drawdown * dd_score
        )

        score[has_trades] = score_valid[has_trades]
        return score

    def best(
        self,
        w_pnl: float = 0.45,
        w_winrate: float = 0.25,
        w_trade_count: float = 0.15,
        w_drawdown: float = 0.15,
    ) -> dict:
        sc  = self.score(w_pnl, w_winrate, w_trade_count, w_drawdown)
        idx = int(np.argmax(sc))
        tc  = int(self.trade_count[idx])
        wc  = int(self.win_count[idx])
        lc  = int(self.loss_count[idx])
        pnl = float(self.total_pnl[idx])
        ddm = float(self.max_drawdown[idx])
        xv  = float(self._xv_np[idx])
        return {
            "symbol":       self.symbol,
            "prev_close":   self.prev_close,
            "best_x":       round(xv, 8),
            "score":        round(float(sc[idx]), 6),
            "total_pnl":    round(pnl, 2),
            "trade_count":  tc,
            "win_count":    wc,
            "loss_count":   lc,
            "win_rate_pct": round(wc / tc * 100, 2) if tc > 0 else 0.0,
            "max_drawdown": round(ddm, 2),
            "avg_return":   round(pnl / tc, 2) if tc > 0 else 0.0,
            "last_event":   (
                self.trade_logs[idx][-1]["event"] + " @ " +
                str(self.trade_logs[idx][-1].get("level_px", "—"))
            ) if self.trade_logs[idx] else "—",
        }

    def row_data(self) -> dict:
        """Rich table row. Metric fields blank until first trade."""
        total = int(self.trade_count.sum())
        if total == 0:
            return {
                "symbol": self.symbol,
                "prev_close": f"{self.prev_close:.2f}",
                "last_price": f"{self.last_price:.2f}" if self.last_price else "—",
                "best_x": "—", "total_pnl": "—", "win_rate_pct": "—",
                "trade_count": "—", "win_count": "—", "loss_count": "—",
                "last_event": "—", "vs_current": "—", "has_data": False,
            }
        b = self.best()
        return {
            "symbol":       b["symbol"],
            "prev_close":   f"{b['prev_close']:.2f}",
            "last_price":   f"{self.last_price:.2f}" if self.last_price else "—",
            "best_x":       f"{b['best_x']:.6f}",
            "total_pnl":    f"₹{b['total_pnl']:.2f}",
            "win_rate_pct": f"{b['win_rate_pct']:.1f}%",
            "trade_count":  str(b["trade_count"]),
            "win_count":    str(b["win_count"]),
            "loss_count":   str(b["loss_count"]),
            "last_event":   b.get("last_event", "—"),
            "vs_current":   "—",
            "has_data":     True,
        }

    # ── Live-state snapshot (for x.py aggregation) ───────────────────────────

    def dump_state(self, *, compact: bool = True, max_points: int = 480) -> dict:
        """Return a JSON-serialisable snapshot for x.py / cross-scanner fusion to read.
        v10.5 FIX: best_x is 0.0 when no trades to avoid showing X_MIN as 'best'.
        Uses multi-metric score (P&L + win-rate + trade-count + drawdown).
        """
        import numpy as _np
        tc_arr  = self.trade_count.astype(float)
        wc_arr  = self.win_count.astype(float)
        pnl_arr = self.total_pnl.astype(float)
        dd_arr  = self.max_drawdown.astype(float)

        # For dashboard responsiveness: do NOT ship 16k/32k arrays per symbol unless requested.
        # Default is compact, downsampled arrays for charts; full detail lives in breach_logs/*.jsonl.
        if os.getenv("DUMP_FULL_SWEEP_ARRAYS", "0").strip() in ("1", "true", "True"):
            compact = False

        def _downsample(arr: _np.ndarray) -> List[float]:
            try:
                n = int(arr.shape[0])
                if not compact or n <= max_points or max_points <= 0:
                    return arr.astype(float).tolist()
                step = max(1, n // int(max_points))
                return arr[::step].astype(float).tolist()
            except Exception:
                return arr.astype(float).tolist()

        xv_full = self._xv_np.astype(float)
        xv_ds = _downsample(xv_full)
        pnl_ds = _downsample(pnl_arr)
        tc_ds = _downsample(tc_arr)

        total_tc = int(tc_arr.sum())

        # v10.5 FIX: only compute best when actual trades have occurred
        if total_tc == 0:
            # No trades yet — return zero best_x so dashboard shows "—" not X_MIN
            zero_breach = {**{f"T{i+1}": 0 for i in range(5)}, **{f"ST{i+1}": 0 for i in range(5)}}
            return {
                "symbol":           self.symbol,
                "prev_close":       self.prev_close,
                "last_price":       self.last_price,
                "tick_count":       self.tick_count,
                "x_values":         xv_ds,
                "total_pnl":        pnl_ds,
                "trade_count":      [int(x) for x in tc_ds],
                "x_values_full_len": int(len(xv_full)),
                "compact":          bool(compact),
                "win_count":        self.win_count.tolist(),
                "loss_count":       self.loss_count.tolist(),
                "max_drawdown":    self.max_drawdown.tolist(),
                "best_max_drawdown": 0.0,
                "best_breach_ticks_total": 0,
                "best_breach_ticks_by_level": zero_breach,
                "best_first_breach_tick_by_level": zero_breach,
                "best_x":           0.0,   # 0 = no data yet
                "best_pnl":         0.0,
                "best_win_rate":    0.0,
                "best_trade_count": 0,
                "has_trades":       False,
            }

        # Multi-metric scoring (same logic as self.best()).
        score_arr = self.score()
        best_idx  = int(_np.argmax(score_arr))
        best_tc   = int(self.trade_count[best_idx])
        best_wc   = int(self.win_count[best_idx])
        best_xv   = float(self._xv_np[best_idx])
        best_pnl  = float(pnl_arr[best_idx])
        best_wr   = (best_wc / best_tc * 100.0) if best_tc > 0 else 0.0
        best_dd   = float(dd_arr[best_idx]) if dd_arr is not None else 0.0

        t_ticks  = self.breach_ticks_T[:, best_idx]
        st_ticks = self.breach_ticks_ST[:, best_idx]
        fT_ticks  = self.first_breach_tick_T[:, best_idx]
        fST_ticks = self.first_breach_tick_ST[:, best_idx]

        best_breach_ticks_by_level = {
            **{f"T{i+1}": int(t_ticks[i]) for i in range(5)},
            **{f"ST{i+1}": int(st_ticks[i]) for i in range(5)},
        }
        best_breach_total = int(sum(best_breach_ticks_by_level.values()))
        best_first_breach_tick_by_level = {
            **{f"T{i+1}": int(fT_ticks[i]) for i in range(5)},
            **{f"ST{i+1}": int(fST_ticks[i]) for i in range(5)},
        }
        return {
            "symbol":           self.symbol,
            "prev_close":       self.prev_close,
            "last_price":       self.last_price,
            "tick_count":       self.tick_count,
            "x_values":         xv_ds,
            "total_pnl":        pnl_ds,
            "trade_count":      [int(x) for x in tc_ds],
            "x_values_full_len": int(len(xv_full)),
            "compact":          bool(compact),
            "win_count":        self.win_count.tolist(),
            "loss_count":       self.loss_count.tolist(),
            "max_drawdown":     self.max_drawdown.tolist(),
            "best_max_drawdown": round(best_dd, 2),
            "best_breach_ticks_total": best_breach_total,
            "best_breach_ticks_by_level": best_breach_ticks_by_level,
            "best_first_breach_tick_by_level": best_first_breach_tick_by_level,
            # Cross-scanner fusion keys:
            "best_x":           round(best_xv, 8),
            "best_pnl":         round(best_pnl, 2),
            "best_win_rate":    round(best_wr, 2),
            "best_trade_count": best_tc,
            "has_trades":       True,
        }


# ════════════════════════════════════════════════════════════════════════════
# EOD WRITERS
# ════════════════════════════════════════════════════════════════════════════

def _write_one_x_csv(args: tuple) -> None:
    """Worker: write one CSV file for a single X variation. Runs in subprocess."""
    x_val, trades, out_dir, date_str = args
    fname = os.path.join(out_dir, f"x_{x_val:.6f}_trades.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(fname, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["ts","sym","event","level_breached","level_px",
                        "price","qty","entry_px","gross","net","x"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(trades)


def save_results(
    sweeps:   Dict[str, "StockSweep"],
    date_str: str,
    out_dir:  str,
    *,
    n_workers: int = CPU_WORKERS,
    scanner_name: str = "Scanner",
) -> None:
    """
    Write EOD results:
      - prices.csv         (tick log — already written live by PriceSubscriber)
      - summary.xlsx       (best X per stock)
      - live_state.json    (for x.py aggregation)
      - per-X CSV files    (parallel via ProcessPool)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Summary XLSX ──────────────────────────────────────────────────────────
    summary_rows = []
    for sym, sw in sweeps.items():
        if sw.trade_count.sum() == 0:
            continue
        b = sw.best()
        summary_rows.append(b)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        xl = os.path.join(out_dir, f"summary_{date_str}.xlsx")
        df.to_excel(xl, index=False)
        log.info("summary.xlsx → %s", xl)

    # ── Live state JSON for x.py ──────────────────────────────────────────────
    state = {
        "scanner":    scanner_name,
        "date":       date_str,
        "written_at": now_ist().isoformat(),
        "sweeps":     {sym: sw.dump_state() for sym, sw in sweeps.items()},
    }
    _state_path = os.path.join(out_dir, "live_state.json")
    _state_tmp  = _state_path + ".tmp"
    with open(_state_tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, separators=(",", ":"))
    _replaced = False
    for _ in range(8):
        try:
            os.replace(_state_tmp, _state_path)   # atomic write
            _replaced = True
            break
        except PermissionError:
            time.sleep(0.05)
        except Exception:
            break
    if not _replaced:
        # Windows can briefly lock files during antivirus/indexing; best-effort fallback.
        try:
            with open(_state_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, separators=(",", ":"))
            try:
                if os.path.exists(_state_tmp):
                    os.remove(_state_tmp)
            except Exception:
                pass
        except Exception:
            pass

    # ── Breach logs (disk) ──────────────────────────────────────────────────
    # We persist "full breached levels" per X-variation to disk so the public
    # dashboard can remain responsive (live_state.json stays compact).
    #
    # Format: one JSONL line per (variation, level) where the level was breached
    # at least once. Includes:
    #   - x multiplier
    #   - breach_ticks (count of ticks beyond the level while open)
    #   - first_breach_tick (tick index inside this StockSweep)
    if os.getenv("WRITE_BREACH_LOGS", "1") == "1":
        breach_dir = os.path.join(out_dir, "breach_logs")
        os.makedirs(breach_dir, exist_ok=True)
        max_lines = int(os.getenv("BREACH_LOG_MAX_LINES", "250000"))

        for sym, sw in sweeps.items():
            # Skip if a symbol never traded (reduces empty logs).
            if sw.trade_count.sum() == 0:
                continue

            # Rotate output files after max_lines to avoid runaway single files.
            part_idx = 0
            written  = 0
            fh = None

            def _open_next() -> None:
                nonlocal part_idx, written, fh
                if fh:
                    fh.close()
                fname = os.path.join(breach_dir, f"breach_{sym}_part{part_idx}.jsonl")
                fh = open(fname, "w", encoding="utf-8")
                part_idx += 1
                written = 0

            _open_next()

            xv = sw._xv_np  # (N,)

            # Helper to write a line with minimal overhead.
            def _write_line(obj: dict) -> None:
                nonlocal written
                fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
                written += 1

            # T1..T5
            for lvl_i in range(5):
                ticks_arr = sw.breach_ticks_T[lvl_i]
                first_arr = sw.first_breach_tick_T[lvl_i]
                nz = np.nonzero(ticks_arr)[0]
                for vi in nz:
                    if written >= max_lines:
                        _open_next()
                    _write_line({
                        "sym": sym,
                        "level": f"T{lvl_i+1}",
                        "x": round(float(xv[vi]), 8),
                        "breach_ticks": int(ticks_arr[vi]),
                        "first_breach_tick": int(first_arr[vi]),
                    })

            # ST1..ST5
            for lvl_i in range(5):
                ticks_arr = sw.breach_ticks_ST[lvl_i]
                first_arr = sw.first_breach_tick_ST[lvl_i]
                nz = np.nonzero(ticks_arr)[0]
                for vi in nz:
                    if written >= max_lines:
                        _open_next()
                    _write_line({
                        "sym": sym,
                        "level": f"ST{lvl_i+1}",
                        "x": round(float(xv[vi]), 8),
                        "breach_ticks": int(ticks_arr[vi]),
                        "first_breach_tick": int(first_arr[vi]),
                    })

            if fh:
                fh.close()

    # ── Per-X CSV files (parallel) ────────────────────────────────────────────
    # Collect all unique X values across symbols
    all_x: Dict[float, List[dict]] = {}
    for sw in sweeps.values():
        for vi in range(sw._n):
            if not sw.trade_logs[vi]:
                continue
            xv = round(float(sw._xv_np[vi]), 6)
            all_x.setdefault(xv, []).extend(sw.trade_logs[vi])

    if all_x:
        args_list = [(xv, rows, out_dir, date_str) for xv, rows in all_x.items()]
        try:
            # ThreadPoolExecutor: 3-5x faster than ProcessPool for I/O-bound CSV writes
            # Avoids Windows process-spawn overhead entirely
            with ThreadPoolExecutor(max_workers=min(CPU_WORKERS, len(args_list), 8)) as pool:
                list(pool.map(_write_one_x_csv, args_list, chunksize=20))
        except Exception:
            for a in args_list:
                _write_one_x_csv(a)
        log.info("Wrote %d per-X CSV files → %s", len(args_list), out_dir)


def spill_trade_logs_to_disk(sweeps: Dict[str, "StockSweep"],
                              date_str: str,
                              ram_limit: int = 200) -> int:
    """Trim in-memory trade logs when they exceed ram_limit entries per variation."""
    evicted = 0
    for sw in sweeps.values():
        for vi in range(sw._n):
            log_len = len(sw.trade_logs[vi])
            if log_len > ram_limit:
                sw.trade_logs[vi] = sw.trade_logs[vi][-100:]
                evicted += log_len - 100
    return evicted
