"""
vectorsweep.py
==============
Author  : Ridhaant Ajoy Thackur
Project : vectorsweep
License : MIT

Vectorised parameter sweep engine for quantitative trading strategies.

Evaluates thousands of (entry_threshold, stop_loss, target) combinations
simultaneously using NumPy arrays, with optional Numba JIT and CuPy/CUDA
acceleration.  Designed for the exact sweep pattern used in real NSE equity
and crypto trading systems.

Core idea:
  Given a price series and N strategy parameter variants, compute P&L for
  ALL N variants in a single vectorised pass — rather than N sequential loops.

Acceleration tiers (auto-detected, falls back gracefully):
  1. CuPy  (NVIDIA CUDA 12) — all arrays on GPU VRAM; ~1 ms per tick for 32K variants
  2. Numba JIT              — LLVM-compiled CPU kernels; ~5–10× faster than pure NumPy
  3. NumPy                  — always available; baseline

Usage:
    from vectorsweep import SweepEngine, SweepConfig, gen_x_values

    cfg = SweepConfig(
        x_min=0.005, x_max=0.020, x_step=0.0001,
        budget=100_000, brokerage_per_side=10.0,
    )
    engine = SweepEngine(cfg)
    results = engine.run(price_series=my_prices)  # list[SweepResult]
    best = max(results, key=lambda r: r.net_pnl)
    print(f"Best X={best.x_val:.5f}  net_pnl=₹{best.net_pnl:,.0f}  trades={best.trade_count}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("vectorsweep")

# Exact decimal precision to avoid float64 step-drift (mirrors sweep_core.py)
getcontext().prec = 28

# ── Acceleration backend detection ────────────────────────────────────────────
_BACKEND = "numpy"
_cp = None
_numba_jit = None

try:
    import cupy as _cupy
    _cupy.cuda.Device(0).use()
    _test = _cupy.zeros(1024, dtype=_cupy.float32)
    _test += 1
    assert float(_test.sum()) == 1024.0
    del _test
    _cp = _cupy
    _BACKEND = "cupy"
    log.info("VectorSweep backend: CuPy (CUDA)")
except Exception:
    pass

if _BACKEND == "numpy":
    try:
        from numba import njit, prange as _prange

        @njit(parallel=True, cache=True, fastmath=True)
        def _numba_tick_kernel(
            prices_arr,           # (T,)       float64 — full price series
            buy_above_arr,        # (N,)       float64 — entry levels
            sell_below_arr,       # (N,)       float64
            sl_buy_arr,           # (N,)       float64
            sl_sell_arr,          # (N,)       float64
            t1_buy_arr,           # (N,)       float64 — first target
            t1_sell_arr,          # (N,)       float64
            pnl_arr,              # (N,)       float64 — cumulative P&L (in-place)
            trade_count_arr,      # (N,)       int64   — trade count (in-place)
            qty_arr,              # (N,)       float64
            brokerage,            # scalar     float64
        ):
            N = buy_above_arr.shape[0]
            T = prices_arr.shape[0]
            for i in _prange(N):
                in_trade = False
                entry_px = 0.0
                is_buy = False
                for t in range(T):
                    px = prices_arr[t]
                    qty = qty_arr[i]
                    if not in_trade:
                        if px >= buy_above_arr[i]:
                            in_trade = True
                            entry_px = px
                            is_buy = True
                        elif px <= sell_below_arr[i]:
                            in_trade = True
                            entry_px = px
                            is_buy = False
                    else:
                        if is_buy:
                            if px >= t1_buy_arr[i]:  # target hit
                                gross = (t1_buy_arr[i] - entry_px) * qty
                                pnl_arr[i] += gross - brokerage * 2
                                trade_count_arr[i] += 1
                                in_trade = False
                            elif px <= sl_buy_arr[i]:  # SL hit
                                gross = (sl_buy_arr[i] - entry_px) * qty
                                pnl_arr[i] += gross - brokerage * 2
                                trade_count_arr[i] += 1
                                in_trade = False
                        else:
                            if px <= t1_sell_arr[i]:  # sell target
                                gross = (entry_px - t1_sell_arr[i]) * qty
                                pnl_arr[i] += gross - brokerage * 2
                                trade_count_arr[i] += 1
                                in_trade = False
                            elif px >= sl_sell_arr[i]:  # sell SL
                                gross = (entry_px - sl_sell_arr[i]) * qty
                                pnl_arr[i] += gross - brokerage * 2
                                trade_count_arr[i] += 1
                                in_trade = False

        _numba_jit = _numba_tick_kernel
        _BACKEND = "numba"
        log.info("VectorSweep backend: Numba JIT")
    except ImportError:
        log.info("VectorSweep backend: NumPy (install numba for 5-10× speedup)")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & RESULT TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    """
    Parameter sweep configuration.

    x_min / x_max / x_step define the range of entry-deviation values.
    Each X value generates:
      buy_above  = prev_close + X
      sell_below = prev_close - X
      sl_buy     = buy_above  - X
      sl_sell    = sell_below + X
      t1_buy     = buy_above  + X       (first target)
      t1_sell    = sell_below - X
    """
    x_min: float = 0.005
    x_max: float = 0.050
    x_step: float = 0.0001
    budget: float = 100_000.0          # ₹ per position
    brokerage_per_side: float = 10.0   # flat ₹ per side (matches AlgoStack)
    prev_close: float = 100.0          # anchor price (updated per session)
    max_workers: int = 4               # ProcessPoolExecutor workers for CPU path


@dataclass
class SweepResult:
    x_val: float
    net_pnl: float
    gross_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    buy_above: float
    sell_below: float
    sl_buy: float
    sl_sell: float


# ══════════════════════════════════════════════════════════════════════════════
# X-VALUE GENERATOR  (exact Decimal steps — mirrors sweep_core.gen_x_values)
# ══════════════════════════════════════════════════════════════════════════════

def gen_x_values(x_min: float, x_max: float, step: float) -> np.ndarray:
    """
    Generate inclusive X values with exact decimal steps (no float64 drift).

    >>> gen_x_values(0.008, 0.009, 1e-6).shape[0]
    1001
    """
    d_min  = Decimal(str(x_min))
    d_max  = Decimal(str(x_max))
    d_step = Decimal(str(step))
    values = []
    v = d_min
    while v <= d_max + d_step * Decimal("0.5"):
        values.append(float(v))
        v += d_step
    return np.array(values, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL ARRAYS
# ══════════════════════════════════════════════════════════════════════════════

def _build_level_arrays(
    x_values: np.ndarray, prev_close: float
) -> Tuple[np.ndarray, ...]:
    """
    Vectorised level generation for all N X-values simultaneously.

    Returns (buy_above, sell_below, sl_buy, sl_sell, t1_buy, t1_sell, qty).
    """
    pc = prev_close
    buy_above  = pc + x_values
    sell_below = pc - x_values
    sl_buy     = buy_above  - x_values
    sl_sell    = sell_below + x_values
    t1_buy     = buy_above  + x_values
    t1_sell    = sell_below - x_values
    qty        = np.floor(100_000.0 / np.maximum(buy_above, 0.01)).astype(np.float64)
    return buy_above, sell_below, sl_buy, sl_sell, t1_buy, t1_sell, qty


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY SWEEP KERNEL  (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def _numpy_sweep(
    prices: np.ndarray,           # (T,) float64
    buy_above: np.ndarray,        # (N,)
    sell_below: np.ndarray,       # (N,)
    sl_buy: np.ndarray,           # (N,)
    sl_sell: np.ndarray,          # (N,)
    t1_buy: np.ndarray,           # (N,)
    t1_sell: np.ndarray,          # (N,)
    qty: np.ndarray,              # (N,)
    brokerage: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy tick-by-tick sweep over the full price series.

    Iterates over each price tick once; uses boolean masks to update all N
    variants simultaneously.  Correct for single-entry-per-variant logic.
    """
    N = buy_above.shape[0]
    pnl = np.zeros(N, dtype=np.float64)
    trades = np.zeros(N, dtype=np.int64)
    in_trade = np.zeros(N, dtype=bool)
    is_buy = np.zeros(N, dtype=bool)
    entry_px = np.zeros(N, dtype=np.float64)

    for px in prices:
        # ── Entry logic (no open trade) ────────────────────────────────────
        can_enter = ~in_trade
        buy_entry  = can_enter & (px >= buy_above)
        sell_entry = can_enter & (~buy_entry) & (px <= sell_below)
        in_trade   = in_trade | buy_entry | sell_entry
        is_buy     = np.where(buy_entry, True,  np.where(sell_entry, False, is_buy))
        entry_px   = np.where(buy_entry | sell_entry, px, entry_px)

        # ── Exit logic (in trade) ──────────────────────────────────────────
        buy_t1_hit   = in_trade &  is_buy  & (px >= t1_buy)
        buy_sl_hit   = in_trade &  is_buy  & (px <= sl_buy)
        sell_t1_hit  = in_trade & ~is_buy  & (px <= t1_sell)
        sell_sl_hit  = in_trade & ~is_buy  & (px >= sl_sell)
        any_exit     = buy_t1_hit | buy_sl_hit | sell_t1_hit | sell_sl_hit

        # Compute gross P&L for exits
        buy_exit_pnl  = np.where(
            buy_t1_hit, (t1_buy - entry_px) * qty,
            np.where(buy_sl_hit, (sl_buy - entry_px) * qty, 0.0),
        )
        sell_exit_pnl = np.where(
            sell_t1_hit, (entry_px - t1_sell) * qty,
            np.where(sell_sl_hit, (entry_px - sl_sell) * qty, 0.0),
        )
        gross = np.where(is_buy, buy_exit_pnl, sell_exit_pnl)
        net   = gross - brokerage * 2.0

        pnl    = np.where(any_exit, pnl + net, pnl)
        trades = np.where(any_exit, trades + 1, trades)
        in_trade = np.where(any_exit, False, in_trade)

    return pnl, trades


# ══════════════════════════════════════════════════════════════════════════════
# CUPY SWEEP KERNEL  (GPU path)
# ══════════════════════════════════════════════════════════════════════════════

def _cupy_sweep(
    prices: np.ndarray,
    buy_above: np.ndarray,
    sell_below: np.ndarray,
    sl_buy: np.ndarray,
    sl_sell: np.ndarray,
    t1_buy: np.ndarray,
    t1_sell: np.ndarray,
    qty: np.ndarray,
    brokerage: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated sweep using CuPy arrays on NVIDIA VRAM."""
    xp = _cp
    g_ba  = xp.asarray(buy_above,  dtype=xp.float32)
    g_sb  = xp.asarray(sell_below, dtype=xp.float32)
    g_slb = xp.asarray(sl_buy,     dtype=xp.float32)
    g_sls = xp.asarray(sl_sell,    dtype=xp.float32)
    g_t1b = xp.asarray(t1_buy,     dtype=xp.float32)
    g_t1s = xp.asarray(t1_sell,    dtype=xp.float32)
    g_qty = xp.asarray(qty,        dtype=xp.float32)

    N = g_ba.shape[0]
    g_pnl    = xp.zeros(N, dtype=xp.float32)
    g_trades = xp.zeros(N, dtype=xp.int32)
    g_intrd  = xp.zeros(N, dtype=bool)
    g_isbuy  = xp.zeros(N, dtype=bool)
    g_entry  = xp.zeros(N, dtype=xp.float32)

    brok = xp.float32(brokerage)

    for px_np in prices:
        px = xp.float32(px_np)

        can_enter  = ~g_intrd
        buy_entry  = can_enter & (px >= g_ba)
        sell_entry = can_enter & ~buy_entry & (px <= g_sb)
        g_intrd    = g_intrd | buy_entry | sell_entry
        g_isbuy    = xp.where(buy_entry, True,  xp.where(sell_entry, False, g_isbuy))
        g_entry    = xp.where(buy_entry | sell_entry, px, g_entry)

        bt1  = g_intrd &  g_isbuy & (px >= g_t1b)
        bsl  = g_intrd &  g_isbuy & (px <= g_slb)
        st1  = g_intrd & ~g_isbuy & (px <= g_t1s)
        ssl  = g_intrd & ~g_isbuy & (px >= g_sls)
        ex   = bt1 | bsl | st1 | ssl

        b_pnl  = xp.where(bt1, (g_t1b - g_entry) * g_qty,
                 xp.where(bsl, (g_slb - g_entry) * g_qty, xp.float32(0)))
        s_pnl  = xp.where(st1, (g_entry - g_t1s) * g_qty,
                 xp.where(ssl, (g_entry - g_sls) * g_qty, xp.float32(0)))
        gross  = xp.where(g_isbuy, b_pnl, s_pnl)

        g_pnl    = xp.where(ex, g_pnl + gross - brok * 2, g_pnl)
        g_trades = xp.where(ex, g_trades + 1, g_trades)
        g_intrd  = xp.where(ex, False, g_intrd)

    return xp.asnumpy(g_pnl).astype(np.float64), xp.asnumpy(g_trades).astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP ENGINE  (public API)
# ══════════════════════════════════════════════════════════════════════════════

class SweepEngine:
    """
    Vectorised strategy parameter sweep engine.

    Evaluates all X-value variants against a price series in a single pass.
    Backend is auto-selected: CuPy → Numba → NumPy.

    Example
    -------
    >>> import numpy as np
    >>> prices = np.random.normal(100, 0.5, 390).cumsum()   # synthetic intraday
    >>> cfg = SweepConfig(x_min=0.5, x_max=5.0, x_step=0.1, prev_close=prices[0])
    >>> engine = SweepEngine(cfg)
    >>> results = engine.run(prices)
    >>> best = max(results, key=lambda r: r.net_pnl)
    >>> print(f"Best X={best.x_val:.2f}  P&L=₹{best.net_pnl:,.0f}")
    """

    def __init__(self, config: SweepConfig) -> None:
        self.cfg = config
        self._x_values = gen_x_values(config.x_min, config.x_max, config.x_step)
        log.info(
            "SweepEngine init | backend=%s | %d X-values | range=[%.5f, %.5f]",
            _BACKEND,
            len(self._x_values),
            self._x_values[0],
            self._x_values[-1],
        )

    @property
    def backend(self) -> str:
        return _BACKEND

    @property
    def n_variants(self) -> int:
        return len(self._x_values)

    def run(
        self,
        price_series: np.ndarray,
        prev_close: Optional[float] = None,
    ) -> List[SweepResult]:
        """
        Run the full sweep over price_series.

        Parameters
        ----------
        price_series : array_like, shape (T,)
            Sequence of prices (e.g. 1-minute OHLCV close prices for one session).
        prev_close : float, optional
            Previous-day close used to anchor entry levels.
            Defaults to config.prev_close.

        Returns
        -------
        list[SweepResult] — one per X-value, sorted by net_pnl descending.
        """
        pc = prev_close if prev_close is not None else self.cfg.prev_close
        prices = np.asarray(price_series, dtype=np.float64)

        # Build level arrays for all N variants
        buy_above, sell_below, sl_buy, sl_sell, t1_buy, t1_sell, qty = \
            _build_level_arrays(self._x_values, pc)

        t0 = time.perf_counter()

        if _BACKEND == "cupy":
            pnl, trades = _cupy_sweep(
                prices, buy_above, sell_below, sl_buy, sl_sell,
                t1_buy, t1_sell, qty, self.cfg.brokerage_per_side,
            )
        elif _BACKEND == "numba" and _numba_jit is not None:
            pnl_arr    = np.zeros(len(self._x_values), dtype=np.float64)
            trade_arr  = np.zeros(len(self._x_values), dtype=np.int64)
            _numba_jit(
                prices, buy_above, sell_below, sl_buy, sl_sell,
                t1_buy, t1_sell, pnl_arr, trade_arr, qty,
                self.cfg.brokerage_per_side,
            )
            pnl, trades = pnl_arr, trade_arr
        else:
            pnl, trades = _numpy_sweep(
                prices, buy_above, sell_below, sl_buy, sl_sell,
                t1_buy, t1_sell, qty, self.cfg.brokerage_per_side,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        evals = len(self._x_values) * len(prices)
        log.info(
            "Sweep complete | %d variants × %d ticks = %s evals | %.2f ms | backend=%s",
            len(self._x_values), len(prices), f"{evals:,}", elapsed_ms, _BACKEND,
        )

        results: List[SweepResult] = []
        for i, x in enumerate(self._x_values):
            gross = pnl[i] + self.cfg.brokerage_per_side * 2 * trades[i]
            results.append(
                SweepResult(
                    x_val=round(float(x), 8),
                    net_pnl=float(pnl[i]),
                    gross_pnl=float(gross),
                    trade_count=int(trades[i]),
                    win_count=0,    # extend: track per-trade outcomes
                    loss_count=0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    buy_above=float(buy_above[i]),
                    sell_below=float(sell_below[i]),
                    sl_buy=float(sl_buy[i]),
                    sl_sell=float(sl_sell[i]),
                )
            )

        results.sort(key=lambda r: r.net_pnl, reverse=True)
        return results

    def benchmark(self, n_ticks: int = 390) -> Dict[str, float]:
        """
        Run a synthetic benchmark and return timing stats.

        Parameters
        ----------
        n_ticks : int
            Number of synthetic price ticks to generate (390 = full NSE session).
        """
        rng = np.random.default_rng(42)
        prices = (self.cfg.prev_close + rng.normal(0, 0.3, n_ticks)).cumsum() / n_ticks
        prices = np.maximum(prices, 0.01)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            self.run(prices)
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "backend": _BACKEND,
            "n_variants": self.n_variants,
            "n_ticks": n_ticks,
            "evaluations_per_tick": self.n_variants,
            "total_evaluations": self.n_variants * n_ticks,
            "min_ms": round(min(times), 3),
            "mean_ms": round(sum(times) / len(times), 3),
            "max_ms": round(max(times), 3),
        }
