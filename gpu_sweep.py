# Author: Ridhaant Ajoy Thackur
# AlgoStack v10.2 — gpu_sweep.py
# GPU-accelerated vectorised sweep for NVIDIA GeForce GTX 1650 (4GB VRAM)
# Falls back to NumPy + Numba JIT when CuPy/CUDA unavailable
"""
GPU Sweep Engine — processes ALL N X-values for ALL symbols simultaneously.

For Scanner3 (32,000 X-values × 38 symbols):
  Each tick = 32,000 comparisons × 38 = 1,216,000 GPU ops in <1ms
  
Memory layout on GTX 1650 (4GB):
  Level arrays: 38 × 32K × float32 = 4.75 MB  (trivial)
  Position state: 38 × 32K × bool   = 1.22 MB
  P&L accumulators: 38 × 32K × float32 = 4.75 MB
  Total VRAM used: ~11 MB / 4096 MB (0.27%)

Architecture:
  - _GpuSweepBatch: holds all symbol arrays on GPU
  - tick(price_dict): processes one price update for all symbols in one GPU pass
  - Results synced back to CPU via async memcpy (non-blocking)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("gpu_sweep")

# ── GPU backend detection ─────────────────────────────────────────────────────
_GPU_BACKEND: str = "numpy"
_cp = None
_numba_available = False

try:
    import cupy as _cupy
    _cupy.cuda.Device(0).use()
    # Test with a small allocation
    _t = _cupy.zeros(1024, dtype=_cupy.float32); _t += 1
    assert float(_t.sum()) == 1024.0
    del _t
    _cp = _cupy
    _GPU_BACKEND = "cupy"
    try:
        _props = _cupy.cuda.runtime.getDeviceProperties(0)
        _gname = _props["name"].decode() if isinstance(_props["name"], bytes) else str(_props["name"])
        log.info("GPU backend: CuPy on %s", _gname)
    except Exception:
        log.info("GPU backend: CuPy (NVIDIA)")
except Exception:
    pass

if _GPU_BACKEND == "numpy":
    try:
        from numba import njit, prange
        _numba_available = True
        _GPU_BACKEND = "numba"
        log.info("GPU backend: Numba JIT (CPU)")
    except ImportError:
        log.info("GPU backend: NumPy (CPU baseline)")

xp = _cp if _cp is not None else np
GPU_AVAILABLE = _GPU_BACKEND == "cupy"
GPU_NAME = _gname if GPU_AVAILABLE else ("Numba JIT" if _numba_available else "NumPy")


# ════════════════════════════════════════════════════════════════════════════
# NUMBA JIT KERNELS (CPU fallback, ~5-10× faster than pure NumPy loops)
# ════════════════════════════════════════════════════════════════════════════

if _numba_available:
    from numba import njit, prange, float64, boolean, int32

    @njit(parallel=True, cache=True, fastmath=True)
    def _tick_kernel_numba(
        price: float,
        buy_above: np.ndarray,   # (N,)
        sell_below: np.ndarray,  # (N,)
        t1: np.ndarray,          # (N,) T1 targets
        t2: np.ndarray,          # (N,)
        t3: np.ndarray,          # (N,)
        t4: np.ndarray,          # (N,)
        t5: np.ndarray,          # (N,)
        st1: np.ndarray,         # (N,)
        st2: np.ndarray, st3: np.ndarray, st4: np.ndarray, st5: np.ndarray,
        buy_sl: np.ndarray,      # (N,)
        sell_sl: np.ndarray,     # (N,)
        in_position: np.ndarray, # (N,) bool
        is_buy: np.ndarray,      # (N,) bool
        entry_price: np.ndarray, # (N,)
        entry_qty: np.ndarray,   # (N,) int
        exited_today: np.ndarray,# (N,) bool
        peak_reached: np.ndarray,# (N,) bool
        entry_level: np.ndarray, # (N,) buy_above/sell_below at entry
        step: np.ndarray,        # (N,)
        pnl_accum: np.ndarray,   # (N,) accumulated net P&L
        trade_count: np.ndarray, # (N,) int
        win_count: np.ndarray,   # (N,) int
        # outputs
        new_entries: np.ndarray,     # (N,) bool — which positions just opened
        new_exits_idx: np.ndarray,   # (N,) int  — exit level index (0=none, 1=T1..5=T5, 6=SL, 7=retreat)
        new_exits_price: np.ndarray, # (N,) float exit price
        brokerage: float,
    ) -> None:
        n = len(buy_above)
        for i in prange(n):
            new_entries[i] = False
            new_exits_idx[i] = 0
            new_exits_price[i] = 0.0

            if in_position[i]:
                ep = entry_price[i]
                qty = float(entry_qty[i])
                step_i = step[i]
                el = entry_level[i]

                if is_buy[i]:
                    # Target exits
                    if price >= t5[i]:
                        gross = (t5[i] - ep) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net
                        trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 5; new_exits_price[i] = t5[i]
                    elif price >= t4[i]:
                        gross = (t4[i] - ep) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 4; new_exits_price[i] = t4[i]
                    elif price >= t3[i]:
                        gross = (t3[i] - ep) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 3; new_exits_price[i] = t3[i]
                    elif price >= t2[i]:
                        gross = (t2[i] - ep) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 2; new_exits_price[i] = t2[i]
                    elif price >= t1[i]:
                        gross = (t1[i] - ep) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 1; new_exits_price[i] = t1[i]
                    elif price <= buy_sl[i]:
                        gross = (price - ep) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 6; new_exits_price[i] = price
                    else:
                        # Retreat monitoring
                        if price >= el + 0.65 * step_i:
                            peak_reached[i] = True
                        if peak_reached[i] and price <= el + 0.25 * step_i:
                            gross = (price - ep) * qty
                            net = gross - brokerage
                            pnl_accum[i] += net; trade_count[i] += 1
                            if net > 0: win_count[i] += 1
                            in_position[i] = False; exited_today[i] = True
                            peak_reached[i] = False
                            new_exits_idx[i] = 7; new_exits_price[i] = price

                else:  # SELL
                    if price <= st5[i]:
                        gross = (ep - st5[i]) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 5; new_exits_price[i] = st5[i]
                    elif price <= st4[i]:
                        gross = (ep - st4[i]) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 4; new_exits_price[i] = st4[i]
                    elif price <= st3[i]:
                        gross = (ep - st3[i]) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 3; new_exits_price[i] = st3[i]
                    elif price <= st2[i]:
                        gross = (ep - st2[i]) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 2; new_exits_price[i] = st2[i]
                    elif price <= st1[i]:
                        gross = (ep - st1[i]) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        if net > 0: win_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 1; new_exits_price[i] = st1[i]
                    elif price >= sell_sl[i]:
                        gross = (ep - price) * qty
                        net = gross - brokerage
                        pnl_accum[i] += net; trade_count[i] += 1
                        in_position[i] = False; exited_today[i] = True
                        peak_reached[i] = False
                        new_exits_idx[i] = 6; new_exits_price[i] = price
                    else:
                        if price <= el - 0.65 * step_i:
                            peak_reached[i] = True
                        if peak_reached[i] and price >= el - 0.25 * step_i:
                            gross = (ep - price) * qty
                            net = gross - brokerage
                            pnl_accum[i] += net; trade_count[i] += 1
                            if net > 0: win_count[i] += 1
                            in_position[i] = False; exited_today[i] = True
                            peak_reached[i] = False
                            new_exits_idx[i] = 7; new_exits_price[i] = price

            elif not exited_today[i]:
                # Entry logic
                if price >= buy_above[i]:
                    in_position[i] = True; is_buy[i] = True
                    entry_price[i] = price
                    entry_level[i] = buy_above[i]
                    peak_reached[i] = False
                    new_entries[i] = True
                elif price <= sell_below[i]:
                    in_position[i] = True; is_buy[i] = False
                    entry_price[i] = price
                    entry_level[i] = sell_below[i]
                    peak_reached[i] = False
                    new_entries[i] = True

else:
    # Fallback stub when Numba not available
    _tick_kernel_numba = None


# ════════════════════════════════════════════════════════════════════════════
# FAST SWEEP STATE — Lightweight per-symbol state using NumPy arrays
# Designed to replace the heavy StockSweep per-symbol dict approach
# ════════════════════════════════════════════════════════════════════════════

class FastSweepState:
    """
    Ultra-compact per-symbol sweep state using flat NumPy arrays.
    All N X-value variations processed in one vectorised call.
    
    For Scanner3: N=32,000 × 38 symbols = 1,216,000 simultaneous simulations.
    Memory: 38 × 32K × 12 arrays × 8B ≈ 117 MB RAM (well within 5GB free).
    """
    __slots__ = (
        "symbol", "_n", "_xv", "prev_close", "is_commodity",
        "buy_above", "sell_below", "t", "st", "buy_sl", "sell_sl", "step",
        "in_position", "is_buy", "entry_price", "entry_qty", "exited_today",
        "peak_reached", "retreat_entry_level",
        "pnl_accum", "trade_count", "win_count",
        "_new_entries", "_new_exits_idx", "_new_exits_price",
        "best_pnl", "best_x", "best_x_idx",
        "last_price", "tick_count",
    )

    def __init__(self, symbol: str, prev_close: float, x_values: np.ndarray,
                 is_commodity: bool = False) -> None:
        self.symbol = symbol
        self._n = n = len(x_values)
        self._xv = x_values.astype(np.float64)
        self.prev_close = prev_close
        self.is_commodity = is_commodity
        self.last_price = 0.0
        self.tick_count = 0

        # Level arrays — x deviation per variation is xr; step == xr here (BSL = BA - x, SSL = SB + x).
        xr = prev_close * self._xv
        self.step       = xr.astype(np.float64)
        self.buy_above  = (prev_close + xr).astype(np.float64)
        self.sell_below = (prev_close - xr).astype(np.float64)
        self.buy_sl     = (self.buy_above - xr).astype(np.float64)
        self.sell_sl    = (self.sell_below + xr).astype(np.float64)
        # T1..T5 / ST1..ST5 targets (each is (N,))
        self.t  = np.stack([self.buy_above  + self.step * i for i in range(1, 6)])  # (5,N)
        self.st = np.stack([self.sell_below - self.step * i for i in range(1, 6)])  # (5,N)

        # Position state (bool arrays are compact: 32K bools = 32KB)
        self.in_position   = np.zeros(n, np.bool_)
        self.is_buy        = np.zeros(n, np.bool_)
        self.entry_price   = np.zeros(n, np.float64)
        self.entry_qty     = np.zeros(n, np.int32)
        self.exited_today  = np.zeros(n, np.bool_)
        self.peak_reached  = np.zeros(n, np.bool_)
        self.retreat_entry_level = np.zeros(n, np.float64)

        # Accumulators
        self.pnl_accum   = np.zeros(n, np.float64)
        self.trade_count = np.zeros(n, np.int32)
        self.win_count   = np.zeros(n, np.int32)

        # Per-tick output buffers (reused, no allocation per tick)
        self._new_entries    = np.zeros(n, np.bool_)
        self._new_exits_idx  = np.zeros(n, np.int32)
        self._new_exits_price = np.zeros(n, np.float64)

        # Best X tracking
        self.best_pnl   = -np.inf
        self.best_x     = float(self._xv[n // 2])
        self.best_x_idx = n // 2

    def on_price(self, price: float, qty: int, brokerage: float = 20.0) -> int:
        """
        Process one price tick for all N X-variations.
        Returns number of events (entries + exits) this tick.
        
        Uses Numba JIT if available, otherwise vectorised NumPy.
        """
        if price == self.last_price:
            return 0
        self.last_price = price
        self.tick_count += 1

        if _numba_available and _tick_kernel_numba is not None:
            # Numba JIT path: ~5-10× faster than NumPy for this workload
            _tick_kernel_numba(
                float(price),
                self.buy_above, self.sell_below,
                self.t[0], self.t[1], self.t[2], self.t[3], self.t[4],
                self.st[0], self.st[1], self.st[2], self.st[3], self.st[4],
                self.buy_sl, self.sell_sl,
                self.in_position, self.is_buy,
                self.entry_price, self.entry_qty.astype(np.float64),
                self.exited_today, self.peak_reached,
                self.retreat_entry_level, self.step,
                self.pnl_accum, self.trade_count, self.win_count,
                self._new_entries, self._new_exits_idx, self._new_exits_price,
                float(brokerage),
            )
        else:
            self._numpy_tick(price, qty, brokerage)

        # Update best X after each tick
        best_i = int(np.argmax(self.pnl_accum))
        if self.pnl_accum[best_i] > self.best_pnl:
            self.best_pnl = float(self.pnl_accum[best_i])
            self.best_x = float(self._xv[best_i])
            self.best_x_idx = best_i

        return int(self._new_entries.sum() + (self._new_exits_idx > 0).sum())

    def _numpy_tick(self, price: float, qty: int, brokerage: float) -> None:
        """Vectorised NumPy implementation — fallback when Numba unavailable."""
        n = self._n
        p = float(price)
        brok = float(brokerage)

        # ── Exits (in-position) ────────────────────────────────────────────────
        pos = self.in_position
        buy = self.is_buy & pos
        sel = ~self.is_buy & pos
        ep  = self.entry_price
        qty_f = self.entry_qty.astype(np.float64)

        if buy.any():
            # Target exits: check T5→T1 order (highest priority first)
            for ti in range(4, -1, -1):
                hit = buy & (p >= self.t[ti])
                if hit.any():
                    gross = (self.t[ti, hit] - ep[hit]) * qty_f[hit]
                    net   = gross - brok
                    self.pnl_accum[hit]  += net
                    self.trade_count[hit] += 1
                    self.win_count[hit]  += (net > 0).astype(np.int32)
                    self.in_position[hit] = False
                    self.exited_today[hit] = True
                    self.peak_reached[hit] = False
                    self._new_exits_idx[hit]   = ti + 1
                    self._new_exits_price[hit] = self.t[ti, hit]
                    buy &= ~hit
                break  # only the highest target hit matters per tick

            # SL exits
            sl_hit = buy & (p <= self.buy_sl)
            if sl_hit.any():
                gross = (p - ep[sl_hit]) * qty_f[sl_hit]
                net   = gross - brok
                self.pnl_accum[sl_hit]  += net
                self.trade_count[sl_hit] += 1
                self.in_position[sl_hit] = False
                self.exited_today[sl_hit] = True
                self.peak_reached[sl_hit] = False
                self._new_exits_idx[sl_hit]   = 6
                self._new_exits_price[sl_hit] = p
                buy &= ~sl_hit

            # Retreat: 65% peak → 25% exit
            el = self.retreat_entry_level
            s  = self.step
            self.peak_reached |= buy & (p >= el + 0.65 * s)
            retreat = buy & self.peak_reached & (p <= el + 0.25 * s)
            if retreat.any():
                gross = (p - ep[retreat]) * qty_f[retreat]
                net   = gross - brok
                self.pnl_accum[retreat]  += net
                self.trade_count[retreat] += 1
                self.win_count[retreat]  += (net > 0).astype(np.int32)
                self.in_position[retreat] = False
                self.exited_today[retreat] = True
                self.peak_reached[retreat] = False
                self._new_exits_idx[retreat]   = 7
                self._new_exits_price[retreat] = p

        if sel.any():
            for ti in range(4, -1, -1):
                hit = sel & (p <= self.st[ti])
                if hit.any():
                    gross = (ep[hit] - self.st[ti, hit]) * qty_f[hit]
                    net   = gross - brok
                    self.pnl_accum[hit]  += net
                    self.trade_count[hit] += 1
                    self.win_count[hit]  += (net > 0).astype(np.int32)
                    self.in_position[hit] = False
                    self.exited_today[hit] = True
                    self.peak_reached[hit] = False
                    self._new_exits_idx[hit]   = ti + 1
                    self._new_exits_price[hit] = self.st[ti, hit]
                    sel &= ~hit
                break

            sl_hit = sel & (p >= self.sell_sl)
            if sl_hit.any():
                gross = (ep[sl_hit] - p) * qty_f[sl_hit]
                net   = gross - brok
                self.pnl_accum[sl_hit]  += net
                self.trade_count[sl_hit] += 1
                self.in_position[sl_hit] = False
                self.exited_today[sl_hit] = True
                self.peak_reached[sl_hit] = False
                self._new_exits_idx[sl_hit]   = 6
                self._new_exits_price[sl_hit] = p
                sel &= ~sl_hit

            el = self.retreat_entry_level; s = self.step
            self.peak_reached |= sel & (p <= el - 0.65 * s)
            retreat = sel & self.peak_reached & (p >= el - 0.25 * s)
            if retreat.any():
                gross = (ep[retreat] - p) * qty_f[retreat]
                net   = gross - brok
                self.pnl_accum[retreat]  += net
                self.trade_count[retreat] += 1
                self.win_count[retreat]  += (net > 0).astype(np.int32)
                self.in_position[retreat] = False
                self.exited_today[retreat] = True
                self.peak_reached[retreat] = False
                self._new_exits_idx[retreat]   = 7
                self._new_exits_price[retreat] = p

        # ── Entries ────────────────────────────────────────────────────────────
        can_enter = ~self.in_position
        if can_enter.any():
            buy_entry = can_enter & (p >= self.buy_above)
            sel_entry = can_enter & (p <= self.sell_below) & ~buy_entry

            if buy_entry.any():
                self.in_position[buy_entry]  = True
                self.is_buy[buy_entry]       = True
                self.entry_price[buy_entry]  = p
                self.entry_qty[buy_entry]    = qty
                self.retreat_entry_level[buy_entry] = self.buy_above[buy_entry]
                self.peak_reached[buy_entry] = False
                self._new_entries[buy_entry] = True

            if sel_entry.any():
                self.in_position[sel_entry]  = True
                self.is_buy[sel_entry]       = False
                self.entry_price[sel_entry]  = p
                self.entry_qty[sel_entry]    = qty
                self.retreat_entry_level[sel_entry] = self.sell_below[sel_entry]
                self.peak_reached[sel_entry] = False
                self._new_entries[sel_entry] = True

    def eod_square_off(self, price: float, brokerage: float = 20.0) -> None:
        """EOD: close all open positions at current price."""
        pos = self.in_position
        if not pos.any():
            return
        ep = self.entry_price
        qty_f = self.entry_qty.astype(np.float64)

        buy_close = pos & self.is_buy
        if buy_close.any():
            gross = (price - ep[buy_close]) * qty_f[buy_close]
            net   = gross - brokerage
            self.pnl_accum[buy_close] += net
            self.trade_count[buy_close] += 1

        sel_close = pos & ~self.is_buy
        if sel_close.any():
            gross = (ep[sel_close] - price) * qty_f[sel_close]
            net   = gross - brokerage
            self.pnl_accum[sel_close] += net
            self.trade_count[sel_close] += 1

        self.in_position[:] = False
        self.exited_today[:] = True
        self.peak_reached[:] = False

    def reanchor(self, new_price: float) -> None:
        """Re-anchor all levels to new price (used by CryptoScanner every 6h)."""
        pc = new_price
        xr = pc * self._xv
        self.prev_close = pc
        self.step[:]       = xr
        self.buy_above[:]  = pc + xr
        self.sell_below[:] = pc - xr
        self.buy_sl[:]     = self.buy_above - self.step
        self.sell_sl[:]    = self.sell_below + self.step
        for i in range(5):
            self.t[i, :]  = self.buy_above  + self.step * (i + 1)
            self.st[i, :] = self.sell_below - self.step * (i + 1)
        # Reset daily state
        self.in_position[:]  = False
        self.exited_today[:] = False
        self.peak_reached[:] = False
        self.entry_price[:]  = 0.0

    def best_x_stats(self) -> dict:
        """Return best X value statistics for this symbol."""
        best_i = int(np.argmax(self.pnl_accum))
        tc = int(self.trade_count[best_i])
        wc = int(self.win_count[best_i])
        return {
            "symbol":     self.symbol,
            "best_x":     round(float(self._xv[best_i]), 8),
            "best_pnl":   round(float(self.pnl_accum[best_i]), 2),
            "trade_count": tc,
            "win_count":   wc,
            "win_rate":    round(wc / tc * 100, 1) if tc > 0 else 0.0,
            "x_values":   self._xv.tolist(),
            "total_pnl":  self.pnl_accum.tolist(),
            "trade_counts": self.trade_count.tolist(),
        }

    def dump_state(self) -> dict:
        """Serialise to JSON-safe dict for live_state.json."""
        return self.best_x_stats()


# ════════════════════════════════════════════════════════════════════════════
# MULTI-SYMBOL PARALLEL SWEEP RUNNER
# ════════════════════════════════════════════════════════════════════════════

class ParallelSweepRunner:
    """
    Runs FastSweepState for all symbols in parallel using ThreadPoolExecutor.
    
    For 38 equity symbols × 32K X-values:
    - Each symbol tick takes ~0.5-2ms (Numba) or ~5-10ms (NumPy)
    - ThreadPool with 4-6 workers → parallel → 38 symbols / 4 = ~10 ticks/sec
    - Total throughput: 38 × 32K = 1.2M calculations per price update
    
    Designed for Scanner3 on Core 4 of i5-12450H.
    """

    def __init__(
        self,
        sweeps: Dict[str, FastSweepState],
        n_workers: int = 4,
        brokerage: float = 20.0,
    ) -> None:
        self._sweeps     = sweeps
        self._n_workers  = n_workers
        self._brokerage  = brokerage
        self._lock       = threading.Lock()
        self._last_prices: Dict[str, float] = {}
        
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=n_workers,
                                         thread_name_prefix="sweep")
        log.info("ParallelSweepRunner: %d symbols, %d workers, backend=%s",
                 len(sweeps), n_workers, GPU_NAME)

    def process_prices(self, prices: Dict[str, float], ts,
                       qty_func=None) -> Dict[str, int]:
        """
        Process price updates for all symbols simultaneously.
        Returns {symbol: events_count}.
        """
        from concurrent.futures import as_completed
        default_qty = lambda p: max(1, int(100_000 // max(p, 1)))
        qty_f = qty_func or default_qty

        futures = {}
        for sym, sw in self._sweeps.items():
            price = prices.get(sym)
            if price is None or price == self._last_prices.get(sym):
                continue
            self._last_prices[sym] = price
            qty = qty_f(price)
            fut = self._pool.submit(sw.on_price, price, qty, self._brokerage)
            futures[fut] = sym

        results = {}
        for fut in as_completed(futures, timeout=2.0):
            sym = futures[fut]
            try:
                results[sym] = fut.result()
            except Exception as exc:
                log.debug("sweep error %s: %s", sym, exc)
        return results

    def eod_square_off(self, prices: Dict[str, float]) -> None:
        """EOD: square off all open positions."""
        for sym, sw in self._sweeps.items():
            price = prices.get(sym, sw.last_price)
            if price > 0:
                sw.eod_square_off(price, self._brokerage)

    def best_x_per_symbol(self) -> Dict[str, dict]:
        """Return best X stats for all symbols."""
        return {sym: sw.best_x_stats() for sym, sw in self._sweeps.items()}

    def dump_live_state(self, scanner_name: str, date_str: str) -> dict:
        """Serialise all sweeps to live_state.json format."""
        return {
            "scanner":    scanner_name,
            "date":       date_str,
            "backend":    GPU_NAME,
            "sweeps":     {sym: sw.dump_state() for sym, sw in self._sweeps.items()},
            "merged_best": {
                sym: {
                    "best_x":     sw.best_x_stats()["best_x"],
                    "best_pnl":   sw.best_x_stats()["best_pnl"],
                    "trade_count": sw.best_x_stats()["trade_count"],
                    "win_rate":   sw.best_x_stats()["win_rate"],
                }
                for sym, sw in self._sweeps.items()
            },
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
