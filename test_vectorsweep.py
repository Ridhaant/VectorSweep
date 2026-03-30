"""
tests/test_vectorsweep.py
Author : Ridhaant Ajoy Thackur
"""

import numpy as np
import pytest

from src.vectorsweep import (
    SweepConfig,
    SweepEngine,
    SweepResult,
    gen_x_values,
    _build_level_arrays,
    _numpy_sweep,
)


class TestGenXValues:
    def test_count(self):
        x = gen_x_values(0.008, 0.009, 1e-6)
        assert x.shape[0] == 1001

    def test_endpoints(self):
        x = gen_x_values(1.0, 2.0, 0.5)
        assert abs(x[0] - 1.0) < 1e-9
        assert abs(x[-1] - 2.0) < 1e-9

    def test_monotonic(self):
        x = gen_x_values(0.01, 0.10, 0.01)
        diffs = np.diff(x)
        assert np.all(diffs > 0)


class TestBuildLevelArrays:
    def test_shapes(self):
        x = gen_x_values(1.0, 5.0, 0.5)
        arrs = _build_level_arrays(x, prev_close=100.0)
        for arr in arrs:
            assert arr.shape == x.shape

    def test_buy_above_gt_sell_below(self):
        x = gen_x_values(1.0, 5.0, 0.5)
        ba, sb, *_ = _build_level_arrays(x, prev_close=100.0)
        assert np.all(ba > sb)

    def test_sl_below_entry(self):
        x = gen_x_values(1.0, 5.0, 0.5)
        ba, sb, sl_buy, sl_sell, *_ = _build_level_arrays(x, prev_close=100.0)
        assert np.all(sl_buy < ba)
        assert np.all(sl_sell > sb)


class TestNumpySweep:
    def _flat_levels(self, n=10):
        """All variants with identical levels for a known-outcome test."""
        buy_above  = np.full(n, 101.0)
        sell_below = np.full(n, 99.0)
        sl_buy     = np.full(n, 100.0)
        sl_sell    = np.full(n, 100.0)
        t1_buy     = np.full(n, 102.0)
        t1_sell    = np.full(n, 98.0)
        qty        = np.full(n, 10.0)
        return buy_above, sell_below, sl_buy, sl_sell, t1_buy, t1_sell, qty

    def test_no_trade_flat_price(self):
        """Flat price = no entries at all."""
        prices = np.full(100, 100.0)
        ba, sb, slb, sls, t1b, t1s, qty = self._flat_levels()
        pnl, trades = _numpy_sweep(prices, ba, sb, slb, sls, t1b, t1s, qty, brokerage=10.0)
        assert np.all(trades == 0)
        assert np.all(pnl == 0.0)

    def test_buy_target_hit(self):
        """Price rises past buy_above then hits t1_buy."""
        prices = np.array([100.0, 101.5, 102.5])  # enters at 101.5, exits at t1=102
        ba  = np.array([101.0])
        sb  = np.array([99.0])
        slb = np.array([100.0])
        sls = np.array([100.0])
        t1b = np.array([102.0])
        t1s = np.array([98.0])
        qty = np.array([10.0])
        pnl, trades = _numpy_sweep(prices, ba, sb, slb, sls, t1b, t1s, qty, brokerage=10.0)
        assert trades[0] == 1
        # gross = (102.0 - 101.5) * 10 = 5.0; net = 5.0 - 20 = -15.0
        assert abs(pnl[0] - (5.0 - 20.0)) < 1e-6


class TestSweepEngine:
    def test_returns_sorted_results(self):
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.3, 200))
        cfg = SweepConfig(x_min=0.5, x_max=3.0, x_step=0.5, prev_close=prices[0])
        engine = SweepEngine(cfg)
        results = engine.run(prices)
        pnls = [r.net_pnl for r in results]
        assert pnls == sorted(pnls, reverse=True)

    def test_result_count_matches_x_values(self):
        prices = np.linspace(100, 110, 100)
        cfg = SweepConfig(x_min=1.0, x_max=5.0, x_step=1.0, prev_close=100.0)
        engine = SweepEngine(cfg)
        results = engine.run(prices)
        assert len(results) == engine.n_variants

    def test_benchmark_returns_dict(self):
        cfg = SweepConfig(x_min=0.5, x_max=2.0, x_step=0.5, prev_close=100.0)
        engine = SweepEngine(cfg)
        bench = engine.benchmark(n_ticks=50)
        assert "backend" in bench
        assert "total_evaluations" in bench
        assert bench["total_evaluations"] > 0
