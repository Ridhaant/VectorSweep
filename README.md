# vectorsweep

**Author:** Ridhaant Ajoy Thackur  
**License:** MIT  
**Python:** 3.9+

> GPU-accelerated vectorised strategy parameter sweep engine.  
> Evaluates thousands of (entry_threshold, stop_loss, target) variants  
> against a full intraday price series — simultaneously — in under 1 ms.

---

## What it does

Given a price series and a range of `X` deviation values, `vectorsweep` finds the **optimal entry threshold** for a breakout strategy by sweeping the full parameter space in one vectorised pass.

For each `X` value:

```
buy_above  = prev_close + X        # long entry trigger
sell_below = prev_close − X        # short entry trigger
sl_buy     = buy_above  − X        # long stop-loss
sl_sell    = sell_below + X        # short stop-loss
t1_buy     = buy_above  + X        # first long target
t1_sell    = sell_below − X        # first short target
qty        = floor(₹100,000 / price)  # dynamic position sizing
```

---

## Architecture

```
SweepConfig (x_min, x_max, x_step, prev_close, budget)
        │
        ▼
  gen_x_values()          ← Decimal-exact step generation (no float64 drift)
        │
        ▼
 _build_level_arrays()    ← vectorised NumPy; builds all N level arrays at once
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │       Backend selection (auto-detect)        │
  │                                             │
  │  CuPy / CUDA 12  ──► _cupy_sweep()          │
  │  (NVIDIA GPU)        arrays on VRAM          │
  │                                             │
  │  Numba JIT (CPU) ──► _numba_tick_kernel()   │
  │  (LLVM-compiled)     parallel prange loop   │
  │                                             │
  │  NumPy (baseline) ─► _numpy_sweep()         │
  │  (always available)  masked array updates   │
  └─────────────────────────────────────────────┘
        │
        ▼
  List[SweepResult]  — sorted by net_pnl descending
```

---

## Performance

| Backend | Variants | Ticks | Evaluations | Time |
|---|---|---|---|---|
| CuPy (GTX 1650, 4GB) | 32,000 | 390 | 12,480,000 | < 1 ms |
| Numba JIT (CPU) | 32,000 | 390 | 12,480,000 | ~40 ms |
| NumPy (baseline) | 32,000 | 390 | 12,480,000 | ~250 ms |

Run the built-in benchmark:

```python
from src.vectorsweep import SweepEngine, SweepConfig
cfg = SweepConfig(x_min=0.001, x_max=0.5, x_step=0.0001, prev_close=2450.0)
engine = SweepEngine(cfg)
print(engine.benchmark(n_ticks=390))
```

---

## Installation

```bash
pip install numpy numba pytz
# Optional GPU acceleration (NVIDIA CUDA 12):
# pip install cupy-cuda12x
```

```bash
pip install -r requirements.txt
```

---

## Quickstart

```python
import numpy as np
from src.vectorsweep import SweepEngine, SweepConfig

# Simulate one trading session (390 1-minute bars)
rng = np.random.default_rng(0)
prices = 2450.0 + np.cumsum(rng.normal(0, 1.2, 390))

# Sweep X from ₹0.50 to ₹15.00 in ₹0.10 steps → 146 variants
cfg = SweepConfig(
    x_min=0.50,
    x_max=15.00,
    x_step=0.10,
    prev_close=2450.0,
    budget=100_000,
    brokerage_per_side=10.0,
)
engine = SweepEngine(cfg)
results = engine.run(prices)

best = results[0]
print(f"Best  X={best.x_val:.2f}  net_pnl=₹{best.net_pnl:,.0f}  trades={best.trade_count}")
print(f"Worst X={results[-1].x_val:.2f}  net_pnl=₹{results[-1].net_pnl:,.0f}")
print(f"Backend: {engine.backend}")
```

---

## SweepConfig reference

| Field | Default | Description |
|---|---|---|
| `x_min` | `0.005` | Minimum deviation value |
| `x_max` | `0.050` | Maximum deviation value |
| `x_step` | `0.0001` | Step size (exact Decimal arithmetic) |
| `budget` | `100_000` | Budget per position (₹) |
| `brokerage_per_side` | `10.0` | Flat brokerage per trade leg (₹) |
| `prev_close` | `100.0` | Previous-day close (level anchor) |

---

## SweepResult fields

```python
@dataclass
class SweepResult:
    x_val: float          # deviation parameter
    net_pnl: float        # net P&L (after brokerage)
    gross_pnl: float      # gross P&L
    trade_count: int      # total round-trips
    buy_above: float      # long entry level
    sell_below: float     # short entry level
    sl_buy: float         # long stop-loss
    sl_sell: float        # short stop-loss
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Project context

`vectorsweep` extracts and generalises the GPU sweep engine from [AlgoStack](https://github.com/ridhaant/algostack) — specifically `gpu_sweep.py` and `sweep_core.py`. In production AlgoStack runs 32,000 variants across 38 NSE symbols simultaneously, performing 2,352,000 evaluations per tick on a GTX 1650 in under 1 ms.

---

## License

MIT © 2025 Ridhaant Ajoy Thackur
