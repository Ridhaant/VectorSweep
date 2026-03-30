<div align="center">

```
██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗
██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝
╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
 ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
  ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
      SWEEP
```

# vectorsweep

**GPU-Accelerated Vectorised Strategy Parameter Sweep Engine**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-Optional_GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![NumPy](https://img.shields.io/badge/NumPy-Backend-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-blue?style=for-the-badge)](./tests/)

<br/>

> Evaluates **thousands of (entry, stop-loss, target) parameter variants**
> against a full intraday price series — simultaneously — in **< 1 ms on GPU**.
> Auto-detects CuPy CUDA / Numba JIT / NumPy. Same code. Optimal hardware.

<br/>

[⚡ Quickstart](#quickstart) · [🏗 How It Works](#how-it-works) · [🔢 Performance](#performance) · [📐 API Reference](#api-reference) · [🔗 Project Context](#project-context)

</div>

---

## The Problem with Parameter Sweeps

Most quant backtests loop over parameter variants one at a time:

```python
# ❌ Naive approach — O(N) Python loops = slow
for x in x_values:
    simulate_strategy(prices, x)    # 32,000 iterations × 390 ticks = 12.48M ops
                                    # Python loop: ~45 seconds
```

`vectorsweep` solves this by evaluating **all N variants simultaneously** as a single vectorised operation:

```python
# ✅ vectorsweep — one pass, all variants in parallel
results = engine.run(prices)        # 12.48M ops in < 1ms on GPU
                                    # ~40ms on CPU (Numba JIT)
```

---

## How It Works

For each `X` deviation value, the strategy is:

```
buy_above   = prev_close + X          ← long entry trigger
sell_below  = prev_close − X          ← short entry trigger
sl_buy      = buy_above  − X          ← long stop-loss
sl_sell     = sell_below + X          ← short stop-loss
t1_buy      = buy_above  + X          ← first long target
t1_sell     = sell_below − X          ← first short target
qty         = floor(budget / price)   ← dynamic position size
```

Instead of running these serially, vectorsweep builds **N-dimensional arrays** where each index represents one X variant, then evaluates all variants simultaneously on every price tick using boolean mask operations — no Python loops in the hot path.

### Sweep Architecture

```
SweepConfig (x_min, x_max, x_step, prev_close, budget)
        │
        ▼
  gen_x_values()           ← Decimal-exact step generation (no float64 drift)
        │
        ▼
 _build_level_arrays()     ← NumPy broadcasting: builds buy_above[N], sl[N], t1[N]
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │       Backend auto-detection (import time)  │
  │                                             │
  │  ① CuPy/CUDA 12  → _cupy_sweep()            │
  │     arrays on GPU VRAM, cupy.where()        │
  │     < 1ms for 32K variants × 390 ticks      │
  │                                             │
  │  ② Numba JIT     → _numba_tick_kernel()     │
  │     @njit(parallel=True, fastmath=True)     │
  │     prange over N variants per tick         │
  │     ~40ms for same workload                 │
  │                                             │
  │  ③ NumPy         → _numpy_sweep()           │
  │     masked array updates                    │
  │     ~250ms baseline (always available)      │
  └─────────────────────────────────────────────┘
        │
        ▼
  List[SweepResult]   — sorted by net_pnl descending
```

**Why Decimal step generation?**
Float64 accumulation drift: `0.008 + 1000 × 0.000001 ≠ 0.009` in float64. A drifted endpoint silently skips the best X value. `gen_x_values()` uses Python `Decimal` internally before converting to `float64` — O(N) overhead of < 1ms, zero drift.

---

## Performance

| Backend | Hardware | Variants | Ticks | Evaluations | Time |
|---------|---------|----------|-------|-------------|------|
| **CuPy CUDA** | GTX 1650 (4GB) | 32,000 | 390 | 12,480,000 | **< 1 ms** |
| **Numba JIT** | i5-12450H (CPU) | 32,000 | 390 | 12,480,000 | **~40 ms** |
| **NumPy** | Any CPU | 32,000 | 390 | 12,480,000 | **~250 ms** |

**VRAM footprint:** 32K variants × float32 arrays ≈ **11 MB** of 4,096 MB (< 0.3%)

Run the built-in benchmark:

```python
from src.vectorsweep import SweepEngine, SweepConfig
cfg = SweepConfig(x_min=0.001, x_max=0.5, x_step=0.0001, prev_close=2450.0)
engine = SweepEngine(cfg)
print(engine.benchmark(n_ticks=390))
# Backend: cupy | 12480000 evals | 0.8ms | 15600000 evals/ms
```

---

## Quickstart

### Install

```bash
pip install numpy numba pytz
# Optional GPU acceleration (NVIDIA CUDA 12):
pip install cupy-cuda12x
```

```bash
pip install -r requirements.txt
```

### Basic sweep

```python
import numpy as np
from src.vectorsweep import SweepEngine, SweepConfig

# One intraday session — 390 one-minute bars
rng = np.random.default_rng(42)
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

print(f"Best:  X={results[0].x_val:.2f}  P&L=₹{results[0].net_pnl:,.0f}  trades={results[0].trade_count}")
print(f"Worst: X={results[-1].x_val:.2f}  P&L=₹{results[-1].net_pnl:,.0f}")
print(f"Backend: {engine.backend}")   # "cupy" | "numba" | "numpy"
```

### Large-scale sweep (GPU)

```python
# 32,000 variants — real AlgoStack production config
cfg = SweepConfig(
    x_min=0.001,
    x_max=0.032,
    x_step=0.000001,       # 1001-step range → 32,001 variants
    prev_close=2450.0,
    budget=100_000,
)
engine = SweepEngine(cfg)
results = engine.run(prices)   # < 1ms on GPU
```

---

## API Reference

### `SweepConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `x_min` | `0.005` | Minimum X deviation value |
| `x_max` | `0.050` | Maximum X deviation value |
| `x_step` | `0.0001` | Step size (Decimal-exact, no float64 drift) |
| `prev_close` | `100.0` | Previous-day close (level anchor) |
| `budget` | `100_000` | Capital per position (₹) |
| `brokerage_per_side` | `10.0` | Flat brokerage per trade leg (₹) |

### `SweepEngine`

| Method | Description |
|--------|-------------|
| `.run(prices: np.ndarray)` | Run sweep, return `List[SweepResult]` sorted by net P&L |
| `.benchmark(n_ticks: int)` | Performance report for active backend |
| `.backend` | `"cupy"` \| `"numba"` \| `"numpy"` |

### `SweepResult`

```python
@dataclass
class SweepResult:
    x_val:       float   # deviation parameter
    net_pnl:     float   # net P&L after brokerage
    gross_pnl:   float   # gross P&L
    trade_count: int     # total round-trips
    buy_above:   float   # long entry level
    sell_below:  float   # short entry level
    sl_buy:      float   # long stop-loss
    sl_sell:     float   # short stop-loss
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Context

`vectorsweep` is extracted and generalised from the GPU sweep engine in [`AlgoStack`](https://github.com/Ridhaant/algostack) — specifically `gpu_sweep.py` (GPU backend) and `sweep_core.py` (1,830 lines, multi-backend sweep kernel).

In production AlgoStack runs **32,000 variants × 38 NSE symbols = 1,216,000 parallel evaluations per price tick**, completing in < 1ms on a GTX 1650 — enabling real-time optimal strategy configuration selection while markets are live.

**Part of the AlgoStack open-source layer:**
- **[nexus-price-bus](https://github.com/Ridhaant/nexus-price-bus)** — multi-source ZMQ market data bus
- **vectorsweep** — GPU strategy parameter sweep (this library)
- **[sentitrade](https://github.com/Ridhaant/sentitrade)** — Indian market NLP sentiment pipeline

---

## Author

**[Ridhaant Ajoy Thackur](https://github.com/Ridhaant)**
*Quant Developer · GPU ML Engineer · LNMIIT Jaipur*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/ridhaant-thackur-09947a1b0)
[![GitHub](https://img.shields.io/badge/GitHub-@Ridhaant-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Ridhaant)
[![Email](https://img.shields.io/badge/Email-redantthakur%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:redantthakur@gmail.com)

---

## License

MIT © 2026 Ridhaant Ajoy Thackur
