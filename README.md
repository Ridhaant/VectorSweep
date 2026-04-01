<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&duration=2500&pause=800&color=76B900&center=true&vCenter=true&width=800&lines=VectorSweep;GPU-Accelerated+Strategy+Parameter+Optimization;2%2C352%2C000+Evaluations%2FTick+%7C+%3C1ms+CUDA;CuPy+%E2%86%92+Numba+JIT+%E2%86%92+NumPy+%E2%80%94+3-Tier+Fallback" alt="VectorSweep" />

<br/>

![Evaluations](https://img.shields.io/badge/Evaluations%2FTick-2%2C352%2C000-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Latency](https://img.shields.io/badge/GPU%20Latency-%3C1ms-00FF41?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

<img src="https://skillicons.dev/icons?i=python,linux,git,github&theme=dark" />

<br/><br/>

![CuPy CUDA 12](https://img.shields.io/badge/CuPy_CUDA_12-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Numba JIT](https://img.shields.io/badge/Numba_JIT-00A3E0?style=flat-square&logo=llvm&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Decimal](https://img.shields.io/badge/Decimal_Precision-FF6B35?style=flat-square)

*Sole-authored by **[Ridhaant Ajoy Thackur](https://github.com/Ridhaant)** · Extracted from [AlgoStack](https://github.com/Ridhaant/AlgoStack)*

</div>

---

## ⚡ What Is VectorSweep?

A production-tested GPU-accelerated parameter sweep library that evaluates **thousands of strategy parameter hypotheses in a single vectorised pass** — eliminating per-variant Python loops entirely. Extracted from AlgoStack's live trading platform where it processes **2,352,000 X-multiplier evaluations per price tick in <1ms**.

---

## 📊 Performance

| Scanner | Symbols | X-values | Evaluations/Tick | Backend | Latency | Status |
|:---|:---|:---|---:|:---|:---|:---:|
| S1 Narrow | 38 equity | 1,000 | 38,000 | NumPy | ~5ms | 🟢 LIVE |
| S2 Dual | 38 equity | 16,000 | 608,000 | Numba JIT | ~40ms | 🟢 LIVE |
| S3 Wide | 38 equity | 32,000 | **1,216,000** | **CuPy CUDA** | **<1ms** | 🟢 LIVE |
| Commodity | 5 MCX | 49,000 | 245,000 | Auto-detect | varies | 🟢 LIVE |
| Crypto | 5 Binance | 49,000 | 245,000 | Auto-detect | varies | 🟢 LIVE |
| **Total** | **48** | **147,000** | **2,352,000** | | | ✅ PROD |

**Session throughput:** 32,000 variants × 390 ticks = **12.48M evaluations/session** in <1ms per tick on GTX 1650.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "📥 Input"
        TICK["Market Price Tick"]
        XVALS["X-Value Grid<br/>1,000 → 32,000 candidates"]
    end
    subgraph "⚙️ SweepEngine"
        LEVELS["Level Arrays via NumPy Broadcasting<br/>buy_above[N], sell_below[N], t1..t5[N]<br/>Built once — O(1) per tick"]
        EVAL["Boolean Mask Evaluation<br/>any_exit = bt1 | bsl | st1 | ssl<br/>All N variants simultaneously"]
        SCORE["Composite Score<br/>0.5×P&L + 0.3×win_rate + 0.2×(1−drawdown)"]
    end
    subgraph "🖥️ 3-Tier Compute Backend"
        GPU["CuPy CUDA 12<br/>15–30× NumPy · <1ms"]
        JIT["Numba JIT<br/>@njit parallel · 5–10× · ~40ms"]
        CPU["NumPy Baseline<br/>~250ms"]
    end
    TICK & XVALS --> LEVELS --> EVAL --> SCORE
    SCORE -->|"auto-detect"| GPU
    SCORE -->|"fallback"| JIT
    SCORE -->|"fallback"| CPU

    style GPU fill:#0d1117,stroke:#76B900,stroke-width:2px,color:#76B900
    style JIT fill:#0d1117,stroke:#00A3E0,stroke-width:2px,color:#00A3E0
    style CPU fill:#0d1117,stroke:#013243,stroke-width:2px,color:#e6edf3
```

---

## 🔗 Proven in Production

Extracted from [AlgoStack](https://github.com/Ridhaant/AlgoStack) v10.7's `sweep_core.py` — battle-tested across **16 concurrent processes** on a live multi-asset trading platform processing NSE, MCX, and Binance markets simultaneously.

---

## 📦 Related

[![AlgoStack](https://img.shields.io/badge/AlgoStack-Parent%20Platform-00D4FF?style=for-the-badge)](https://github.com/Ridhaant/AlgoStack)
[![nexus-price-bus](https://img.shields.io/badge/nexus--price--bus-Price%20Ticks-DF0000?style=for-the-badge)](https://github.com/Ridhaant/Nexus-Price-Bus)
[![sentitrade](https://img.shields.io/badge/sentitrade-NLP%20Signals-3fb950?style=for-the-badge)](https://github.com/Ridhaant/SentiTrade)

---

<div align="center">

© 2026 Ridhaant Ajoy Thackur · MIT License

</div>
