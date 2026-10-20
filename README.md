# ⚡ TRE PREDATOR v1.0 • Aggressive Predictive Microstructure & Multi-Agent LLM Quant OS

**TRE PREDATOR v1.0** (`openclaw-predator`) is an experimental, ultra-aggressive quantitative trading operating system built for **OKX Spot & Demo Trading (`x-simulated-trading: 1`)**.

Unlike passive grid systems that wait for wide price excursions, **TRE PREDATOR** combines **real-time Limit Order Book (LOB) microstructure physics**, **Online Streaming Machine Learning (Recursive Least Squares + Kalman Velocity Filter)**, and a **3-Agent Adversarial LLM Debate Council** (`TradingAgents` / `Trading-R1` architecture) to actively predict short-horizon returns (in basis points) and execute with hybrid **Post-Only Maker + IOC Breakout Sniper** orders.

---

## 🔬 SCIENTIFIC & ARCHITECTURAL FOUNDATIONS

This system synthesizes four peer-reviewed quantitative and AI paradigms into a zero-bloat, pure-Python engine (`< 25 MB RAM`, zero heavy external ML dependencies):

1. **Multi-Level Order Flow Imbalance (OFI) — *Cont, Kukanov & Stoikov (2014) / Cartea & Jaimungal (2018)***
   - Tracks net supply/demand queue changes across the top 5 levels of the OKX Limit Order Book with exponential depth decay ($w_k = e^{-0.45 k}$).
   - Computes **Stoikov's Micro-Price** ($P_{\text{micro}}$) weighted by inverse top-of-book queue sizes to anticipate mid-price ticks before they print.

2. **Volume-Synchronized Probability of Informed Trading (VPIN) — *Easley, López de Prado & O'Hara (2012)***
   - Groups the live trade tape into volume-synchronized buckets ($V$) and measures order flow toxicity:
     $$\text{VPIN} = \frac{1}{N} \sum_{\tau=1}^{N} \frac{|V_\tau^B - V_\tau^S|}{V}$$
   - **Asymmetric Toxicity Shield:** When $\text{VPIN} > 0.68$ during selling pressure, the engine automatically widens buy traps (`TOXIC_DIP_DEFENSE`) to avoid catching falling knives and instead buys the panic exhaustion wick.

3. **Hawkes Self-Exciting Burst Process & Online RLS Predictor — *Bacry et al. (2015)***
   - Models trade arrival clustering via recursive exponential intensity:
     $$\lambda(t) = \mu + \sum_{t_i < t} \alpha \, e^{-\beta (t - t_i)}$$
   - **Online Recursive Least Squares (RLS) with Forgetting Factor ($\lambda = 0.985$):** Continuously learns and updates its weight vector $\mathbf{w}_t$ and covariance matrix $\mathbf{P}_t$ on every single tick, predicting the next short-horizon return $\hat{r}_{t+1}$ in basis points (`bps`).
   - **Kalman Price-Velocity Filter:** Extracts instantaneous price velocity ($\text{bps/s}$) stripped of bid-ask bounce noise.

4. **3-Agent Adversarial LLM Council — *Inspired by TauricResearch/TradingAgents & Trading-R1 (2025/2026)***
   - **Momentum Bull Agent (*Alpha Predator*):** Hunts for OFI breakouts, Hawkes trade bursts, and positive RLS velocity.
   - **Toxicity Bear Agent (*VPIN Sentinel*):** Audits VPIN order toxicity, spoofing risk, and inventory concentration.
   - **Chief Quant Judge (*CIO Synthesizer*):** Adjudicates the debate via **OpenRouter (`openrouter/free`)** or **Local/Tailnet Ollama GPU (`Nemotron 3.5 Lightning`)** and outputs strict JSON conviction multipliers (`0.5x` to `2.0x`).

---

## ⚔️ WHY IT IS RADICALLY MORE AGGRESSIVE THAN TRE OS v5.6

| Feature | TRE OS v5.6 (Conservative Production) | ⚡ TRE PREDATOR v1.0 (Experimental Aggressive) |
| :--- | :--- | :--- |
| **Signal Generation** | Reactive 24h High/Low Range | **Predictive OFI + VPIN + Hawkes Burst + Online RLS (`bps`)** |
| **AI Architecture** | Single hourly prompt | **3-Agent Adversarial Debate (Bull vs Bear vs CIO Judge)** |
| **Order Execution** | 100% Passive Maker-Only | **Hybrid: Post-Only Maker + `SNIPER_IOC` on High-Conviction Breakouts** |
| **Stale Order Handling** | Waits passively at old TP/Buy levels | **Dynamic Quote Chaser (Auto-cancels & re-pegs drifted orders)** |
| **Capital Deployment** | 85% Max Allocation, static splits | **Up to 92% Active Velocity Allocation weighted by RLS Confidence** |
| **Exchange Mode** | Production Spot (`brody`) | **Native OKX Demo (`x-simulated-trading: 1`) + Live Switch** |

---

## 📁 REPOSITORY STRUCTURE

```text
openclaw-predator/
├── core/
│   ├── microstructure.py       # Cont-Stoikov OFI, VPIN Toxicity, Hawkes Intensity, Stoikov Micro-Price
│   └── predictor.py            # Online RLS Streaming Learner (λ=0.985) + 2-State Kalman Velocity Filter
├── agents/
│   └── llm_council.py          # 3-Agent Bull/Bear/Judge Debate (OpenRouter Free + Ollama + Deterministic Fallback)
├── execution/
│   └── okx_hybrid_executor.py  # Native HMAC-SHA256 OKX v5 Client with `x-simulated-trading: 1` Demo Header
├── predator_daemon.py          # Main autonomous loop, SQLite WAL telemetry & Stale Order Chaser
├── install.sh                  # 1-Click VPS installer & bytecode verifier
└── .env.example                # Clean credential template (zero secrets committed)
```

---

## 🚀 QUICKSTART & DEMO TESTING

### 1. Clone & Install (1-Click)
```bash
curl -fsSL https://raw.githubusercontent.com/glaucogaribaldi/openclaw-predator/main/install.sh | bash
```

### 2. Configure OKX Demo Credentials
Edit `.env` and insert your OKX Demo Trading API keys (created in OKX under *Demo Trading -> Personal Center -> Demo Trading API*):
```env
OKX_API_KEY="your_demo_api_key"
OKX_SECRET_KEY="your_demo_secret_key"
OKX_PASSPHRASE="your_demo_passphrase"
OKX_SIMULATED_DEMO="1"
OPENROUTER_API_KEY="your_openrouter_key"
```

### 3. Run Live Predictive Scan & Multi-Agent Debate
Even before inserting Demo keys, you can run `predator_daemon.py` in **Public Feed Dry-Run Mode** against real-time OKX Level-2 order books:
```bash
python3 predator_daemon.py --once --debate
```
To start the continuous high-frequency daemon loop:
```bash
python3 predator_daemon.py --interval 10
```
