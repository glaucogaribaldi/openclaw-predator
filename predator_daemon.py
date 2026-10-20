#!/usr/bin/env python3
"""
TRE PREDATOR v1.0 — Main Autonomous Orchestrator (`predator_daemon.py`)
Combines:
1. Real-Time OKX v5 Market Microstructure Scanner (OFI, VPIN, Hawkes Process, Stoikov Micro-Price)
2. Online Streaming RLS + Kalman Velocity Predictor (updates weights live every tick)
3. Multi-Agent Adversarial LLM Debate Council (Bull vs Bear vs Judge on OpenRouter / Nemotron)
4. Hybrid Execution Engine (Post-Only Maker + IOC Breakout Sniper + Stale Order Re-Pegging)
5. SQLite WAL Telemetry & Audit Ledger (`data/predator.db`)
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from agents.llm_council import AdversarialLLMCouncil
from core.microstructure import MicrostructureEngine
from core.predictor import PredatorPredictiveEngine
from execution.okx_hybrid_executor import OKXNativeClient


DEFAULT_UNIVERSE: Dict[str, Dict[str, Any]] = {
    "SOL-USDC":    {"coin": "SOL",    "dec_px": 2, "dec_sz": 3},
    "SUI-USDC":    {"coin": "SUI",    "dec_px": 4, "dec_sz": 1},
    "NEAR-USDC":   {"coin": "NEAR",   "dec_px": 3, "dec_sz": 1},
    "RENDER-USDC": {"coin": "RENDER", "dec_px": 3, "dec_sz": 2},
    "FET-USDC":    {"coin": "FET",    "dec_px": 4, "dec_sz": 1},
    "AVAX-USDC":   {"coin": "AVAX",   "dec_px": 3, "dec_sz": 2},
    "SEI-USDC":    {"coin": "SEI",    "dec_px": 4, "dec_sz": 1},
    "LINK-USDC":   {"coin": "LINK",   "dec_px": 3, "dec_sz": 2},
}


def load_dotenv(env_path: str) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), env_vars[k.strip()])
    return env_vars


def init_predator_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS microstructure_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            inst_id TEXT,
            mid_price REAL,
            micro_price REAL,
            spread_bps REAL,
            obi_top5 REAL,
            ofi_norm REAL,
            vpin_toxicity REAL,
            hawkes_net_burst REAL,
            composite_alpha REAL,
            predicted_return_bps REAL,
            kalman_velocity_bps REAL,
            confidence REAL,
            regime TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predator_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            inst_id TEXT,
            side TEXT,
            execution_style TEXT,
            size REAL,
            price REAL,
            predicted_bps REAL,
            regime TEXT,
            ord_id TEXT,
            mode TEXT
        )
        """
    )
    conn.commit()
    return conn


def run_predator_cycle(
    client: OKXNativeClient,
    micro_engine: MicrostructureEngine,
    pred_engine: PredatorPredictiveEngine,
    council: AdversarialLLMCouncil,
    conn: sqlite3.Connection,
    run_llm_debate: bool = False,
) -> Dict[str, Any]:
    balances = client.get_account_balances() if client.is_authenticated else []
    open_orders = client.get_open_orders() if client.is_authenticated else []

    nav_usdc = sum(float(b.get("eqUsd") or 0.0) for b in balances) if balances else 1000.0
    free_usdc = (
        sum(float(b.get("availBal") or 0.0) for b in balances if b.get("ccy") in ("USDC", "USDT"))
        if balances
        else 850.0
    )

    # Load latest LLM council directive if present
    convictions: Dict[str, float] = {}
    agg_mult = 1.20
    if council.directive_out_path and os.path.exists(council.directive_out_path):
        try:
            with open(council.directive_out_path, "r", encoding="utf-8") as f:
                saved_dir = json.load(f)
                convictions = saved_dir.get("token_convictions", {})
                agg_mult = float(saved_dir.get("aggressiveness_multiplier", 1.20))
        except Exception:
            pass

    telemetry_rows: List[Dict[str, Any]] = []
    actions_taken: List[Dict[str, Any]] = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Scan all instruments in universe
    for inst_id, meta in DEFAULT_UNIVERSE.items():
        try:
            bids, asks = client.get_orderbook(inst_id, sz=5)
            trades = client.get_recent_trades(inst_id, limit=25)
        except Exception as e:
            print(f"[!] Market data fetch warning for {inst_id}: {e}")
            continue

        if not bids or not asks:
            continue

        sig = micro_engine.process_snapshot(inst_id, bids, asks, trades)

        coin = meta["coin"]
        token_eq = sum(float(b.get("eqUsd") or 0.0) for b in balances if b.get("ccy") == coin)
        inv_weight = (token_eq / nav_usdc) if nav_usdc > 0 else 0.0
        llm_bias = float(convictions.get(inst_id, 0.0))

        pred = pred_engine.evaluate(sig, inventory_weight_nav=inv_weight, llm_conviction_bias=llm_bias)

        row_dict = {
            "inst_id": inst_id,
            "mid_price": round(sig.mid_price, meta["dec_px"]),
            "micro_price": round(sig.micro_price, meta["dec_px"]),
            "spread_bps": sig.spread_bps,
            "obi_top5": sig.obi_top5,
            "ofi_norm": sig.ofi_norm,
            "vpin_toxicity": sig.vpin_toxicity,
            "hawkes_net_burst": sig.hawkes_net_burst,
            "composite_alpha": sig.composite_alpha,
            "predicted_return_bps": pred.predicted_return_bps,
            "kalman_velocity_bps": pred.kalman_velocity_bps,
            "confidence": pred.confidence,
            "regime": pred.regime,
            "buy_offset_bps": pred.recommended_buy_offset_bps,
            "sell_tp_bps": pred.recommended_sell_tp_bps,
            "allow_sniper_IOC": pred.allow_sniper_IOC,
        }
        telemetry_rows.append(row_dict)

        conn.execute(
            """
            INSERT INTO microstructure_telemetry (
                timestamp, inst_id, mid_price, micro_price, spread_bps,
                obi_top5, ofi_norm, vpin_toxicity, hawkes_net_burst, composite_alpha,
                predicted_return_bps, kalman_velocity_bps, confidence, regime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                inst_id,
                sig.mid_price,
                sig.micro_price,
                sig.spread_bps,
                sig.obi_top5,
                sig.ofi_norm,
                sig.vpin_toxicity,
                sig.hawkes_net_burst,
                sig.composite_alpha,
                pred.predicted_return_bps,
                pred.kalman_velocity_bps,
                pred.confidence,
                pred.regime,
            ),
        )

        # 2. Stale Order Chaser: Cancel resting orders that drifted > 45 bps away from optimal quote
        inst_orders = [o for o in open_orders if o.get("instId") == inst_id]
        for ord_obj in inst_orders:
            try:
                ord_px = float(ord_obj.get("px") or 0.0)
                ord_side = str(ord_obj.get("side") or "").lower()
                ord_id = str(ord_obj.get("ordId") or "")
                if ord_px > 0 and sig.mid_price > 0:
                    drift_bps = abs(ord_px - sig.mid_price) / sig.mid_price * 10000.0
                    # If a buy order is stranded > 95 bps away during an aggressive regime, re-peg it
                    if ord_side == "buy" and drift_bps > 95.0 and pred.regime != "TOXIC_DIP_DEFENSE":
                        client.cancel_order(inst_id, ord_id)
                        actions_taken.append({"action": "CANCEL_STALE_BUY", "inst_id": inst_id, "ord_id": ord_id})
            except Exception:
                pass

        # 3. Calculate Dynamic Aggressive Lot Size (92% Active Allocation, Conviction Weighted)
        base_lot_usd = (nav_usdc * 0.92) / len(DEFAULT_UNIVERSE)
        conv_boost = 1.0 + max(-0.35, min(0.65, llm_bias * 0.5))
        lot_usd = max(12.0, round(base_lot_usd * agg_mult * conv_boost, 2))

        dec_px = meta["dec_px"]
        dec_sz = meta["dec_sz"]

        # 4. Determine Execution Action (Sniper IOC vs Post-Only Maker Trap)
        if pred.allow_sniper_IOC and free_usdc >= lot_usd:
            exec_style = "SNIPER_IOC"
            target_buy_px = round(asks[0][0] * 1.0005, dec_px)
        else:
            exec_style = "PASSIVE_MAKER"
            target_buy_px = round(sig.mid_price * (1.0 - pred.recommended_buy_offset_bps / 10000.0), dec_px)

        buy_sz = round(lot_usd / target_buy_px, dec_sz) if target_buy_px > 0 else 0.0

        # Place or simulate order if no active buy order exists for this instrument
        has_buy_order = any(str(o.get("side")).lower() == "buy" for o in inst_orders)
        if not has_buy_order and buy_sz > 0:
            res_place = client.place_spot_order(
                inst_id=inst_id,
                side="buy",
                sz=buy_sz,
                px=target_buy_px,
                execution_style=exec_style,
            )
            mode_str = "DEMO_LIVE" if (client.is_authenticated and client.simulated_demo) else (
                "PROD_LIVE" if client.is_authenticated else "DRY_RUN_SIM"
            )
            ord_id_placed = ""
            if isinstance(res_place.get("data"), list) and res_place["data"]:
                ord_id_placed = str(res_place["data"][0].get("ordId", ""))

            conn.execute(
                """
                INSERT INTO predator_orders (
                    timestamp, inst_id, side, execution_style, size, price,
                    predicted_bps, regime, ord_id, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    inst_id,
                    "buy",
                    exec_style,
                    buy_sz,
                    target_buy_px,
                    pred.predicted_return_bps,
                    pred.regime,
                    ord_id_placed,
                    mode_str,
                ),
            )
            actions_taken.append(
                {
                    "inst_id": inst_id,
                    "side": "buy",
                    "style": exec_style,
                    "sz": buy_sz,
                    "px": target_buy_px,
                    "pred_bps": pred.predicted_return_bps,
                    "regime": pred.regime,
                    "mode": mode_str,
                }
            )

    conn.commit()
    pred_engine.save_state()

    debate_res = None
    if run_llm_debate and telemetry_rows:
        debate_res = council.run_council_debate(
            telemetry=telemetry_rows,
            nav_usdc=nav_usdc,
            free_usdc=free_usdc,
            trigger_reason="predator_cycle_synthesis",
        )

    return {
        "timestamp": now_iso,
        "nav_usdc": round(nav_usdc, 2),
        "free_usdc": round(free_usdc, 2),
        "authenticated": client.is_authenticated,
        "simulated_demo": client.simulated_demo,
        "telemetry": telemetry_rows,
        "actions": actions_taken,
        "llm_debate": asdict(debate_res) if debate_res else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRE PREDATOR v1.0 — Aggressive Predictive & LLM Trading OS")
    parser.add_argument("--once", action="store_true", help="Run a single predictive scan cycle and exit")
    parser.add_argument("--debate", action="store_true", help="Run the 3-Agent LLM Adversarial Council debate")
    parser.add_argument("--interval", type=float, default=10.0, help="Loop interval in seconds (default: 10s)")
    parser.add_argument("--live", action="store_true", help="Disable x-simulated-trading: 1 flag (use real account)")
    args = parser.parse_args()

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    simulated_demo = not args.live
    if os.environ.get("OKX_SIMULATED_DEMO", "1") == "0":
        simulated_demo = False

    client = OKXNativeClient(simulated_demo=simulated_demo)
    micro_engine = MicrostructureEngine(bucket_volume_usd=2500.0)
    pred_engine = PredatorPredictiveEngine(state_path=os.path.join(BASE_DIR, "data", "rls_model_state.json"))
    council = AdversarialLLMCouncil(
        directive_out_path=os.path.join(BASE_DIR, "config", "predator_llm_directive.json")
    )
    conn = init_predator_db(os.path.join(BASE_DIR, "data", "predator.db"))

    if args.once:
        summary = run_predator_cycle(
            client=client,
            micro_engine=micro_engine,
            pred_engine=pred_engine,
            council=council,
            conn=conn,
            run_llm_debate=args.debate,
        )
        print("=== TRE PREDATOR v1.0 TELEMETRY & PREDICTION REPORT ===")
        print(
            f"Timestamp: {summary['timestamp']} | Mode: "
            f"{'OKX_DEMO (x-simulated-trading:1)' if summary['simulated_demo'] else 'OKX_LIVE'} | "
            f"Auth: {'ACTIVE' if summary['authenticated'] else 'DRY_RUN_PUBLIC_FEED'}"
        )
        print("-" * 108)
        print(
            f"{'INST_ID':<12} {'MID_PX':<10} {'OFI':<8} {'VPIN':<7} {'HAWKES':<8} "
            f"{'ALPHA':<8} {'PRED_BPS':<10} {'CONF':<7} {'REGIME':<22} {'BUY_BPS':<8} {'TP_BPS'}"
        )
        print("-" * 108)
        for t in summary["telemetry"]:
            print(
                f"{t['inst_id']:<12} ${t['mid_price']:<9} {t['ofi_norm']:>+6.2f}  "
                f"{t['vpin_toxicity']:<6.2f} {t['hawkes_net_burst']:>+6.2f}  "
                f"{t['composite_alpha']:>+6.2f}  {t['predicted_return_bps']:>+7.2f}bps "
                f"{t['confidence']:<6.2f} {t['regime']:<22} -{t['buy_offset_bps']:<6.1f} +{t['sell_tp_bps']:.1f}"
            )
        if summary["llm_debate"]:
            deb = summary["llm_debate"]
            print("\n=== MULTI-AGENT ADVERSARIAL LLM COUNCIL VERDICT ===")
            print(f"Model Engine: {deb['model_used']} | Global Regime: {deb['global_regime']} ({deb['aggressiveness_multiplier']}x)")
            print(f"  [BULL AGENT]:  {deb['bull_thesis']}")
            print(f"  [BEAR AGENT]:  {deb['bear_thesis']}")
            print(f"  [CIO JUDGE]:   {deb['judge_verdict']}")
        conn.close()
        return

    print(f"[⚡ TRE PREDATOR v1.0] Starting continuous aggressive loop (interval={args.interval}s)...")
    cycle_num = 0
    try:
        while True:
            cycle_num += 1
            do_debate = (cycle_num % 18 == 1) or args.debate
            summary = run_predator_cycle(
                client=client,
                micro_engine=micro_engine,
                pred_engine=pred_engine,
                council=council,
                conn=conn,
                run_llm_debate=do_debate,
            )
            print(
                f"[{summary['timestamp']}] Cycle #{cycle_num} | Scanned {len(summary['telemetry'])} pairs | "
                f"Orders evaluated: {len(summary['actions'])}"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[!] Shutting down TRE PREDATOR v1.0 cleanly.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
