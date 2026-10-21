#!/usr/bin/env python3
"""
TRE PREDATOR v1.2 — Main Autonomous Orchestrator (`predator_daemon.py`)
Combines:
1. Real-Time OKX v5 Market Microstructure Scanner (OFI, VPIN, Hawkes Process, Stoikov Micro-Price)
2. Online Streaming RLS + Kalman Velocity Predictor (updates weights live every tick)
3. NeoQuasar/Kronos-small (24.7M) GPU Foundation Model Oracle (`100.125.189.103:8090`)
4. Multi-Agent Adversarial LLM Debate Council (Bull vs Bear vs Judge on OpenRouter / Nemotron)
5. Hybrid Execution Engine with Hardened Audit Safeguards:
   - Cost-Basis Floor Protection (`openAvgPx * 1.0022` minimum SELL price — zero loss sells)
   - Dynamic Profit-Lock Re-Pegging (tightens SELL TP to +12 bps when unrealized gain >= +180 bps)
   - 25% NAV Hard Inventory Cap per asset
   - Continuous FIFO Fills Ledger Sync & Autonomous Nemotron GPU Co-Auditor (`core/predator_auditor.py`)
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
from core.kronos_oracle import KronosOracleClient
from core.microstructure import MicrostructureEngine
from core.predator_auditor import PredatorAuditor
from core.predictor import PredatorPredictiveEngine
from execution.okx_hybrid_executor import OKXNativeClient


DEFAULT_UNIVERSE: Dict[str, Dict[str, Any]] = {
    "SOL-USDC":  {"coin": "SOL",  "dec_px": 1, "dec_sz": 3},
    "SUI-USDC":  {"coin": "SUI",  "dec_px": 4, "dec_sz": 2},
    "NEAR-USDC": {"coin": "NEAR", "dec_px": 3, "dec_sz": 2},
    "LINK-USDC": {"coin": "LINK", "dec_px": 3, "dec_sz": 2},
    "HYPE-USDC": {"coin": "HYPE", "dec_px": 2, "dec_sz": 2},
    "WLD-USDC":  {"coin": "WLD",  "dec_px": 4, "dec_sz": 1},
    "ONDO-USDC": {"coin": "ONDO", "dec_px": 4, "dec_sz": 1},
    "DOGE-USDC": {"coin": "DOGE", "dec_px": 4, "dec_sz": 1},
    "ARB-USDC":  {"coin": "ARB",  "dec_px": 3, "dec_sz": 1},
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
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
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
    kronos_client: KronosOracleClient,
    council: AdversarialLLMCouncil,
    auditor: PredatorAuditor,
    conn: sqlite3.Connection,
    run_llm_debate: bool = False,
    run_audit: bool = False,
) -> Dict[str, Any]:
    balances = client.get_account_balances() if client.is_authenticated else []
    open_orders = client.get_open_orders() if client.is_authenticated else []
    kronos_forecasts = kronos_client.fetch_forecasts(list(DEFAULT_UNIVERSE.keys()))

    tracked_ccys = {m["coin"] for m in DEFAULT_UNIVERSE.values()} | {"USDC"}
    nav_usdc = (
        sum(float(b.get("eqUsd") or 0.0) for b in balances if b.get("ccy") in tracked_ccys)
        if balances
        else 313.88
    )
    free_usdc = (
        sum(float(b.get("availBal") or 0.0) for b in balances if b.get("ccy") == "USDC")
        if balances
        else 313.88
    )

    # Build cost-basis (`openAvgPx`) and holdings summary from live OKX Demo account
    open_avg_px_map: Dict[str, float] = {}
    holdings_summary: List[Dict[str, Any]] = []
    for b in balances:
        ccy = str(b.get("ccy") or "")
        if ccy in tracked_ccys and ccy != "USDC":
            cash_bal = float(b.get("cashBal") or 0.0)
            eq_usd = float(b.get("eqUsd") or 0.0)
            open_avg = float(b.get("openAvgPx") or b.get("accAvgPx") or 0.0)
            upl = float(b.get("spotUpl") or b.get("totalPnl") or 0.0)
            if open_avg > 0:
                open_avg_px_map[ccy] = open_avg
            if eq_usd > 0.25:
                holdings_summary.append(
                    {
                        "ccy": ccy,
                        "qty": round(cash_bal, 4),
                        "eq_usd": round(eq_usd, 2),
                        "open_avg_px": round(open_avg, 6),
                        "upl_usd": round(upl, 2),
                    }
                )

    # Sync recent OKX Demo fills into SQLite FIFO ledger
    if client.is_authenticated:
        try:
            raw_fills = client.get_recent_fills(limit=50)
            auditor.sync_okx_fills(
                conn=conn,
                raw_fills=raw_fills,
                open_avg_px_map=open_avg_px_map,
                allowed_pairs=set(DEFAULT_UNIVERSE.keys()),
            )
        except Exception:
            pass

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
    cost_basis_fixes = 0
    profit_lock_repegs = 0
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    telemetry_tuples: List[tuple] = []
    order_tuples: List[tuple] = []

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
        dec_px = meta["dec_px"]
        dec_sz = meta["dec_sz"]

        token_bal_obj = next((b for b in balances if b.get("ccy") == coin), {})
        token_eq = float(token_bal_obj.get("eqUsd") or 0.0)
        cash_token_qty = float(token_bal_obj.get("cashBal") or 0.0)
        avail_token_qty = float(token_bal_obj.get("availBal") or 0.0)
        open_avg_px = open_avg_px_map.get(coin, 0.0)

        inv_weight = (token_eq / nav_usdc) if nav_usdc > 0 else 0.0
        llm_bias = float(convictions.get(inst_id, 0.0))

        k_fc = kronos_forecasts.get(inst_id)
        k_ret_bps = k_fc.expected_return_1h_bps if k_fc else 0.0
        k_up_bps = k_fc.upside_high_bps if k_fc else 0.0
        k_down_bps = k_fc.downside_low_bps if k_fc else 0.0

        pred = pred_engine.evaluate(
            sig,
            inventory_weight_nav=inv_weight,
            llm_conviction_bias=llm_bias,
            kronos_return_1h_bps=k_ret_bps,
            kronos_upside_bps=k_up_bps,
            kronos_downside_bps=k_down_bps,
        )

        # Calculate Cost-Basis Floor (+22 bps minimum net profit over openAvgPx)
        cost_floor_px = round(open_avg_px * 1.0022, dec_px) if open_avg_px > 0 else 0.0
        unrealized_bps = ((sig.mid_price - open_avg_px) / open_avg_px * 10000.0) if open_avg_px > 0 else 0.0

        row_dict = {
            "inst_id": inst_id,
            "mid_price": round(sig.mid_price, dec_px),
            "micro_price": round(sig.micro_price, dec_px),
            "open_avg_px": round(open_avg_px, dec_px) if open_avg_px > 0 else 0.0,
            "unrealized_bps": round(unrealized_bps, 1),
            "spread_bps": sig.spread_bps,
            "obi_top5": sig.obi_top5,
            "ofi_norm": sig.ofi_norm,
            "vpin_toxicity": sig.vpin_toxicity,
            "hawkes_net_burst": sig.hawkes_net_burst,
            "composite_alpha": sig.composite_alpha,
            "kronos_1h_bps": round(k_ret_bps, 2),
            "kronos_high_bps": round(k_up_bps, 2),
            "kronos_low_bps": round(k_down_bps, 2),
            "predicted_return_bps": pred.predicted_return_bps,
            "kalman_velocity_bps": pred.kalman_velocity_bps,
            "confidence": pred.confidence,
            "regime": pred.regime,
            "buy_offset_bps": pred.recommended_buy_offset_bps,
            "sell_tp_bps": pred.recommended_sell_tp_bps,
            "allow_sniper_IOC": pred.allow_sniper_IOC,
        }
        telemetry_rows.append(row_dict)

        telemetry_tuples.append(
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
            )
        )

        # 2. Audit & Self-Heal Active Orders (Stale BUY Chaser + Cost-Basis Floor + Profit-Lock Re-Peg)
        inst_orders = [o for o in open_orders if o.get("instId") == inst_id]
        has_buy_order = False
        has_sell_order = False

        for ord_obj in inst_orders:
            try:
                ord_px = float(ord_obj.get("px") or 0.0)
                ord_side = str(ord_obj.get("side") or "").lower()
                ord_id = str(ord_obj.get("ordId") or "")
                if ord_px <= 0 or sig.mid_price <= 0:
                    continue
                drift_bps = abs(ord_px - sig.mid_price) / sig.mid_price * 10000.0

                if ord_side == "buy":
                    has_buy_order = True
                    # Cancel stale BUY order if drifted > 95 bps away or if inventory exceeds 25% NAV cap
                    if (drift_bps > 95.0 and pred.regime != "TOXIC_DIP_DEFENSE") or inv_weight >= 0.25:
                        client.cancel_order(inst_id, ord_id)
                        has_buy_order = False
                        actions_taken.append({"action": "CANCEL_STALE_OR_CAPPED_BUY", "inst_id": inst_id, "ord_id": ord_id})

                elif ord_side == "sell":
                    has_sell_order = True
                    # AUDIT SAFEGUARD 1: Never allow a SELL order below openAvgPx * 1.0022 (Cost-Basis Floor)
                    if cost_floor_px > 0 and ord_px < cost_floor_px:
                        client.cancel_order(inst_id, ord_id)
                        has_sell_order = False
                        avail_token_qty = cash_token_qty
                        cost_basis_fixes += 1
                        actions_taken.append({
                            "action": "FIX_BELOW_COST_SELL",
                            "inst_id": inst_id,
                            "old_px": ord_px,
                            "new_floor_px": cost_floor_px,
                        })
                    # AUDIT SAFEGUARD 2: Dynamic Profit-Lock Re-Pegging
                    # Tier 1: Unrealized >= +85 bps and order > 12 bps away -> re-peg tight (+5 bps above mid)
                    # Tier 2: Unrealized >= +180 bps and order > 8 bps away -> re-peg ultra-tight at best ask
                    elif (
                        (unrealized_bps >= 180.0 and drift_bps > 8.0)
                        or (unrealized_bps >= 85.0 and drift_bps > 12.0 and pred.predicted_return_bps < 6.0)
                    ):
                        client.cancel_order(inst_id, ord_id)
                        has_sell_order = False
                        avail_token_qty = cash_token_qty
                        profit_lock_repegs += 1
                        actions_taken.append({
                            "action": "PROFIT_LOCK_REPEG",
                            "inst_id": inst_id,
                            "unrealized_bps": round(unrealized_bps, 1),
                            "old_px": ord_px,
                        })
            except Exception:
                pass

        # 3A. Place Take-Profit SELL orders enforcing Cost-Basis Floor & Profit-Lock
        if not has_sell_order and avail_token_qty * sig.mid_price >= 10.0:
            sell_sz = math.floor(avail_token_qty * (10 ** dec_sz)) / (10 ** dec_sz)
            if unrealized_bps >= 180.0:
                # Ultra-tight top-of-book Ask join to harvest large unrealized profit immediately at 0% maker fee
                raw_tp_px = asks[0][0]
                tp_style_tag = "PROFIT_LOCK_ASK_JOIN"
            elif unrealized_bps >= 85.0 and pred.predicted_return_bps < 6.0:
                # Tight profit-harvesting (+5 bps)
                raw_tp_px = sig.mid_price * 1.0005
                tp_style_tag = "PROFIT_LOCK_TP"
            else:
                raw_tp_px = sig.mid_price * (1.0 + pred.recommended_sell_tp_bps / 10000.0)
                tp_style_tag = "PASSIVE_MAKER_TP"

            target_sell_px = round(max(raw_tp_px, cost_floor_px, asks[0][0]), dec_px)
            if sell_sz > 0 and target_sell_px > 0:
                res_sell = client.place_spot_order(
                    inst_id=inst_id,
                    side="sell",
                    sz=sell_sz,
                    px=target_sell_px,
                    execution_style="PASSIVE_MAKER",
                )
                ord_id_sell = ""
                if isinstance(res_sell.get("data"), list) and res_sell["data"]:
                    ord_id_sell = str(res_sell["data"][0].get("ordId", ""))
                actions_taken.append({
                    "inst_id": inst_id,
                    "side": "sell",
                    "style": tp_style_tag,
                    "sz": sell_sz,
                    "px": target_sell_px,
                    "cost_floor_px": cost_floor_px,
                    "pred_bps": pred.predicted_return_bps,
                    "regime": pred.regime,
                    "ord_id": ord_id_sell,
                })

        # 3B. Place BUY order only if inventory weight < 25% NAV Hard Cap and free_usdc >= $12
        base_lot_usd = (nav_usdc * 0.92) / len(DEFAULT_UNIVERSE)
        conv_boost = 1.0 + max(-0.35, min(0.65, llm_bias * 0.5))
        lot_usd = min(free_usdc * 0.95, max(12.0, round(base_lot_usd * agg_mult * conv_boost, 2)))

        if pred.allow_sniper_IOC and free_usdc >= lot_usd:
            exec_style = "SNIPER_IOC"
            target_buy_px = round(asks[0][0] * 1.0005, dec_px)
        else:
            exec_style = "PASSIVE_MAKER"
            target_buy_px = round(sig.mid_price * (1.0 - pred.recommended_buy_offset_bps / 10000.0), dec_px)

        buy_sz = round(lot_usd / target_buy_px, dec_sz) if target_buy_px > 0 else 0.0

        if not has_buy_order and buy_sz > 0 and free_usdc >= 12.0 and inv_weight < 0.25:
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

            if ord_id_placed:
                free_usdc = max(0.0, free_usdc - lot_usd)

            order_tuples.append(
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
                )
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

    # Commit all telemetry and order rows in a single fast atomic transaction (prevents SQLite lock across HTTP calls)
    try:
        if telemetry_tuples:
            conn.executemany(
                """
                INSERT INTO microstructure_telemetry (
                    timestamp, inst_id, mid_price, micro_price, spread_bps,
                    obi_top5, ofi_norm, vpin_toxicity, hawkes_net_burst, composite_alpha,
                    predicted_return_bps, kalman_velocity_bps, confidence, regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                telemetry_tuples,
            )
        if order_tuples:
            conn.executemany(
                """
                INSERT INTO predator_orders (
                    timestamp, inst_id, side, execution_style, size, price,
                    predicted_bps, regime, ord_id, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                order_tuples,
            )
        conn.commit()
    except Exception as e:
        print(f"[!] SQLite commit warning: {e}")

    pred_engine.save_state()

    debate_res = None
    if run_llm_debate and telemetry_rows:
        debate_res = council.run_council_debate(
            telemetry=telemetry_rows,
            nav_usdc=nav_usdc,
            free_usdc=free_usdc,
            trigger_reason="predator_cycle_synthesis",
        )

    audit_res = None
    if run_audit:
        audit_res = auditor.run_full_audit(
            conn=conn,
            current_nav_usd=nav_usdc,
            holdings_summary=holdings_summary,
            cost_basis_fixes=cost_basis_fixes,
            profit_lock_repegs=profit_lock_repegs,
            query_ai=True,
        )

    return {
        "timestamp": now_iso,
        "nav_usdc": round(nav_usdc, 2),
        "free_usdc": round(free_usdc, 2),
        "authenticated": client.is_authenticated,
        "simulated_demo": client.simulated_demo,
        "cost_basis_fixes": cost_basis_fixes,
        "profit_lock_repegs": profit_lock_repegs,
        "telemetry": telemetry_rows,
        "actions": actions_taken,
        "llm_debate": asdict(debate_res) if debate_res else None,
        "audit_report": asdict(audit_res) if audit_res else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRE PREDATOR v1.2 — Hardened Predictive & Kronos GPU Trading OS")
    parser.add_argument("--once", action="store_true", help="Run a single predictive scan cycle and exit")
    parser.add_argument("--debate", action="store_true", help="Run the 3-Agent LLM Adversarial Council debate")
    parser.add_argument("--audit", action="store_true", help="Run the Scientific Nemotron + FIFO Ledger Audit")
    parser.add_argument("--interval", type=float, default=15.0, help="Loop interval in seconds (default: 15s)")
    parser.add_argument("--live", action="store_true", help="Disable x-simulated-trading: 1 flag (use real account)")
    args = parser.parse_args()

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    simulated_demo = not args.live
    if os.environ.get("OKX_SIMULATED_DEMO", "1") == "0":
        simulated_demo = False

    db_path = os.path.join(BASE_DIR, "data", "predator.db")
    client = OKXNativeClient(simulated_demo=simulated_demo)
    micro_engine = MicrostructureEngine(bucket_volume_usd=2500.0)
    pred_engine = PredatorPredictiveEngine(state_path=os.path.join(BASE_DIR, "data", "rls_model_state.json"))
    kronos_client = KronosOracleClient()
    council = AdversarialLLMCouncil(
        directive_out_path=os.path.join(BASE_DIR, "config", "predator_llm_directive.json")
    )
    auditor = PredatorAuditor(
        db_path=db_path,
        report_out_path=os.path.join(BASE_DIR, "config", "predator_audit_report.json"),
    )
    conn = init_predator_db(db_path)

    if args.once:
        summary = run_predator_cycle(
            client=client,
            micro_engine=micro_engine,
            pred_engine=pred_engine,
            kronos_client=kronos_client,
            council=council,
            auditor=auditor,
            conn=conn,
            run_llm_debate=args.debate,
            run_audit=args.audit,
        )
        print("=== TRE PREDATOR v1.2 (HARDENED AUDIT + KRONOS-SMALL GPU + OFI/VPIN/RLS) ===")
        print(
            f"Timestamp: {summary['timestamp']} | NAV: ${summary['nav_usdc']:.2f} USD | "
            f"Free Cash: ${summary['free_usdc']:.2f} USD | "
            f"Fixes: {summary['cost_basis_fixes']} Cost-Floor / {summary['profit_lock_repegs']} Profit-Lock"
        )
        print("-" * 124)
        print(
            f"{'INST_ID':<11} {'MID_PX':<9} {'AVG_COST':<9} {'UNR_BPS':<9} "
            f"{'OFI':<6} {'VPIN':<5} {'KRONOS_1H':<10} {'PRED_BPS':<10} {'REGIME':<21} {'BUY_BPS':<8} {'TP_BPS'}"
        )
        print("-" * 124)
        for t in summary["telemetry"]:
            print(
                f"{t['inst_id']:<11} ${t['mid_price']:<8} ${t['open_avg_px']:<8} {t['unrealized_bps']:>+7.1f}  "
                f"{t['ofi_norm']:>+5.2f} {t['vpin_toxicity']:<5.2f} {t['kronos_1h_bps']:>+7.2f}bps "
                f"{t['predicted_return_bps']:>+7.2f}bps {t['regime']:<21} -{t['buy_offset_bps']:<6.1f} +{t['sell_tp_bps']:.1f}"
            )
        if summary["actions"]:
            print("\n=== SELF-HEALING & EXECUTION ACTIONS TAKEN ===")
            for act in summary["actions"]:
                print(" ", act)
        if summary["audit_report"]:
            ar = summary["audit_report"]
            print("\n=== SCIENTIFIC & ACCOUNTING AUDIT REPORT ===")
            print(
                f"Score: {ar['audit_score_100']}/100 | Auditor: {ar['auditor_model']} | "
                f"Baseline: ${ar['baseline_nav_usd']:.2f} -> Live NAV: ${ar['current_nav_usd']:.2f} "
                f"(+${ar['net_pnl_usd']:.2f} / +{ar['net_pnl_pct']:.2f}%)"
            )
            print(
                f"Ledger Fills: {ar['total_fills_count']} (Buy: {ar['buy_fills_count']}, Sell: {ar['sell_fills_count']}, "
                f"Maker Ratio: {ar['maker_ratio_pct']}%) | Realized PnL: +${ar['realized_pnl_usd']:.4f} USD | "
                f"Unrealized MTM: ${ar['unrealized_mtm_usd']:+.2f} USD"
            )
            print(f"Verdict: {ar['executive_verdict']}")
        conn.close()
        return

    print(f"[⚡ TRE PREDATOR v1.2] Starting hardened continuous loop (interval={args.interval}s)...")
    cycle_num = 0
    try:
        while True:
            cycle_num += 1
            do_debate = (cycle_num % 20 == 1) or args.debate
            do_audit = (cycle_num % 40 == 1) or args.audit
            summary = run_predator_cycle(
                client=client,
                micro_engine=micro_engine,
                pred_engine=pred_engine,
                kronos_client=kronos_client,
                council=council,
                auditor=auditor,
                conn=conn,
                run_llm_debate=do_debate,
                run_audit=do_audit,
            )
            print(
                f"[{summary['timestamp']}] Cycle #{cycle_num} | NAV: ${summary['nav_usdc']:.2f} | "
                f"Actions: {len(summary['actions'])} | Fixes: {summary['cost_basis_fixes']}/{summary['profit_lock_repegs']}"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[!] Shutting down TRE PREDATOR v1.2 cleanly.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
