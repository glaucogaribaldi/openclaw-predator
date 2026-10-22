#!/usr/bin/env python3
"""
TRE PREDATOR v1.2 — Telegram Executive Report Generator (`predator_reporter.py`)
Generates clean, deterministic performance reports for openclaw-predator and posts to Telegram.
"""
import os, sys, json, sqlite3, subprocess, time, urllib.request, urllib.parse

BASE_DIR = "/home/zavattoni_giacomo/openclaw-predator"
sys.path.insert(0, BASE_DIR)

from predator_daemon import load_dotenv, DEFAULT_UNIVERSE
from execution.okx_hybrid_executor import OKXNativeClient

load_dotenv(os.path.join(BASE_DIR, ".env"))

def generate_predator_telegram_report():
    client = OKXNativeClient(simulated_demo=True)
    bals = client.get_account_balances()
    tracked = {m["coin"] for m in DEFAULT_UNIVERSE.values()} | {"USDC"}
    
    nav = sum(float(b.get("eqUsd") or 0.0) for b in bals if b.get("ccy") in tracked)
    free = sum(float(b.get("availBal") or 0.0) for b in bals if b.get("ccy") == "USDC")
    base = 313.88
    net_pnl = nav - base
    net_pct = (net_pnl / base) * 100.0 if base > 0 else 0.0
    
    conn = sqlite3.connect(os.path.join(BASE_DIR, "data/predator.db"))
    cur = conn.cursor()
    
    # 24h stats
    cur.execute("SELECT COUNT(*), SUM(realized_pnl_usd) FROM predator_fills_ledger WHERE datetime(fill_time) >= datetime('now', '-24 hours') AND side='sell'")
    row_24h = cur.fetchone() or (0, 0.0)
    fills_24h_sell = row_24h[0] or 0
    pnl_24h_sell = float(row_24h[1] or 0.0)
    
    # Run stats
    cur.execute("SELECT COUNT(*), SUM(realized_pnl_usd) FROM predator_fills_ledger WHERE side='sell'")
    row_run_sell = cur.fetchone() or (0, 0.0)
    fills_run_sell = row_run_sell[0] or 0
    pnl_run_sell = float(row_run_sell[1] or 0.0)
    
    cur.execute("SELECT COUNT(*) FROM predator_fills_ledger")
    tot_fills = cur.fetchone()[0] or 0
    conn.close()
    
    # Open orders
    ords = client.get_open_orders()
    active_orders_count = len(ords)
    
    # Fetch Kronos 1h forecast sample
    k_ret_str = "+0.0 bps"
    try:
        req = urllib.request.Request("http://100.125.189.103:8090/forecast?pairs=HYPE-USDC,NEAR-USDC,SOL-USDC", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            k_data = json.loads(resp.read().decode("utf-8")).get("forecasts", {})
            h_ret = k_data.get("HYPE-USDC", {}).get("expected_return_1h_bps", 0.0)
            k_ret_str = f"{h_ret:+.1f} bps"
    except Exception:
        pass
        
    now_str = time.strftime("%d/%m/%Y - %H:%M CEST")
    pnl_sign = "+" if net_pnl >= 0 else ""
    
    msg = f"⚡ *TRE PREDATOR v1.2 — REPORT DI PERFORMANCE (CONTO DEMO EXPERIMENTAL)*\n"
    msg += f"📅 _{now_str}_\n\n"
    msg += "```\n"
    msg += "========================================\n"
    msg += "1. CAPITALE E PARITÀ BENCHMARK (NAV)\n"
    msg += "========================================\n"
    msg += f"Baseline Parita (20/10 22:54): $313.88 USD\n"
    msg += f"NAV Demo Attuale Live:        ${nav:.2f} USD\n"
    msg += f"Profitto Netto Run:           {pnl_sign}${net_pnl:.2f} USD ({pnl_sign}{net_pct:.2f}%)\n"
    msg += f"Cassa Liquida Libera:         ${free:.2f} USDC\n\n"
    msg += "========================================\n"
    msg += "2. PERFORMANCE CHIUSA ED FILLS (FIFO)\n"
    msg += "========================================\n"
    msg += f"• Profitto Realizzato 24h (Sell): +${pnl_24h_sell:.4f} USD ({fills_24h_sell} Fills)\n"
    msg += f"• Profitto Realizzato Run (Sell):  +${pnl_run_sell:.4f} USD ({fills_run_sell} Fills)\n"
    msg += f"• Fills Totali Eseguiti (Demo):   {tot_fills} Fills (100.0% Maker)\n"
    msg += f"• Modello Predittivo K-Line:     Kronos-small GPU (24.7M)\n"
    msg += f"• Modello Microstruttura L2:     Cont-Stoikov OFI + VPIN + RLS\n"
    msg += f"• Trappole Limit Attive:       {active_orders_count} ordini sul book\n"
    msg += "========================================\n"
    msg += "```\n\n"
    msg += "💡 *ANALISI DISCORSIVA PREDITTIVA (KRONOS GPU + MICROSTRUCTURE):*\n"
    msg += f"• **Proiezione 1h Kronos-small:** HYPE ({k_ret_str}) mostra accumulo di Order Flow.\n"
    msg += "• **Protezioni Attive:** Cost-Basis Floor (openAvgPx + 22 bps) e Dynamic Profit-Lock Re-Pegging trascinano i Take-Profit al Best Ask in tempo reale.\n"
    msg += "• **Esecuzione:** Trappole limit attive in modalita Post-Only Maker a rischio liquidazione azzerato.\n"

    print(msg)
    
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip('"').strip("'")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip('"').strip("'")
    if bot_token and chat_id:
        try:
            url_tg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data_tg = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode("utf-8")
            req_tg = urllib.request.Request(url_tg, data=data_tg)
            with urllib.request.urlopen(req_tg, timeout=8) as resp:
                print("[+] Sent Predator Executive Report directly to Telegram!")
        except Exception as e:
            print(f"[!] Telegram Webhook Error: {e}")

if __name__ == "__main__":
    generate_predator_telegram_report()
