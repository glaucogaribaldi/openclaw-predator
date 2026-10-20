#!/usr/bin/env python3
"""
TRE PREDATOR v1.0 — Market Microstructure Alpha Engine
Implements academic high-frequency microstructure indicators in pure Python (zero heavy dependencies):
1. Multi-Level Order Flow Imbalance (OFI) — Cont, Kukanov & Stoikov (2014) / Cartea & Jaimungal (2018)
2. Volume-Synchronized Probability of Informed Trading (VPIN) — Easley, López de Prado & O'Hara (2012)
3. Hawkes Self-Exciting Process Intensity (λ_buy, λ_sell) — Bacry et al. (2015) for trade burst detection
4. Micro-Price & Book PressureSkew — Stoikov (2018)
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BookSnapshot:
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]


@dataclass
class TradeEvent:
    timestamp: float
    price: float
    size: float
    side: str  # "buy" or "sell"


@dataclass
class MicrostructureSignal:
    inst_id: str
    timestamp: float
    mid_price: float
    micro_price: float
    spread_bps: float
    obi_top5: float          # [-1.0, +1.0] Order Book Imbalance
    ofi_norm: float          # [-1.0, +1.0] Cont-Stoikov Order Flow Imbalance
    vpin_toxicity: float     # [0.0, 1.0]   Order Flow Toxicity (Easley-López de Prado)
    hawkes_buy_intensity: float
    hawkes_sell_intensity: float
    hawkes_net_burst: float  # [-1.0, +1.0] Directional burst excitation
    composite_alpha: float   # [-1.0, +1.0] Combined short-term directional pressure


class HawkesIntensityTracker:
    """
    Univariate/Bivariate Exponential Hawkes Process tracker:
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    Updated recursively in O(1) time per event.
    """

    def __init__(self, mu: float = 0.5, alpha: float = 1.8, beta: float = 2.5):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.last_time: float = time.time()
        self.buy_excitation: float = 0.0
        self.sell_excitation: float = 0.0

    def _decay(self, now: float) -> None:
        dt = max(0.0, now - self.last_time)
        decay_factor = math.exp(-self.beta * dt)
        self.buy_excitation *= decay_factor
        self.sell_excitation *= decay_factor
        self.last_time = now

    def record_trade(self, side: str, notional_weight: float = 1.0, now: Optional[float] = None) -> None:
        ts = now if now is not None else time.time()
        self._decay(ts)
        bump = self.alpha * min(max(notional_weight, 0.2), 5.0)
        if side.lower() == "buy":
            self.buy_excitation += bump
        else:
            self.sell_excitation += bump

    def get_intensities(self, now: Optional[float] = None) -> Tuple[float, float, float]:
        ts = now if now is not None else time.time()
        self._decay(ts)
        lam_buy = self.mu + self.buy_excitation
        lam_sell = self.mu + self.sell_excitation
        total = lam_buy + lam_sell
        net_burst = (lam_buy - lam_sell) / total if total > 0 else 0.0
        return lam_buy, lam_sell, net_burst


class VPINTracker:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).
    Accumulates trades into volume buckets V and measures |V_buy - V_sell| / V
    across a rolling window of N buckets.
    """

    def __init__(self, bucket_volume_usd: float = 2500.0, window_buckets: int = 20):
        self.bucket_volume_usd = max(bucket_volume_usd, 100.0)
        self.window_buckets = window_buckets
        self.current_buy_usd: float = 0.0
        self.current_sell_usd: float = 0.0
        self.buckets: deque[float] = deque(maxlen=window_buckets)

    def update_from_trades(self, trades: List[TradeEvent]) -> float:
        for tr in trades:
            notional = tr.price * tr.size
            if tr.side.lower() == "buy":
                self.current_buy_usd += notional
            else:
                self.current_sell_usd += notional

            total_cur = self.current_buy_usd + self.current_sell_usd
            while total_cur >= self.bucket_volume_usd:
                ratio_buy = self.current_buy_usd / total_cur
                ratio_sell = self.current_sell_usd / total_cur
                imbalance = abs(ratio_buy - ratio_sell)
                self.buckets.append(imbalance)
                # Spill over excess proportional to current mix
                excess = total_cur - self.bucket_volume_usd
                self.current_buy_usd = excess * ratio_buy
                self.current_sell_usd = excess * ratio_sell
                total_cur = self.current_buy_usd + self.current_sell_usd

        return self.get_vpin()

    def get_vpin(self) -> float:
        if not self.buckets:
            tot = self.current_buy_usd + self.current_sell_usd
            if tot <= 0:
                return 0.25
            return abs(self.current_buy_usd - self.current_sell_usd) / tot
        return sum(self.buckets) / len(self.buckets)


class MicrostructureEngine:
    """
    Tracks per-instrument Level-2 book transitions and trade tape to produce
    high-signal predictive microstructure features.
    """

    def __init__(self, bucket_volume_usd: float = 2500.0):
        self.prev_books: Dict[str, BookSnapshot] = {}
        self.ofi_windows: Dict[str, deque[float]] = {}
        self.hawkes: Dict[str, HawkesIntensityTracker] = {}
        self.vpin: Dict[str, VPINTracker] = {}
        self.seen_trade_ids: Dict[str, deque[str]] = {}
        self.bucket_volume_usd = bucket_volume_usd

    def _compute_cont_stoikov_ofi(self, prev: BookSnapshot, curr: BookSnapshot) -> float:
        """
        Cont, Kukanov & Stoikov (2014) multi-level Order Flow Imbalance with exponential depth decay.
        """
        ofi_total = 0.0
        levels = min(len(prev.bids), len(curr.bids), len(prev.asks), len(curr.asks), 5)
        for lvl in range(levels):
            weight = math.exp(-0.45 * lvl)
            p_b_prev, q_b_prev = prev.bids[lvl]
            p_b_curr, q_b_curr = curr.bids[lvl]

            if p_b_curr > p_b_prev:
                e_b = q_b_curr
            elif p_b_curr == p_b_prev:
                e_b = q_b_curr - q_b_prev
            else:
                e_b = -q_b_prev

            p_a_prev, q_a_prev = prev.asks[lvl]
            p_a_curr, q_a_curr = curr.asks[lvl]

            if p_a_curr > p_a_prev:
                e_a = -q_a_prev
            elif p_a_curr == p_a_prev:
                e_a = q_a_curr - q_a_prev
            else:
                e_a = q_a_curr

            ofi_total += weight * (e_b - e_a)
        return ofi_total

    def process_snapshot(
        self,
        inst_id: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        recent_trades: Optional[List[TradeEvent]] = None,
    ) -> MicrostructureSignal:
        now = time.time()
        if not bids or not asks:
            return MicrostructureSignal(
                inst_id=inst_id,
                timestamp=now,
                mid_price=0.0,
                micro_price=0.0,
                spread_bps=0.0,
                obi_top5=0.0,
                ofi_norm=0.0,
                vpin_toxicity=0.0,
                hawkes_buy_intensity=0.5,
                hawkes_sell_intensity=0.5,
                hawkes_net_burst=0.0,
                composite_alpha=0.0,
            )

        best_bid, best_bid_sz = bids[0]
        best_ask, best_ask_sz = asks[0]
        mid_price = 0.5 * (best_bid + best_ask)
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000.0 if mid_price > 0 else 0.0

        # Stoikov Micro-Price weighted by immediate top-of-book inverse queue size
        top_tot = best_bid_sz + best_ask_sz
        if top_tot > 0:
            micro_price = (best_bid * best_ask_sz + best_ask * best_bid_sz) / top_tot
        else:
            micro_price = mid_price

        # Weighted top-5 Order Book Imbalance (OBI)
        bid_vol_w = 0.0
        ask_vol_w = 0.0
        for i in range(min(5, len(bids), len(asks))):
            w = math.exp(-0.35 * i)
            bid_vol_w += w * bids[i][0] * bids[i][1]
            ask_vol_w += w * asks[i][0] * asks[i][1]
        obi_top5 = (bid_vol_w - ask_vol_w) / (bid_vol_w + ask_vol_w) if (bid_vol_w + ask_vol_w) > 0 else 0.0

        # Cont-Stoikov OFI
        curr_snap = BookSnapshot(timestamp=now, bids=bids[:5], asks=asks[:5])
        if inst_id not in self.ofi_windows:
            self.ofi_windows[inst_id] = deque(maxlen=30)

        if inst_id in self.prev_books:
            raw_ofi = self._compute_cont_stoikov_ofi(self.prev_books[inst_id], curr_snap)
            self.ofi_windows[inst_id].append(raw_ofi)
        self.prev_books[inst_id] = curr_snap

        ofi_list = list(self.ofi_windows[inst_id])
        if ofi_list:
            ofi_sum = sum(ofi_list[-10:])
            ofi_abs = sum(abs(x) for x in ofi_list[-10:]) + 1e-9
            ofi_norm = max(-1.0, min(1.0, ofi_sum / ofi_abs))
        else:
            ofi_norm = obi_top5

        # Hawkes & VPIN updates from recent trades
        if inst_id not in self.hawkes:
            self.hawkes[inst_id] = HawkesIntensityTracker()
        if inst_id not in self.vpin:
            self.vpin[inst_id] = VPINTracker(bucket_volume_usd=self.bucket_volume_usd)

        if recent_trades:
            for tr in recent_trades:
                notional = tr.price * tr.size
                weight = notional / 250.0
                self.hawkes[inst_id].record_trade(tr.side, notional_weight=weight, now=tr.timestamp)
            vpin_val = self.vpin[inst_id].update_from_trades(recent_trades)
        else:
            vpin_val = self.vpin[inst_id].get_vpin()

        lam_buy, lam_sell, net_burst = self.hawkes[inst_id].get_intensities(now)

        # Micro-price displacement in bps normalized
        micro_shift = ((micro_price - mid_price) / (best_ask - best_bid + 1e-12)) * 2.0
        micro_shift = max(-1.0, min(1.0, micro_shift))

        # Composite Alpha: blends OBI (25%), Cont-Stoikov OFI (35%), Hawkes Net Burst (25%), Micro-Price Shift (15%)
        composite_alpha = (
            0.25 * obi_top5
            + 0.35 * ofi_norm
            + 0.25 * net_burst
            + 0.15 * micro_shift
        )
        composite_alpha = max(-1.0, min(1.0, composite_alpha))

        return MicrostructureSignal(
            inst_id=inst_id,
            timestamp=now,
            mid_price=mid_price,
            micro_price=micro_price,
            spread_bps=round(spread_bps, 2),
            obi_top5=round(obi_top5, 4),
            ofi_norm=round(ofi_norm, 4),
            vpin_toxicity=round(vpin_val, 4),
            hawkes_buy_intensity=round(lam_buy, 4),
            hawkes_sell_intensity=round(lam_sell, 4),
            hawkes_net_burst=round(net_burst, 4),
            composite_alpha=round(composite_alpha, 4),
        )
