#!/usr/bin/env python3
"""
TRE PREDATOR v1.1 — Kronos-small (24.7M) Foundation Model Client (`core/kronos_oracle.py`)
Connects over Tailscale to the `kronos-oracle.service` running on `geneticai-nemotron-spark`
(`http://100.125.189.103:8090/forecast`), incurring 0 MB RAM overhead on `brody`.

Provides:
- `expected_return_1h_bps`: Autoregressive 12x5m K-line Close forecast in basis points
- `upside_high_bps`: Predicted 1h High envelope in bps (calibrates dynamic Take-Profit)
- `downside_low_bps`: Predicted 1h Low envelope in bps (calibrates dynamic Buy Trap depth)
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class KronosForecast:
    inst_id: str
    last_close: float
    pred_1h_high: float
    pred_1h_low: float
    pred_1h_close: float
    upside_high_bps: float
    downside_low_bps: float
    expected_return_1h_bps: float
    infer_ms: float


class KronosOracleClient:
    def __init__(self, endpoint_url: str = "http://100.125.189.103:8090"):
        self.endpoint_url = (os.environ.get("KRONOS_ORACLE_URL") or endpoint_url).rstrip("/")
        self._last_cache: Dict[str, KronosForecast] = {}

    def fetch_forecasts(self, pairs: List[str], timeout: float = 12.0) -> Dict[str, KronosForecast]:
        if not pairs:
            return {}
        try:
            qs = urllib.parse.urlencode({"pairs": ",".join(pairs)})
            url = f"{self.endpoint_url}/forecast?{qs}"
            req = urllib.request.Request(url, headers={"User-Agent": "TRE-Predator-KronosClient/1.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            raw_map = payload.get("forecasts", {})
            for inst_id, item in raw_map.items():
                if isinstance(item, dict) and "expected_return_1h_bps" in item:
                    self._last_cache[inst_id] = KronosForecast(
                        inst_id=inst_id,
                        last_close=float(item.get("last_close", 0.0)),
                        pred_1h_high=float(item.get("pred_1h_high", 0.0)),
                        pred_1h_low=float(item.get("pred_1h_low", 0.0)),
                        pred_1h_close=float(item.get("pred_1h_close", 0.0)),
                        upside_high_bps=float(item.get("upside_high_bps", 25.0)),
                        downside_low_bps=float(item.get("downside_low_bps", 25.0)),
                        expected_return_1h_bps=float(item.get("expected_return_1h_bps", 0.0)),
                        infer_ms=float(item.get("infer_ms", 0.0)),
                    )
        except Exception:
            pass
        return self._last_cache
