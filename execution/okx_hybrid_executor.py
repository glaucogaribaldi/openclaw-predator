#!/usr/bin/env python3
"""
TRE PREDATOR v1.0 — Native OKX v5 REST + Demo Adapter & Hybrid Execution Engine
Zero external SDK bloat: uses Python standard library (`hmac`, `hashlib`, `base64`, `urllib.request`)
with built-in support for:
- OKX Demo Trading (`x-simulated-trading: 1` header)
- OKX Live Trading (`x-simulated-trading: 0`)
- Dry-Run Paper Simulation (when API keys are not yet inserted)

Hybrid Execution Modes:
1. PASSIVE_MAKER (`post_only`): Captures spread with 0.00% Taker fee drag.
2. SNIPER_IOC (`ioc` / immediate-or-cancel): Triggered ONLY when the Online RLS Predictor +
   Cont-Stoikov OFI + Hawkes Burst exceed the breakout threshold, capturing momentum before the book moves.
3. DYNAMIC_QUOTE_CHASER: Automatically cancels and replaces stale resting orders when mid-price
   drifts away by > 18 bps, eliminating multi-day capital stagnation.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from core.microstructure import TradeEvent


def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class OKXNativeClient:
    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        simulated_demo: bool = True,
        base_url: str = "https://www.okx.com",
    ):
        self.api_key = api_key or os.environ.get("OKX_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("OKX_SECRET_KEY", "")
        self.passphrase = passphrase or os.environ.get("OKX_PASSPHRASE", "")
        self.simulated_demo = simulated_demo
        self.base_url = base_url.rstrip("/")

    @property
    def is_authenticated(self) -> bool:
        return bool(self.api_key and self.secret_key and self.passphrase and "YOUR_" not in self.api_key)

    def _timestamp_iso(self) -> str:
        now = datetime.datetime.now(datetime.timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(now.microsecond / 1000):03d}Z"

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body_dict: Optional[Dict[str, Any]] = None,
        auth: bool = False,
        timeout: int = 8,
    ) -> Dict[str, Any]:
        request_path = path
        if params:
            qs = urllib.parse.urlencode(params)
            request_path = f"{path}?{qs}"

        body_str = json.dumps(body_dict) if body_dict else ""
        url = f"{self.base_url}{request_path}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TRE-Predator-QuantOS/1.0",
        }
        if auth and self.simulated_demo:
            headers["x-simulated-trading"] = "1"

        if auth:
            if not self.is_authenticated:
                return {"code": "DRY_RUN", "msg": "No API keys configured (Dry-Run Mode)", "data": []}
            ts = self._timestamp_iso()
            headers["OK-ACCESS-KEY"] = self.api_key
            headers["OK-ACCESS-SIGN"] = self._sign(ts, method, request_path, body_str)
            headers["OK-ACCESS-TIMESTAMP"] = ts
            headers["OK-ACCESS-PASSPHRASE"] = self.passphrase

        for attempt in range(3):
            try:
                data_bytes = body_str.encode("utf-8") if body_str else None
                req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
                with urllib.request.urlopen(req, timeout=timeout, context=_build_ssl_context()) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(0.25 * (attempt + 1))
        return {}

    # --- Public Market Data (Always Live from OKX v5) ---
    def get_orderbook(self, inst_id: str, sz: int = 5) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        res = self._request("GET", "/api/v5/market/books", params={"instId": inst_id, "sz": str(sz)}, auth=False)
        data = res.get("data", [])
        if not data:
            return [], []
        bids = [(float(b[0]), float(b[1])) for b in data[0].get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in data[0].get("asks", [])]
        return bids, asks

    def get_recent_trades(self, inst_id: str, limit: int = 25) -> List[TradeEvent]:
        res = self._request("GET", "/api/v5/market/trades", params={"instId": inst_id, "limit": str(limit)}, auth=False)
        events: List[TradeEvent] = []
        for item in res.get("data", []):
            try:
                events.append(
                    TradeEvent(
                        timestamp=float(item["ts"]) / 1000.0,
                        price=float(item["px"]),
                        size=float(item["sz"]),
                        side=str(item["side"]).lower(),
                    )
                )
            except Exception:
                pass
        events.sort(key=lambda x: x.timestamp)
        return events

    # --- Private Account & Order Execution (Demo or Live) ---
    def get_account_balances(self) -> List[Dict[str, Any]]:
        res = self._request("GET", "/api/v5/account/balance", auth=True)
        data = res.get("data", [])
        if data and "details" in data[0]:
            return data[0]["details"]
        return []

    def get_open_orders(self, inst_type: str = "SPOT") -> List[Dict[str, Any]]:
        res = self._request("GET", "/api/v5/trade/orders-pending", params={"instType": inst_type}, auth=True)
        return res.get("data", [])

    def cancel_order(self, inst_id: str, ord_id: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v5/trade/cancel-order",
            body_dict={"instId": inst_id, "ordId": str(ord_id)},
            auth=True,
        )

    def place_spot_order(
        self,
        inst_id: str,
        side: str,
        sz: float,
        px: Optional[float] = None,
        execution_style: str = "PASSIVE_MAKER",
    ) -> Dict[str, Any]:
        """
        execution_style:
        - "PASSIVE_MAKER": ordType="post_only" (0.00% Taker fee, pure maker)
        - "SNIPER_IOC":    ordType="ioc" with aggressive limit price (immediate breakout capture)
        - "LIMIT_STANDARD": ordType="limit"
        """
        if execution_style == "PASSIVE_MAKER":
            ord_type = "post_only"
        elif execution_style == "SNIPER_IOC":
            ord_type = "ioc"
        else:
            ord_type = "limit"

        payload: Dict[str, Any] = {
            "instId": inst_id,
            "tdMode": "cash",
            "side": side.lower(),
            "ordType": ord_type,
            "sz": str(sz),
        }
        if px is not None:
            payload["px"] = str(px)

        return self._request("POST", "/api/v5/trade/order", body_dict=payload, auth=True)
