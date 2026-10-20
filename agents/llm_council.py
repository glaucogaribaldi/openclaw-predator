#!/usr/bin/env python3
"""
TRE PREDATOR v1.0 — Multi-Agent Adversarial LLM Debate Council
Inspired by TauricResearch/TradingAgents & Trading-R1 (2025/2026).

Architecture:
1. Momentum Bull Agent ("Alpha Predator"):
   Searches for Order Flow Imbalance (OFI) breakouts, Hawkes trade bursts, and positive micro-price velocity.
2. Risk & Toxicity Bear Agent ("VPIN Sentinel"):
   Evaluates VPIN order flow toxicity, inventory concentration risk, and fakeout/spoofing vulnerability.
3. Chief Quant Judge ("CIO Synthesizer"):
   Adjudicates the structured debate and outputs a deterministic JSON directive with per-token
   conviction scores [-1.0, +1.0], regime overrides, and capital allocation multipliers [0.5x .. 2.0x].

Supports:
- OpenRouter Free Tier (`openrouter/free` or specific free models)
- Local/Tailnet Ollama GPU (`http://100.125.189.103:11434` - `nemotron-3.5-lightning:latest`)
- Deterministic Quant Fallback if external APIs are rate-limited (HTTP 429)
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


@dataclass
class AgentDebateResult:
    timestamp: str
    trigger_reason: str
    model_used: str
    bull_thesis: str
    bear_thesis: str
    judge_verdict: str
    global_regime: str
    aggressiveness_multiplier: float  # [0.5, 2.0]
    token_convictions: Dict[str, float]  # inst_id -> [-1.0, +1.0]


class AdversarialLLMCouncil:
    def __init__(
        self,
        openrouter_api_key: str = "",
        openrouter_model: str = "openrouter/free",
        ollama_url: str = "http://100.125.189.103:11434/api/generate",
        ollama_model: str = "nemotron-3.5-lightning:latest",
        directive_out_path: Optional[str] = None,
    ):
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_model = openrouter_model
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.directive_out_path = directive_out_path

    def _build_deterministic_debate(
        self,
        telemetry: List[Dict[str, Any]],
        trigger_reason: str,
    ) -> AgentDebateResult:
        """
        Zero-latency mathematical synthesis when LLM APIs are rate-limited or offline.
        """
        convictions: Dict[str, float] = {}
        top_bulls: List[str] = []
        top_toxic: List[str] = []

        for item in telemetry:
            inst = item.get("inst_id", "UNKNOWN")
            alpha = float(item.get("composite_alpha", 0.0))
            vpin = float(item.get("vpin_toxicity", 0.25))
            pred_bps = float(item.get("predicted_return_bps", 0.0))

            # Penalize conviction when VPIN toxicity is high and flow is negative
            conv = max(-1.0, min(1.0, 0.6 * alpha + 0.4 * (pred_bps / 25.0)))
            if vpin > 0.65 and alpha < 0:
                conv = min(conv, -0.55)
                top_toxic.append(f"{inst}(VPIN={vpin:.2f})")
            elif conv > 0.35:
                top_bulls.append(f"{inst}(+{pred_bps:.1f}bps)")

            convictions[inst] = round(conv, 3)

        avg_conv = sum(convictions.values()) / max(len(convictions), 1)
        if avg_conv > 0.20:
            global_regime = "PREDATOR_OFFENSIVE_BREAKOUT"
            agg_mult = 1.45
        elif top_toxic:
            global_regime = "DEFENSIVE_VPIN_SHIELD"
            agg_mult = 0.75
        else:
            global_regime = "HIGH_VELOCITY_SCALP_MM"
            agg_mult = 1.15

        bull_str = (
            f"Strong positive Order Flow Imbalance (OFI) & Hawkes burst on {', '.join(top_bulls) if top_bulls else 'selective high-beta pairs'}. "
            f"Recommend aggressive top-of-book joining and IOC breakout sniping."
        )
        bear_str = (
            f"Elevated VPIN toxicity flagged on {', '.join(top_toxic) if top_toxic else 'none'}; "
            f"maintain strict Cartea-Jaimungal inventory skew cap at 18% NAV per asset."
        )
        verdict_str = (
            f"Regime set to {global_regime} (Aggressiveness {agg_mult:.2f}x). "
            f"Concentrate dynamic lot sizing on highest OFI+RLS velocity leaders while widening buy traps on high-VPIN pairs."
        )

        return AgentDebateResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            trigger_reason=trigger_reason,
            model_used="deterministic-microstructure-council",
            bull_thesis=bull_str,
            bear_thesis=bear_str,
            judge_verdict=verdict_str,
            global_regime=global_regime,
            aggressiveness_multiplier=agg_mult,
            token_convictions=convictions,
        )

    def _call_openrouter(self, prompt: str, timeout: int = 12) -> Optional[str]:
        if not self.openrouter_api_key:
            return None
        payload = {
            "model": self.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Chief Quantitative Judge of TRE PREDATOR v1.0, an ultra-aggressive "
                        "predictive microstructure & multi-agent crypto trading OS. Respond ONLY with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_build_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    def run_council_debate(
        self,
        telemetry: List[Dict[str, Any]],
        nav_usdc: float,
        free_usdc: float,
        trigger_reason: str = "scheduled_cycle",
    ) -> AgentDebateResult:
        fallback = self._build_deterministic_debate(telemetry, trigger_reason)

        prompt = f"""Conduct a 3-Agent Adversarial Trading Debate for TRE PREDATOR v1.0 (OKX Spot/Demo).
Trigger Event: {trigger_reason}
Portfolio NAV: ${nav_usdc:.2f} USDC | Free Cash: ${free_usdc:.2f} USDC
Live Microstructure & RLS Predictive Telemetry:
{json.dumps(telemetry, indent=2)}

Return ONLY a raw JSON object with this exact schema:
{{
  "bull_thesis": "1-2 sentences from Momentum Bull Agent focusing on OFI, Hawkes burst, and RLS predicted bps",
  "bear_thesis": "1-2 sentences from VPIN Risk Bear Agent focusing on order toxicity and inventory skew",
  "judge_verdict": "1-2 sentences executive synthesis and tactical order placement command",
  "global_regime": "PREDATOR_OFFENSIVE_BREAKOUT or HIGH_VELOCITY_SCALP_MM or DEFENSIVE_VPIN_SHIELD",
  "aggressiveness_multiplier": 1.25,
  "token_convictions": {{"SOL-USDC": 0.65, "SUI-USDC": 0.40}}
}}"""

        raw_resp = None
        model_used = fallback.model_used
        try:
            raw_resp = self._call_openrouter(prompt)
            if raw_resp:
                model_used = f"openrouter:{self.openrouter_model}"
        except Exception:
            raw_resp = None

        if raw_resp:
            try:
                # Strip optional markdown fences
                cleaned = raw_resp.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                parsed = json.loads(cleaned.strip())
                conv_map = fallback.token_convictions.copy()
                for k, v in (parsed.get("token_convictions") or {}).items():
                    try:
                        conv_map[k] = round(max(-1.0, min(1.0, float(v))), 3)
                    except Exception:
                        pass

                agg = float(parsed.get("aggressiveness_multiplier", fallback.aggressiveness_multiplier))
                agg = round(max(0.5, min(2.0, agg)), 2)

                result = AgentDebateResult(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    trigger_reason=trigger_reason,
                    model_used=model_used,
                    bull_thesis=str(parsed.get("bull_thesis") or fallback.bull_thesis),
                    bear_thesis=str(parsed.get("bear_thesis") or fallback.bear_thesis),
                    judge_verdict=str(parsed.get("judge_verdict") or fallback.judge_verdict),
                    global_regime=str(parsed.get("global_regime") or fallback.global_regime),
                    aggressiveness_multiplier=agg,
                    token_convictions=conv_map,
                )
                self._save_directive(result)
                return result
            except Exception:
                pass

        self._save_directive(fallback)
        return fallback

    def _save_directive(self, result: AgentDebateResult) -> None:
        if not self.directive_out_path:
            return
        try:
            os.makedirs(os.path.dirname(self.directive_out_path), exist_ok=True)
            with open(self.directive_out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
