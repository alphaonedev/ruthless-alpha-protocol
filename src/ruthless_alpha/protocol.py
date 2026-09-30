"""Deterministic mapping from market data and disclosed assumptions to sizing outputs."""

from __future__ import annotations

from .ruthless_luck import discrete_kelly


def trailing_return(bars: list[dict], sessions: int) -> float:
    if len(bars) < 2:
        raise ValueError("at least two bars are required")
    first = bars[max(0, len(bars) - sessions - 1)]["c"]
    last = bars[-1]["c"]
    return last / first - 1.0


def analyze(symbol: str, bars: list[dict], cash_yield: float, kill_probability: float,
            kill_return: float = -0.40, upside_probability: float = 0.25) -> dict:
    if not 0 < cash_yield < 1:
        raise ValueError("cash_yield must be a decimal in (0, 1)")
    if not 0 <= kill_probability < 1 - upside_probability:
        raise ValueError("probabilities leave no room for the carry state")
    one_year = trailing_return(bars, 252)
    three_year = trailing_return(bars, 756)
    three_year_cagr = (1 + three_year) ** (1 / 3) - 1 if three_year > -1 else -1.0
    upside = max(one_year, three_year_cagr)
    carry_probability = 1 - upside_probability - kill_probability
    returns = [upside, cash_yield, kill_return]
    probabilities = [upside_probability, carry_probability, kill_probability]
    kelly = discrete_kelly(returns, probabilities)
    return {
        "symbol": symbol.upper(), "one_year_return": one_year,
        "three_year_cagr": three_year_cagr, "cash_yield": cash_yield,
        "scenario_returns": returns, "scenario_probabilities": probabilities,
        "full_kelly": kelly, "half_kelly": kelly / 2,
        "verdict": "KEEP" if kelly > 0 else "REJECT",
    }

