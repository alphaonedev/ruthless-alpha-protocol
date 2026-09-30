"""Reference implementation of the Ruthless Luck Algorithm, specification v1.6.

Every function maps to a numbered part of the specification. Pure Python
standard library only; no network, file or subprocess access.

Conventions
-----------
* Luck is ``Y_T - price`` where the price is the expectation of ``Y_T`` under
  the agent's conditional law, with every absorbed (ruined) path scored at the
  ruin value ``y_R``.
* Ruin in the multiplicative lattice is ``W <= w_R`` at any step ``1..T``.
* All probabilities are returned as floats unless ``exact=True`` is requested
  where offered, in which case ``fractions.Fraction`` is used throughout.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "LuckReport",
    "cohort_summary",
    "survivor_luck",
    "price_from_parts",
    "ruin_lattice",
    "binary_growth",
    "binary_kelly",
    "discrete_growth",
    "discrete_kelly",
    "inverse_wealth_drift",
    "predictability_horizon",
    "AnytimeAlarm",
    "randomized_pit",
    "expected_max_std_normal",
    "credal_price",
    "credal_luck",
    "fuzzy_killing_price",
    "attribution_ledger",
    "sum_of_luck",
]

Number = Union[int, float, Fraction]

REPORT_SCHEMA_ID = "https://ruthless-luck.spec/v1.6/luck-report.schema.json"


def _finite(name: str, x: float) -> float:
    if isinstance(x, bool):
        raise ValueError(f"{name} must be a number, got bool")
    try:
        x = float(x)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number, got {x!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return x


def _num(name: str, x) -> float:
    """Strict numeric check for untrusted input: int or float only (no bool, str, None, containers)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(x).__name__}")
    try:
        x = float(x)
    except OverflowError:
        raise ValueError(f"{name} is outside float range") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return x


def _ident(name: str, s) -> str:
    if not isinstance(s, str) or not s or len(s) > 256:
        raise ValueError(f"{name} must be a non-empty string of at most 256 characters")
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} is not valid Unicode") from None
    return s


def _no_dup_object(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def _prob(name: str, p: float, open_interval: bool = False) -> float:
    p = _finite(name, p)
    if open_interval:
        if not (0.0 < p < 1.0):
            raise ValueError(f"{name} must lie in (0, 1), got {p!r}")
    elif not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {p!r}")
    return p


# --------------------------------------------------------------------------
# §2 step 8: the four numbers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LuckReport:
    """The four numbers of §2 step 8 for one agent and one scoring window.

    ``price`` is E_{i,t}[Y_T]; ``outcome`` is the realized Y_T (equal to
    ``ruin_value`` if the path was absorbed); ``ruin_mass`` is
    mu_{i,t}(tau_R <= T); ``survived`` is the indicator of S.
    Luck is derived, never supplied, so it cannot disagree with the rest.
    """

    agent_id: str
    t: float
    horizon: float
    price: float
    outcome: float
    ruin_mass: float
    survived: bool
    ruin_value: float
    model_id: str

    def __post_init__(self) -> None:
        _ident("agent_id", self.agent_id)
        _ident("model_id", self.model_id)
        for f in ("t", "horizon", "price", "outcome", "ruin_value", "ruin_mass"):
            object.__setattr__(self, f, _num(f, getattr(self, f)))
        if not self.horizon > self.t:
            raise ValueError("horizon must be later than t")
        if not 0.0 <= self.ruin_mass <= 1.0:
            raise ValueError("ruin_mass must lie in [0, 1]")
        if not isinstance(self.survived, bool):
            raise ValueError("survived must be a bool")
        if not self.survived and self.outcome != self.ruin_value:
            raise ValueError("an absorbed path must be scored at the ruin value (outcome == ruin_value)")
        if not self.survived and self.ruin_mass == 0.0:
            raise ValueError("path was absorbed but the price assigned ruin probability 0: the model is falsified")
        if self.survived and self.ruin_mass == 1.0:
            raise ValueError("path survived but the price assigned ruin probability 1: the model is falsified")

    @property
    def luck(self) -> float:
        return self.outcome - self.price

    def to_dict(self) -> dict:
        d = asdict(self)
        d["luck"] = self.luck
        d["$schema"] = REPORT_SCHEMA_ID
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict) -> "LuckReport":
        if not isinstance(d, dict):
            raise ValueError("report must be a JSON object")
        fields = ["agent_id", "t", "horizon", "price", "outcome", "ruin_mass", "survived", "ruin_value", "model_id"]
        missing = [f for f in fields if f not in d]
        if missing:
            raise ValueError(f"report is missing required fields: {missing}")
        allowed = set(fields) | {"luck", "$schema"}
        extra = sorted(set(d) - allowed)
        if extra:
            raise ValueError(f"report has unknown fields: {extra}")
        if "$schema" in d and d["$schema"] != REPORT_SCHEMA_ID:
            raise ValueError("unsupported $schema")
        rep = cls(**{f: d[f] for f in fields})
        if "luck" in d:
            claimed = _num("luck", d["luck"])
            if claimed != rep.luck:
                raise ValueError(f"reported luck {claimed!r} disagrees with outcome - price = {rep.luck!r}")
        return rep

    @classmethod
    def from_json(cls, s, max_bytes: int = 65536) -> "LuckReport":
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive int")
        if isinstance(s, (bytes, bytearray)):
            if len(s) > max_bytes:
                raise ValueError("report exceeds size limit")
            try:
                s = bytes(s).decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("report is not valid UTF-8") from None
        elif not isinstance(s, str):
            raise ValueError("report must be str or bytes")
        if len(s) > max_bytes or len(s.encode("utf-8", "surrogatepass")) > max_bytes:
            raise ValueError("report exceeds size limit")
        try:
            obj = json.loads(s, parse_constant=_reject_constant, object_pairs_hook=_no_dup_object)
        except RecursionError:
            raise ValueError("report nesting too deep") from None
        return cls.from_dict(obj)


def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def cohort_summary(reports: Sequence[LuckReport]) -> dict:
    """Mean luck over the whole cohort and over survivors only (Rule 4).

    The difference between the two is selection. Publishing only the
    survivor figure publishes the filter; this function always returns both.
    """
    reports = list(reports)
    if not reports:
        raise ValueError("empty cohort")
    n = len(reports)
    surv = [r for r in reports if r.survived]
    mean_all = sum(r.luck for r in reports) / n
    mean_surv = (sum(r.luck for r in surv) / len(surv)) if surv else None
    return {
        "n": n,
        "n_survived": len(surv),
        "survival_rate": len(surv) / n,
        "mean_luck_all": mean_all,
        "mean_luck_survivors": mean_surv,
        "selection_gap": (mean_surv - mean_all) if surv else None,
    }


# --------------------------------------------------------------------------
# §2 step 4 and Rule 4
# --------------------------------------------------------------------------

def price_from_parts(p_survive: Number, mean_given_survive: Number, ruin_value: Number) -> Number:
    """E[Y_T] = p E[Y_T | S] + (1 - p) y_R."""
    for n, v in (("p_survive", p_survive), ("mean_given_survive", mean_given_survive), ("ruin_value", ruin_value)):
        if not isinstance(v, Fraction):
            _finite(n, v)
    if not (0 <= p_survive <= 1):
        raise ValueError("p_survive must lie in [0, 1]")
    return p_survive * mean_given_survive + (1 - p_survive) * ruin_value


def survivor_luck(p_survive: Number, mean_given_survive: Number, ruin_value: Number) -> Number:
    """Rule 4: E[luck | S] = (1 - p)(E[Y_T | S] - y_R). Requires p in (0, 1]."""
    for n, v in (("p_survive", p_survive), ("mean_given_survive", mean_given_survive), ("ruin_value", ruin_value)):
        if not isinstance(v, Fraction):
            _finite(n, v)
    if not (0 < p_survive <= 1):
        raise ValueError("p_survive must lie in (0, 1]")
    return (1 - p_survive) * (mean_given_survive - ruin_value)


# --------------------------------------------------------------------------
# §5: exact ruin lattice
# --------------------------------------------------------------------------

EXACT_T_MAX = 1000
EXACT_WORK_MAX = 1.2e7  # T^2 * (bits of u, d, q): at most about 10 s
_EXACT_STR = re.compile(r"[+-]?\d{1,7}(?:[./]\d{1,7})?", re.ASCII)
FLOAT_T_MAX = 5000


def ruin_lattice(u: Number, d: Number, q: Number, T: int, w_ruin: Number,
                 w0: Number = 1, ruin_value: Number = 0, exact: bool = False) -> dict:
    """Exact dynamic program on the binomial lattice W_{n+1} = W_n * M,
    M = u w.p. q and d w.p. 1 - q, killed the first time W <= w_ruin.

    Returns price, ruin mass, survival probability, E[Y|S], survivor luck,
    median of Y_T (lower median, atoms included), P(Y_T < price) and the
    variance of Y_T. With ``exact=True`` pass ints, Fractions or decimal
    strings (not floats); every result except ``log_drift`` (irrational) is
    then an exact Fraction. Exact mode is limited to T <= 1000, to u, d, q
    with numerators and denominators at most 1e6, and to w_ruin, w0,
    ruin_value at most 1e30, to bound run time.
    Exact mode also refuses work above T^2 * (bits of u, d, q) = 1.2e7
    (at most about 10 s). Float mode works in log space and
    raises if the price leaves float range; its variance can be inf when
    the price is finite.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 1:
        raise ValueError("T must be a positive int")
    if exact:
        if T > EXACT_T_MAX:
            raise ValueError(f"exact mode supports T <= {EXACT_T_MAX}")
        raw = (u, d, q, w_ruin, w0, ruin_value)
        for x in raw:
            if isinstance(x, (float, bool)) or not isinstance(x, (int, Fraction, str)):
                raise ValueError("exact=True takes int, Fraction or short decimal strings such as '1.5' or '3/5'")
            if isinstance(x, str) and not _EXACT_STR.fullmatch(x.strip()):
                raise ValueError("exact string inputs must look like '1.5' or '3/5'")
            if isinstance(x, int) and abs(x) > 10**30:
                raise ValueError("exact integer inputs must be at most 1e30 in magnitude")
        try:
            u, d, q, w_ruin, w0, ruin_value = (Fraction(x.strip() if isinstance(x, str) else x) for x in raw)
        except (TypeError, ValueError, ZeroDivisionError):
            raise ValueError("exact inputs must be ints, Fractions or short decimal strings") from None
        if any(max(abs(x.numerator), x.denominator) > 10**6 for x in (u, d, q)):
            raise ValueError("exact u, d, q must have numerator and denominator <= 1e6")
        if any(max(abs(x.numerator), x.denominator) > 10**30 for x in (w_ruin, w0, ruin_value)):
            raise ValueError("exact w_ruin, w0, ruin_value must have numerator and denominator <= 1e30")
        bits = sum(x.numerator.bit_length() + x.denominator.bit_length() for x in (u, d, q))
        if T * T * bits > EXACT_WORK_MAX:
            raise ValueError("exact lattice too large; use float mode, a smaller T or simpler rationals")
    else:
        if T > FLOAT_T_MAX:
            raise ValueError(f"float mode supports T <= {FLOAT_T_MAX}")
        u, d, q, w_ruin, w0, ruin_value = (_finite(n, v) for n, v in
                                           (("u", u), ("d", d), ("q", q), ("w_ruin", w_ruin), ("w0", w0), ("ruin_value", ruin_value)))
    if not (u > 1 and 0 < d < 1):
        raise ValueError("require u > 1 and 0 < d < 1")
    if not (0 < q < 1):
        raise ValueError("q must lie in (0, 1)")
    if not (0 < w_ruin < w0):
        raise ValueError("require 0 < w_ruin < w0")

    if exact:
        upow = [u ** k for k in range(T + 1)]
        dpow = [d ** j for j in range(T + 1)]

        def wealth(k: int, n: int):
            return w0 * upow[k] * dpow[n - k]

        def dead(k: int, n: int) -> bool:
            return wealth(k, n) <= w_ruin
    else:
        lu, ld, lw0 = math.log(u), math.log(d), math.log(w0)
        lr = math.log(w_ruin) - lw0

        def logw(k: int, n: int) -> float:
            return lw0 + k * lu + (n - k) * ld

        def dead(k: int, n: int) -> bool:
            return k * lu + (n - k) * ld <= lr + 1e-12

    zero = Fraction(0) if exact else 0.0
    alive = {0: (Fraction(1) if exact else 1.0)}
    ruin = zero
    for n in range(1, T + 1):
        nxt: dict = {}
        for k, p in alive.items():
            for kk, pr in ((k + 1, q), (k, 1 - q)):
                mass = p * pr
                if dead(kk, n):
                    ruin += mass
                else:
                    nxt[kk] = nxt.get(kk, zero) + mass
        alive = nxt
    p_s = sum(alive.values(), zero)

    if exact:
        ys = [(wealth(k, T), p) for k, p in alive.items()]
        sum_y = sum((y * p for y, p in ys), zero)
        sum_y2 = sum((y * y * p for y, p in ys), zero)
        unstopped = w0 * (q * u + (1 - q) * d) ** T
    else:
        terms = [(logw(k, T), p) for k, p in alive.items() if p > 0]
        if terms and max(lw + math.log(p) for lw, p in terms) > 709.0:
            raise ValueError("price exceeds float range; use exact=True")
        ys = [(math.exp(lw) if lw < 709.0 else math.inf, p) for lw, p in terms]
        try:
            sum_y = math.fsum(math.exp(lw + math.log(p)) for lw, p in terms)
        except OverflowError:
            raise ValueError("price exceeds float range; use exact=True") from None
        sum_y2 = None  # float variance is computed centred, below
        lm = T * math.log(q * u + (1 - q) * d) + lw0
        unstopped = math.exp(lm) if lm < 709.0 else math.inf

    price = sum_y + ruin * ruin_value
    if exact:
        variance = sum_y2 + ruin * ruin_value * ruin_value - price * price
    elif any(y == math.inf for y, _ in ys):
        variance = math.inf
    else:
        try:
            variance = math.fsum(p * (y - price) ** 2 for y, p in ys) + ruin * (ruin_value - price) ** 2
        except OverflowError:
            variance = math.inf
    mean_s = (sum_y / p_s) if p_s > 0 else None
    dist = sorted([(ruin_value, ruin)] + ys, key=lambda t: t[0])
    half = Fraction(1, 2) if exact else 0.5
    cum = zero
    median = None
    for y, p in dist:
        cum += p
        if cum >= half:
            median = y
            break
    if exact:
        p_neg = sum((p for y, p in dist if y < price), zero)
    else:
        p_neg = min(1.0, math.fsum(p for y, p in dist if y < price))
    return {
        "price": price,
        "ruin_mass": ruin,
        "p_survive": p_s,
        "mean_given_survive": mean_s,
        "survivor_luck": (mean_s - ruin_value) * (1 - p_s) if mean_s is not None else None,
        "median": median,
        "p_negative_luck": p_neg,
        "variance": variance,
        "mean_multiplier": q * u + (1 - q) * d,
        "log_drift": float(q) * math.log(u) + float(1 - q) * math.log(d),
        "unstopped_mean": unstopped,
    }


# --------------------------------------------------------------------------
# 7.4: stake sizing
# --------------------------------------------------------------------------

def binary_growth(f: float, q: float, gain: float, loss: float) -> float:
    """g(f) = q ln(1 + gain f) + (1 - q) ln(1 - loss f), per step."""
    f = _finite("f", f)
    q = _prob("q", q)
    gain, loss = _finite("gain", gain), _finite("loss", loss)
    if not (gain > 0 and loss > 0):
        raise ValueError("gain and loss must be positive")
    if not (0 <= f and 1 - loss * f > 0):
        raise ValueError("stake must satisfy f >= 0 and 1 - loss*f > 0")
    return q * math.log1p(gain * f) + (1 - q) * math.log1p(-loss * f)


def binary_kelly(q: float, gain: float, loss: float, f_max: float = 1.0) -> float:
    """f* = q/loss - (1-q)/gain, clipped to [0, f_max]."""
    q = _prob("q", q, open_interval=True)
    gain, loss, f_max = _finite("gain", gain), _finite("loss", loss), _finite("f_max", f_max)
    if not (gain > 0 and loss > 0 and f_max > 0):
        raise ValueError("gain, loss and f_max must be positive")
    return min(max(q / loss - (1 - q) / gain, 0.0), f_max)


def discrete_growth(f: float, returns: Sequence[float], probs: Sequence[float]) -> float:
    """E[ln(1 + f R)] for a finite return distribution."""
    _check_dist(returns, probs)
    f = _finite("f", f)
    total = 0.0
    for r, p in zip(returns, probs):
        m = 1 + f * r
        if m <= 0:
            if p > 0:
                return -math.inf
            continue
        total += p * math.log(m)
    return total


def discrete_kelly(returns: Sequence[float], probs: Sequence[float], f_max: float = 1.0, tol: float = 1e-12) -> float:
    """Growth-optimal constant fraction in [0, f_max] by bisection on g'(f),
    which is strictly decreasing where defined. Respects 1 + f R > 0."""
    _check_dist(returns, probs)
    f_max, tol = _finite("f_max", f_max), _finite("tol", tol)
    if f_max <= 0 or tol <= 0:
        raise ValueError("f_max and tol must be positive")
    r_min = min(r for r, p in zip(returns, probs) if p > 0)
    hi = f_max if r_min >= 0 else min(f_max, (-1.0 / r_min) * (1 - 1e-12))

    def dg(f: float) -> float:
        return sum(p * r / (1 + f * r) for r, p in zip(returns, probs) if p > 0)

    if dg(0.0) <= 0:
        return 0.0
    if dg(hi) >= 0:
        return hi
    lo = 0.0
    for _ in range(4096):
        mid = 0.5 * (lo + hi)
        if hi - lo <= tol or not (lo < mid < hi):
            break
        if dg(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def inverse_wealth_drift(f: float, returns: Sequence[float], probs: Sequence[float]) -> float:
    """E[1/M] with M = 1 + f R. Equals 1 at an interior Kelly optimum, which
    makes 1/W a martingale and gives P(ever W <= x W0) <= x (Ville)."""
    _check_dist(returns, probs)
    f = _finite("f", f)
    out = 0.0
    for r, p in zip(returns, probs):
        m = 1 + f * r
        if m <= 0:
            raise ValueError("stake reaches zero wealth")
        out += p / m
    return out


def _check_dist(returns: Sequence[float], probs: Sequence[float]) -> None:
    if len(returns) != len(probs) or not returns:
        raise ValueError("returns and probs must be non-empty and equal length")
    if any((not math.isfinite(p)) or p < 0 for p in probs):
        raise ValueError("probabilities must be finite and non-negative")
    if any(not math.isfinite(r) for r in returns):
        raise ValueError("returns must be finite")
    if not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to 1")


# --------------------------------------------------------------------------
# Rule 6: predictability horizon
# --------------------------------------------------------------------------

def predictability_horizon(lyapunov: float, tolerance: float, resolution: float, kick_scale: float = 0.0) -> float:
    """t_p ~ ln(Delta / max(eps_obs, sigma)) / lambda_L (first order)."""
    lam = _finite("lyapunov", lyapunov)
    tolerance = _finite("tolerance", tolerance)
    if lam <= 0:
        raise ValueError("horizon formula applies to a positive Lyapunov exponent")
    floor = max(_finite("resolution", resolution), _finite("kick_scale", kick_scale))
    if not (0 < floor < tolerance):
        raise ValueError("require 0 < max(resolution, kick_scale) < tolerance")
    return math.log(tolerance / floor) / lam


# --------------------------------------------------------------------------
# 7.2: anytime-valid calibration alarm
# --------------------------------------------------------------------------

class AnytimeAlarm:
    """Mixture betting martingale K_n = sum_j w_j prod_k (1 + lambda_j x_k).

    Under H0: E[x_k | past] = 0 with x_k in [-1, 1], K_n is a nonnegative
    martingale with K_0 = 1, so P(exists n: K_n >= 1/alpha) <= alpha (Ville).
    Residuals must be resolved one at a time (one-step or non-overlapping
    horizons) and scaled into [-1, 1] by a bound fixed in advance.
    """

    DEFAULT_LAMBDAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

    def __init__(self, alpha: float = 0.05, lambdas: Iterable[float] = DEFAULT_LAMBDAS, two_sided: bool = True):
        self.alpha = _prob("alpha", alpha, open_interval=True)
        lams = [_finite("lambda", l) for l in lambdas]
        if not lams:
            raise ValueError("need at least one bet size")
        if any(not (0 < abs(l) < 1) for l in lams):
            raise ValueError("every bet size must satisfy 0 < |lambda| < 1")
        if two_sided:
            lams = lams + [-l for l in lams]
        self.lambdas: Tuple[float, ...] = tuple(lams)
        self._log_w = [0.0] * len(self.lambdas)
        self._log_threshold = math.log(1.0 / self.alpha)
        self.n = 0
        self.alarm_step: Optional[int] = None

    @property
    def log_wealth(self) -> float:
        m = max(self._log_w)
        return m + math.log(sum(math.exp(v - m) for v in self._log_w) / len(self._log_w))

    @property
    def alarmed(self) -> bool:
        return self.alarm_step is not None

    def update(self, x: float) -> bool:
        x = _finite("x", x)
        if not (-1.0 <= x <= 1.0):
            raise ValueError("residual must be scaled into [-1, 1] before updating")
        for j, l in enumerate(self.lambdas):
            self._log_w[j] += math.log1p(l * x)
        self.n += 1
        if self.alarm_step is None and self.log_wealth >= self._log_threshold:
            self.alarm_step = self.n
        return self.alarmed


def randomized_pit(cdf_below: float, cdf_at: float, v: float) -> float:
    """u = F(Y-) + V (F(Y) - F(Y-)), V ~ U(0,1) independent of everything.

    Exactly U(0,1) when F is the true conditional law, atoms included.
    Feed ``2u - 1`` to AnytimeAlarm to test the whole forecast law."""
    a, b, v = _prob("cdf_below", cdf_below), _prob("cdf_at", cdf_at), _prob("v", v)
    if a > b:
        raise ValueError("cdf_below must not exceed cdf_at")
    return a + v * (b - a)


# --------------------------------------------------------------------------
# 7.3: selection
# --------------------------------------------------------------------------

def expected_max_std_normal(n: int, steps: int = 20000) -> float:
    """E[max of n iid N(0,1)] by Simpson's rule on [-12, 12].
    Selection inflation for equal true values and N(0, sigma^2) errors is
    sigma times this number."""
    if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= 10**6):
        raise ValueError("n must be an int in [1, 1e6]")
    if not isinstance(steps, int) or isinstance(steps, bool) or not (2 <= steps <= 10**6):
        raise ValueError("steps must be an int in [2, 1e6]")
    if steps % 2:
        steps += 1
    a, b = -12.0, 12.0
    h = (b - a) / steps
    s = 0.0
    for i in range(steps + 1):
        x = a + i * h
        phi = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
        val = x * n * phi * Phi ** (n - 1)
        s += val * (1 if i in (0, steps) else (4 if i % 2 else 2))
    return s * h / 3


# --------------------------------------------------------------------------
# 7.8: credal sets and fuzzy ruin
# --------------------------------------------------------------------------

def credal_price(prices_under_models: Sequence[float]) -> Tuple[float, float]:
    """[inf, sup] of E_mu[Y_T] over a finite set of plausible measures."""
    vals = [_finite("price", p) for p in prices_under_models]
    if not vals:
        raise ValueError("need at least one model")
    return min(vals), max(vals)


def credal_luck(outcome: float, prices_under_models: Sequence[float]) -> Tuple[float, float]:
    """[Y - sup price, Y - inf price]."""
    lo, hi = credal_price(prices_under_models)
    y = _finite("outcome", outcome)
    return y - hi, y - lo


def fuzzy_killing_price(paths: Sequence[Tuple[Sequence[float], float]], weights: Optional[Sequence[float]], ruin_value: float) -> dict:
    """Price with a graded ruin set read as a per-step killing probability.

    ``paths`` holds (memberships rho_R(x_u) for u = t+1..T, unkilled yield).
    ``weights`` are path probabilities (None means equal Monte Carlo weights).
    Returns price = E[Pi Y°] + y_R (1 - E[Pi]) and ruin mass 1 - E[Pi]."""
    paths = list(paths)
    if not paths:
        raise ValueError("need at least one path")
    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    weights = list(weights)
    _check_dist([0.0] * len(weights), weights)
    if len(weights) != len(paths):
        raise ValueError("one weight per path")
    y_r = _finite("ruin_value", ruin_value)
    e_pi = 0.0
    e_pi_y = 0.0
    for (memb, y), w in zip(paths, weights):
        pi = 1.0
        for r in memb:
            pi *= 1.0 - _prob("membership", r)
        e_pi += w * pi
        e_pi_y += w * pi * _finite("yield", y)
    return {"price": e_pi_y + y_r * (1 - e_pi), "ruin_mass": 1 - e_pi}


# --------------------------------------------------------------------------
# §6 and 7.6: attribution and conservation
# --------------------------------------------------------------------------

def attribution_ledger(outcome: float, reference_price: float, initial_price: float, current_price: float,
                       exact: bool = False) -> dict:
    """Telescoping split of Y_T - reference into head start (A),
    price change since t=0 (B: luck turned into power, plus any other map
    change), and this period's luck (C). Selection (D) is a cohort quantity:
    use cohort_summary. With ``exact=True`` the parts are Fractions of the
    input floats and sum exactly to the total; otherwise each part carries
    one float rounding."""
    y, ref, p0, pt = (_finite(n, v) for n, v in (("outcome", outcome), ("reference_price", reference_price),
                                                 ("initial_price", initial_price), ("current_price", current_price)))
    if exact:
        y, ref, p0, pt = (Fraction(v) for v in (y, ref, p0, pt))
    return {
        "total_vs_reference": y - ref,
        "A_head_start": p0 - ref,
        "B_price_change": pt - p0,
        "C_luck": y - pt,
    }


def sum_of_luck(outcomes: Sequence[float], prices: Sequence[float]) -> float:
    """sum_i (Y_i - E[Y_i | I_i]). Zero on every path exactly when
    sum_i Y_i = sum_i E[Y_i | I_i] (7.6)."""
    if len(outcomes) != len(prices) or not outcomes:
        raise ValueError("need equal-length, non-empty sequences")
    total = sum((Fraction(_finite("outcome", y)) - Fraction(_finite("price", p)) for y, p in zip(outcomes, prices)), Fraction(0))
    return float(total)  # one rounding: zero exactly when the sums are equal
