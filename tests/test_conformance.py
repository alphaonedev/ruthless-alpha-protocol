"""Conformance suite: every number published in spec v1.6 is reproduced here.

Run with:  python -m pytest -q test_conformance.py
Stochastic tests use fixed seeds and tolerances sized from their own
standard errors (at least 4 SE), so they are deterministic and not flaky.
"""

import json
import math
import random
from fractions import Fraction as F

import pytest

from ruthless_alpha import ruthless_luck as rl


# ---------------- §5: exact lattice ----------------

@pytest.fixture(scope="module")
def lattice_exact():
    return rl.ruin_lattice(F(3, 2), F(3, 5), F(1, 2), 100, F(1, 100), exact=True)


def test_lattice_published_values(lattice_exact):
    r = lattice_exact
    assert r["ruin_mass"] + r["p_survive"] == 1
    assert round(float(r["price"]), 2) == 131.39
    assert round(float(r["ruin_mass"]), 5) == 0.69572
    assert round(float(r["p_survive"]), 5) == 0.30428
    assert round(float(r["mean_given_survive"]), 2) == 431.82
    assert round(float(r["survivor_luck"]), 2) == 300.43
    assert r["median"] == 0
    assert round(float(1 - r["p_negative_luck"]), 5) == 0.01044
    assert r["mean_multiplier"] == F(21, 20)
    assert round(r["log_drift"], 5) == -0.05268
    assert round(float(r["unstopped_mean"]), 2) == 131.50


def test_rule4_identity_exact(lattice_exact):
    r = lattice_exact
    assert r["survivor_luck"] == r["mean_given_survive"] - r["price"]
    assert r["survivor_luck"] == rl.survivor_luck(r["p_survive"], r["mean_given_survive"], 0)
    assert r["price"] == rl.price_from_parts(r["p_survive"], r["mean_given_survive"], 0)


def test_float_lattice_matches_exact(lattice_exact):
    f = rl.ruin_lattice(1.5, 0.6, 0.5, 100, 0.01)
    for k in ("price", "ruin_mass", "p_survive", "mean_given_survive", "survivor_luck", "p_negative_luck"):
        assert math.isclose(f[k], float(lattice_exact[k]), rel_tol=1e-10), k


def test_62_ups_threshold(lattice_exact):
    price = lattice_exact["price"]
    assert F(3, 2) ** 61 * F(3, 5) ** 39 < price < F(3, 2) ** 62 * F(3, 5) ** 38


def test_median_unstopped():
    # With a ruin line far below reach, the lattice median is the 50-50 path: 0.9**50 = 0.005154.
    r = rl.ruin_lattice(F(3, 2), F(3, 5), F(1, 2), 100, F(1, 10**30), exact=True)
    assert r["ruin_mass"] == 0
    assert r["median"] == F(9, 10) ** 50
    rf = rl.ruin_lattice(1.5, 0.6, 0.5, 100, 1e-30)
    assert rf["ruin_mass"] == 0
    assert rf["median"] == pytest.approx(0.9 ** 50, rel=1e-12)
    assert round(float(r["median"]), 6) == 0.005154


def test_sd_of_yield(lattice_exact):
    # SD of Y_100 in §5: 603,289
    assert round(math.sqrt(float(lattice_exact["variance"]))) == 603289


# ---------------- 7.4 table ----------------

@pytest.mark.parametrize("f,price,ruin,median,pneg", [
    (0.10, 1.647, 0.0, 1.49, 0.618),
    (0.25, 3.463, 0.0, 1.86, 0.691),
    (0.50, 11.81, 0.0352, 1.00, 0.864),
    (0.75, 39.69, 0.3243, 0.148, 0.956),
    (1.00, 131.39, 0.6957, 0.0, 0.990),
])
def test_stake_table(f, price, ruin, median, pneg):
    r = rl.ruin_lattice(1 + 0.5 * f, 1 - 0.4 * f, 0.5, 100, 0.01)
    assert math.isclose(r["price"], price, rel_tol=5e-4)
    assert abs(r["ruin_mass"] - ruin) < 5e-5
    assert abs(r["median"] - median) < 5e-3
    assert abs(r["p_negative_luck"] - pneg) < 5e-4


def test_price_monotone_to_all_in():
    prices = [rl.ruin_lattice(1 + 0.5 * f, 1 - 0.4 * f, 0.5, 100, 0.01)["price"] for f in [i / 20 for i in range(1, 21)]]
    assert all(a < b for a, b in zip(prices, prices[1:]))


# ---------------- 7.4 Kelly ----------------

def test_kelly_numbers():
    f = rl.binary_kelly(0.5, 0.5, 0.4)
    assert f == pytest.approx(0.25)
    assert rl.binary_growth(0.25, 0.5, 0.5, 0.4) == pytest.approx(0.00621126, abs=1e-8)
    assert rl.binary_growth(0.5, 0.5, 0.5, 0.4) == pytest.approx(0.0, abs=1e-15)
    assert rl.binary_growth(1.0, 0.5, 0.5, 0.4) == pytest.approx(-0.0526803, abs=1e-7)
    half = rl.binary_growth(0.125, 0.5, 0.5, 0.4)
    assert half == pytest.approx(0.00466566, abs=1e-8)
    assert round(half / rl.binary_growth(0.25, 0.5, 0.5, 0.4), 2) == 0.75


def test_growth_closed_form_and_symmetry():
    for i in range(0, 101):
        f = i / 100
        g = rl.binary_growth(f, 0.5, 0.5, 0.4)
        assert g == pytest.approx(0.5 * math.log(1 + 0.1 * f - 0.2 * f * f), abs=1e-14)
    for dlt in (0.05, 0.1, 0.2):
        assert rl.binary_growth(0.25 - dlt, 0.5, 0.5, 0.4) == pytest.approx(rl.binary_growth(0.25 + dlt, 0.5, 0.5, 0.4), abs=1e-14)


def test_discrete_kelly_agrees_with_binary():
    f = rl.discrete_kelly([0.5, -0.4], [0.5, 0.5])
    assert f == pytest.approx(0.25, abs=1e-9)


def test_inverse_wealth_martingale_at_kelly():
    assert rl.inverse_wealth_drift(0.25, [0.5, -0.4], [0.5, 0.5]) == pytest.approx(1.0, abs=1e-15)
    assert rl.inverse_wealth_drift(0.5, [0.5, -0.4], [0.5, 0.5]) == pytest.approx(1.025, abs=1e-12)


def test_kelly_ruin_cap_deterministic():
    # At the true Kelly stake, P(ever W <= 0.01) <= 0.01 (1/W is a martingale; Ville). Exact lattice to T = 2000.
    f = rl.binary_kelly(0.5, 0.5, 0.4)
    assert rl.ruin_lattice(1 + 0.5 * f, 1 - 0.4 * f, 0.5, 2000, 0.01)["ruin_mass"] <= 0.01


# ---------------- Rule 6 / §4 ----------------

def test_horizon():
    assert rl.predictability_horizon(math.log(2), 0.1, 1e-10) == pytest.approx(29.8974, abs=1e-4)
    assert math.log(10) / math.log(2) == pytest.approx(3.32193, abs=1e-5)
    assert rl.predictability_horizon(math.log(2), 0.1, 1e-10, kick_scale=1e-6) == pytest.approx(math.log(1e5) / math.log(2))


def test_logistic_pair_crosses_at_31():
    a, b = 0.3, 0.3 + 1e-10
    for t in range(0, 60):
        if abs(a - b) > 0.1:
            break
        a, b = 4 * a * (1 - a), 4 * b * (1 - b)
    assert t == 31
    assert 2 ** 29 * 1e-10 < 0.1 < 2 ** 30 * 1e-10


# ---------------- 7.2 alarm ----------------

def _run_alarm(bias, n_steps, seed):
    rng = random.Random(seed)
    a = rl.AnytimeAlarm(alpha=0.05)
    for _ in range(n_steps):
        a.update(1.0 if rng.random() < (1 + bias) / 2 else -1.0)
        if a.alarmed:
            break
    return a.alarmed


def test_alarm_false_alarm_rate_bounded():
    runs = 400
    rate = sum(_run_alarm(0.0, 3000, 1000 + s) for s in range(runs)) / runs
    assert rate <= 0.05 + 4 * math.sqrt(0.05 * 0.95 / runs)


def test_alarm_detects_large_bias():
    runs = 100
    rate = sum(_run_alarm(0.1, 3000, 5000 + s) for s in range(runs)) / runs
    # 100 seeded runs; at bias 0.1 the per-run detection rate within 3,000 steps is about 0.99.
    assert rate >= 0.95


def test_alarm_rejects_unscaled_input():
    a = rl.AnytimeAlarm()
    with pytest.raises(ValueError):
        a.update(1.5)
    with pytest.raises(ValueError):
        a.update(float("nan"))
    with pytest.raises(ValueError):
        rl.AnytimeAlarm(lambdas=[1.0])


def test_alarm_martingale_mean_exact():
    # E[K_1] = 1 exactly under H0 for x = +-1 equiprobable.
    up, dn = rl.AnytimeAlarm(), rl.AnytimeAlarm()
    up.update(1.0)
    dn.update(-1.0)
    assert 0.5 * math.exp(up.log_wealth) + 0.5 * math.exp(dn.log_wealth) == pytest.approx(1.0, abs=1e-15)


def test_randomized_pit_uniform_with_atom():
    # Law: Y = 0 w.p. 0.7 (ruin atom), else Uniform(0, 1]. Under the true law u ~ U(0, 1).
    rng = random.Random(3)
    us = []
    for _ in range(40000):
        if rng.random() < 0.7:
            u = rl.randomized_pit(0.0, 0.7, rng.random())
        else:
            y = rng.random()
            c = 0.7 + 0.3 * y
            u = rl.randomized_pit(c, c, rng.random())
        us.append(u)
    mean = sum(us) / len(us)
    assert abs(mean - 0.5) < 4 * math.sqrt(1 / 12 / len(us))
    below = sum(u < 0.25 for u in us) / len(us)
    assert abs(below - 0.25) < 4 * math.sqrt(0.25 * 0.75 / len(us))


# ---------------- 7.3 selection ----------------

def test_expected_max_normals():
    assert rl.expected_max_std_normal(1) == pytest.approx(0.0, abs=1e-9)
    assert rl.expected_max_std_normal(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)
    assert rl.expected_max_std_normal(10) == pytest.approx(1.538753, abs=1e-5)


# ---------------- 7.6 / 7.8 / §6 ----------------

def test_constant_sum_conservation():
    # Two players, shared information: pot of 10 split by a coin, prices are 5 and 5.
    for y1 in (0.0, 3.0, 10.0):
        assert rl.sum_of_luck([y1, 10 - y1], [5.0, 5.0]) == 0.0
    # Private information: player 1 knows the coin, player 2 does not.
    assert rl.sum_of_luck([10.0, 0.0], [10.0, 5.0]) == -5.0


def test_credal_interval():
    lo, hi = rl.credal_price([3.0, 5.0, 4.0])
    assert (lo, hi) == (3.0, 5.0)
    assert rl.credal_luck(6.0, [3.0, 5.0, 4.0]) == (1.0, 3.0)


def test_fuzzy_crisp_recovers_step4():
    # Crisp memberships reproduce p E[Y|S] + (1-p) y_R.
    paths = [([0, 0, 0], 8.0), ([0, 1, 0], 99.0), ([0, 0, 0], 2.0), ([1, 0, 0], 50.0)]
    out = rl.fuzzy_killing_price(paths, None, ruin_value=-1.0)
    assert out["ruin_mass"] == 0.5
    assert out["price"] == pytest.approx(0.5 * 5.0 + 0.5 * (-1.0))


def test_fuzzy_one_step_is_zadeh():
    paths = [([0.2], 1.0), ([0.6], 1.0)]
    out = rl.fuzzy_killing_price(paths, [0.5, 0.5], ruin_value=0.0)
    assert out["ruin_mass"] == pytest.approx(0.4)  # E[rho]


def test_attribution_telescopes():
    led = rl.attribution_ledger(outcome=12.0, reference_price=4.0, initial_price=6.0, current_price=9.0)
    assert led["A_head_start"] + led["B_price_change"] + led["C_luck"] == led["total_vs_reference"]


# ---------------- report format ----------------

def _report(**kw):
    base = dict(agent_id="a1", t=0.0, horizon=100.0, price=131.39, outcome=0.0, ruin_mass=0.6957,
                survived=False, ruin_value=0.0, model_id="lattice-v1")
    base.update(kw)
    return rl.LuckReport(**base)


def test_report_roundtrip_and_luck_derived():
    r = _report()
    s = r.to_json()
    back = rl.LuckReport.from_json(s)
    assert back == r
    assert json.loads(s)["luck"] == pytest.approx(-131.39)


@pytest.mark.parametrize("bad", [
    dict(outcome=5.0),                          # absorbed path not scored at ruin value
    dict(ruin_mass=0.0),                        # absorbed although priced impossible
    dict(survived=True, outcome=10.0, ruin_mass=1.0),
    dict(horizon=0.0),
    dict(model_id=""),
    dict(ruin_mass=1.5),
    dict(price=float("inf")),
])
def test_report_rejects_invalid(bad):
    with pytest.raises(ValueError):
        _report(**bad)


def test_report_rejects_tampered_luck_and_extra_fields():
    d = _report().to_dict()
    d["luck"] = 0.0
    with pytest.raises(ValueError):
        rl.LuckReport.from_dict(d)
    d = _report().to_dict()
    d["bonus"] = 1
    with pytest.raises(ValueError):
        rl.LuckReport.from_dict(d)
    s = _report().to_json().replace('"price": 131.39', '"price": NaN')
    assert "NaN" in s
    with pytest.raises(ValueError, match="non-finite"):
        rl.LuckReport.from_json(s)


def test_cohort_summary_reports_selection():
    reps = [_report(agent_id=f"a{i}") for i in range(7)] + [
        _report(agent_id=f"s{i}", survived=True, outcome=431.82, ruin_mass=0.6957) for i in range(3)]
    s = rl.cohort_summary(reps)
    assert s["n"] == 10 and s["n_survived"] == 3
    assert s["mean_luck_survivors"] > s["mean_luck_all"]
    assert s["selection_gap"] == pytest.approx(s["mean_luck_survivors"] - s["mean_luck_all"])


def test_schema_file_matches_dataclass():
    import dataclasses
    import pathlib
    schema = json.loads((pathlib.Path(__file__).parent.parent / "src" / "ruthless_alpha" / "luck-report.schema.json").read_text())
    fields = {f.name for f in dataclasses.fields(rl.LuckReport)}
    assert set(schema["required"]) == fields
    assert set(schema["properties"]) == fields | {"luck", "$schema"}
    assert schema["additionalProperties"] is False
    assert schema["$id"] == rl.REPORT_SCHEMA_ID
    assert schema["properties"]["$schema"]["const"] == rl.REPORT_SCHEMA_ID
    for f in ("t", "horizon", "price", "outcome", "ruin_mass", "ruin_value", "luck"):
        assert schema["properties"][f]["type"] == "number"


# ---------------- untrusted input and edge cases (code review) ----------------

@pytest.mark.parametrize("field,value", [
    ("price", "131.39"), ("price", True), ("t", False), ("ruin_mass", True),
    ("price", 10**400), ("price", {"a": 1}), ("price", [1]), ("price", None), ("agent_id", 5),
    ("$schema", {"x": 1}), ("$schema", "https://evil"), ("luck", "-131.39"), ("luck", 10**400)])
def test_from_dict_rejects_type_confusion(field, value):
    d = _report().to_dict()
    d[field] = value
    with pytest.raises(ValueError):
        rl.LuckReport.from_dict(d)


def test_from_json_raises_only_valueerror():
    good = _report().to_json()
    for s in ["[" * 5000 + "]" * 5000,
              good.replace('"price": 131.39', '"price": 1' + "0" * 400),
              good.replace('"price": 131.39', '"price": {"a": 1}'),
              good.replace('"price": 131.39', '"price": true'),
              '{"price": 1.0, ' + good[1:],
              good + " " * 70000]:
        with pytest.raises(ValueError):
            rl.LuckReport.from_json(s)
    for bad in (None, 5, [good]):
        with pytest.raises(ValueError):
            rl.LuckReport.from_json(bad)
    assert rl.LuckReport.from_json(good.encode()) == _report()


def test_discrete_kelly_edges_and_termination():
    assert rl.discrete_kelly([0.1, -0.2], [0.5, 0.5]) == 0.0
    assert rl.discrete_kelly([0.5, 0.1], [0.5, 0.5], f_max=0.7) == 0.7
    assert rl.discrete_kelly([0.5, -0.4], [0.5, 0.5], f_max=0.1) == 0.1
    assert rl.discrete_kelly([1.0, -1.0], [0.6, 0.4]) == pytest.approx(0.2, abs=1e-9)
    assert rl.discrete_kelly([0.001, -1e-5], [0.5, 0.5], f_max=1e6) == pytest.approx(49500.0, rel=1e-9)
    for kw in (dict(tol=0.0), dict(tol=-1.0), dict(f_max=float("nan")), dict(f_max=-1.0)):
        with pytest.raises(ValueError):
            rl.discrete_kelly([0.5, -0.4], [0.5, 0.5], **kw)


def test_lattice_barrier_inclusive_exact_types_and_large_T():
    e = rl.ruin_lattice(F(2), F(1, 2), F(1, 2), 2, F(1, 4), exact=True)
    assert e["ruin_mass"] == F(1, 4)                       # W == w_R is ruin
    assert rl.ruin_lattice(2.0, 0.5, 0.5, 2, 0.25)["ruin_mass"] == 0.25
    assert all(isinstance(v, F) for k, v in e.items() if k != "log_drift" and v is not None)
    assert math.isfinite(rl.ruin_lattice(1.5, 0.6, 0.5, 1800, 0.01)["price"])
    with pytest.raises(ValueError):
        rl.ruin_lattice(1.5, 0.6, 0.5, 10, 0.01, exact=True)  # float in exact mode
    with pytest.raises(ValueError):
        rl.ruin_lattice(F(3, 2), F(3, 5), F(1, 2), 1001, F(1, 100), exact=True)
    with pytest.raises(ValueError):
        rl.ruin_lattice(F(3, 2), F(3, 5), F(1, 3) + F(1, 10**300), 10, F(1, 100), exact=True)
    for bad in ("1e50000000", "1/0", object()):
        with pytest.raises(ValueError):
            rl.ruin_lattice(bad, F(3, 5), F(1, 2), 10, F(1, 100), exact=True)
    with pytest.raises(ValueError):  # work cap
        rl.ruin_lattice(F(999983, 999979), F(999979, 999983), F(499979, 999983), 1000, F(1, 999983), exact=True)
    with pytest.raises(ValueError):  # price beyond float range surfaces as ValueError
        rl.ruin_lattice(1.5, 0.6, 0.5, 5, 1e300, w0=1.7e308)
    r = rl.ruin_lattice(1.5, 0.6, 0.5, 1800, 0.01)
    assert r["p_negative_luck"] <= 1.0


def test_float_lattice_large_T_matches_exact():
    ex = rl.ruin_lattice(F(3, 2), F(3, 5), F(1, 2), 400, F(1, 100), exact=True)
    fl = rl.ruin_lattice(1.5, 0.6, 0.5, 400, 0.01)
    assert math.isclose(fl["price"], float(ex["price"]), rel_tol=1e-10)
    assert math.isclose(fl["ruin_mass"], float(ex["ruin_mass"]), rel_tol=1e-10)
    assert math.isclose(fl["variance"], float(ex["variance"]), rel_tol=1e-9)
    small = rl.ruin_lattice(1.00001, 0.99999, 0.5, 3, 0.5)
    exs = rl.ruin_lattice(F("1.00001"), F("0.99999"), F(1, 2), 3, F(1, 2), exact=True)
    assert math.isclose(small["variance"], float(exs["variance"]), rel_tol=1e-9)


def test_alarm_martingale_exact_asymmetric_and_latch():
    import itertools
    law = [(-0.5, 2 / 3), (1.0, 1 / 3)]
    for n in (2, 3, 4):
        E = 0.0
        for seq in itertools.product(law, repeat=n):
            a = rl.AnytimeAlarm()
            p = 1.0
            for x, px in seq:
                a.update(x)
                p *= px
            E += p * math.exp(a.log_wealth)
        assert E == pytest.approx(1.0, abs=1e-12)
    a = rl.AnytimeAlarm()
    while not a.alarmed:
        a.update(1.0)
    step = a.alarm_step
    for _ in range(100):
        a.update(0.0)
    assert a.alarmed and a.alarm_step == step


def test_fuzzy_multistep_is_product():
    assert rl.fuzzy_killing_price([([0.5, 0.5], 4.0)], None, 0.0) == {"price": 1.0, "ruin_mass": 0.75}


def test_expected_max_large_n_and_steps_validation():
    assert rl.expected_max_std_normal(10**6) == pytest.approx(4.8628974862, abs=1e-8)
    for bad in (0, -4, 2.0, 10**9):
        with pytest.raises(ValueError):
            rl.expected_max_std_normal(10, steps=bad)


def test_sum_of_luck_and_ledger_exact_on_floats():
    assert rl.sum_of_luck([1e16, 7, -7], [1e16, 1e-17, 1e-17]) == -2e-17
    led = rl.attribution_ledger(0.1, 0.7, 0.3, 0.2, exact=True)
    assert led["A_head_start"] + led["B_price_change"] + led["C_luck"] == led["total_vs_reference"]


def test_validation_of_nan_inputs():
    for call in (lambda: rl.binary_growth(0.1, float("nan"), 0.5, 0.4),
                 lambda: rl.binary_growth(0.1, 0.5, -0.5, 0.4),
                 lambda: rl.binary_kelly(0.5, 0.5, 0.4, f_max=-1),
                 lambda: rl.price_from_parts(0.5, float("nan"), 0.0),
                 lambda: rl.discrete_growth(float("nan"), [0.5, -0.4], [0.5, 0.5]),
                 lambda: rl.inverse_wealth_drift(float("nan"), [0.5, -0.4], [0.5, 0.5]),
                 lambda: rl.predictability_horizon(0.69, float("inf"), 1e-10)):
        with pytest.raises(ValueError):
            call()
