# Ruthless Alpha Protocol

A repeatable, ruin-aware stock analysis protocol that combines adjusted Alpaca market data with the Ruthless Luck Algorithm v1.6.

The protocol makes its scenario law explicit, computes a growth-optimal fraction with `discrete_kelly`, applies a conservative half-Kelly decision cap, and reserves realized luck for a completed scoring horizon. It does not confuse a forecast with an outcome.

## Safety and scope

- Read-only Alpaca Market Data API; no order endpoint is used.
- Credentials are read from environment variables and never written to reports.
- Generated reports are ignored by Git by default.
- This is research software, not individualized investment, tax, or legal advice.
- Scenario probabilities are assumptions. They are not measured default probabilities.
- Standalone Kelly fractions do not account for cross-position correlation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
# Export the two APCA variables from your secret manager; do not commit .env.
ruthless-alpha config/watchlist.example.json
pytest -q
```

## Protocol

1. Define the yield, horizon, ruin state, and information timestamp.
2. Fetch split/dividend-adjusted daily bars from Alpaca IEX.
3. Set the upside state to the larger of one-year adjusted return and three-year CAGR.
4. Set the carry state to the entered cash yield and the kill state to −40% by default.
5. Disclose the probability of each state; the example uses 25% upside and assigns the remainder after the kill probability to carry.
6. Compute full Kelly with the supplied v1.6 reference implementation; use half Kelly as the decision cap.
7. At the horizon, create a schema-valid `LuckReport`: outcome minus the original price is luck. Include ruined paths.
8. Re-run whenever payout policy, leverage, sponsor incentives, or the business map materially changes.

## Reproducibility

The original `ruthless_luck.py`, its conformance suite, and `luck-report.schema.json` are included. The reference lattice smoke check should produce price `131.39`, ruin mass `0.69572`, and survival probability `0.30428`.

## Signing

All repository commits are SSH-signed. Verify with `git log --show-signature` after configuring an allowed signers file or through GitHub's verified-commit UI.
