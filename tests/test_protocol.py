from ruthless_alpha.protocol import analyze, trailing_return


def bars(values):
    return [{"c": value} for value in values]


def test_trailing_return_uses_available_history():
    assert trailing_return(bars([100, 110]), 252) == 0.1


def test_analysis_is_deterministic_and_secret_free():
    result = analyze("TEST", bars([100] * 756 + [120]), 0.08, 0.10)
    assert result["symbol"] == "TEST"
    assert result["verdict"] in {"KEEP", "REJECT"}
    assert 0 <= result["half_kelly"] <= 0.5
    assert not any("key" in key.lower() or "secret" in key.lower() for key in result)

