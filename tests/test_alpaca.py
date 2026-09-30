import io
import json
import urllib.parse

from ruthless_alpha import alpaca


def test_daily_bars_follows_page_tokens(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test")
    pages = [
        {"bars": {"AAA": [{"c": 1}], "BBB": [{"c": 2}]}, "next_page_token": "p2"},
        {"bars": {"BBB": [{"c": 3}]}, "next_page_token": None},
    ]
    tokens = []

    def fake_urlopen(request, timeout):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        tokens.append(query.get("page_token", [None])[0])
        return io.BytesIO(json.dumps(pages[len(tokens) - 1]).encode())

    monkeypatch.setattr(alpaca.urllib.request, "urlopen", fake_urlopen)
    result = alpaca.daily_bars(["AAA", "BBB"])
    assert tokens == [None, "p2"]
    assert result == {"AAA": [{"c": 1}], "BBB": [{"c": 2}, {"c": 3}]}
